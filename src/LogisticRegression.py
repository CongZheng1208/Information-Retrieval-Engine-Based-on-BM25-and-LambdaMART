import pandas as pd
import numpy as np
import pandas as pd
import EvaluationMetricGenerator as t1
import matplotlib.pyplot as plt
import re
import nltk.data
import warnings
import logging

from tqdm import tqdm
from gensim.models import word2vec
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

warnings.filterwarnings(action = 'ignore')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

#Function A:The following functions are used to implement Word2Vec-based Embedding
#==========================================================================================================#
def GenerateWordlist( raw_review,remove_stopwords=False ):
    '''
    Function to convert a raw review to a string of words
    '''
    # Remove HTML and non-letters
    review_text = BeautifulSoup(raw_review).get_text()     
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 

    # Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    if remove_stopwords:
        stops = set(stopwords.words("english"))                  
        words = [w for w in words if not w in stops]   
    
    return words

def GenerateSentences( review, tokenizer, remove_stopwords=False ):
    '''
    Function to split a review into parsed sentences.
    '''
    # Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())

    sentences = []
    for raw_sentence in raw_sentences:

        if len(raw_sentence) > 0:
            sentences.append( GenerateWordlist( raw_sentence,remove_stopwords ))

    return sentences

def GenerateFeatureVec(words, model, num_features):
    '''
    Function to average all of the word vectors in a given
    '''
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.0
    index2word_set = set(model.wv.index_to_key)

    # Loop over each word in the review and, if it is in the model's vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model.wv[word])
    
    featureVec = np.divide(featureVec,nwords)
    return featureVec

def GetAvgFeatureVecs(reviews, model, num_features):

    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")

    for review in reviews:

        if counter%1000 == 0:
            print("Review %d of %d" % (counter, len(reviews)))
        reviewFeatureVecs[counter] = GenerateFeatureVec(review, model, num_features)
        counter += 1

    return reviewFeatureVecs

def TrainingVectorModels(train_data):
    '''
    Based on the training data, this function trains two Word2Vec models: model_queries and model_passage on the queries and passage data.
    '''
    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # Initialize an empty list of sentences
    sentences_queries = []  
    sentences_passage = []

    # Parse sentences from training set
    for i in tqdm(range(0, len(train_data))):
        sentences_queries += GenerateSentences(train_data.iloc[i]['queries'], tokenizer)
    for i in tqdm(range(0, len(train_data))):
        sentences_passage += GenerateSentences(train_data.iloc[i]['passage'], tokenizer)

    # Train the model
    model_queries = word2vec.Word2Vec(sentences_queries, vector_size = 300, workers=4, min_count = 2,
                            window = 1, sample = 1e-3)

    model_passage = word2vec.Word2Vec(sentences_passage, vector_size = 300, workers=4, min_count = 4,
                            window = 10, sample = 1e-3)

    return model_queries,model_passage


def CalculateAverageVector(train_data,validation_data,model_queries,model_passage):
    '''
    This function is based on two Word2Vec models, calculates the average vector of train_data and validation_data respectively, and returns X_train, y_train, X_validation, y_validation for subsequent training
    '''
    # Create average feature vecs for train reviews    
    clean_train_reviews_queries = []
    clean_train_reviews_passage = []

    for i in tqdm(range(0, len(train_data))):
        clean_train_reviews_queries.append( GenerateWordlist( train_data.iloc[i]['queries'], remove_stopwords=False ))
        clean_train_reviews_passage.append( GenerateWordlist( train_data.iloc[i]['passage'], remove_stopwords=True ))

    # Creating average feature vecs for validation reviews
    clean_validation_reviews_queries = []
    clean_validation_reviews_passage = []

    for i in tqdm(range(0, len(validation_data))):
        clean_validation_reviews_queries.append( GenerateWordlist( validation_data.iloc[i]['queries'], remove_stopwords=False ))
        clean_validation_reviews_passage.append( GenerateWordlist( validation_data.iloc[i]['passage'], remove_stopwords=True ))

    # Based on the mean vector, generate data
    X_train = np.hstack((np.array(GetAvgFeatureVecs( clean_train_reviews_queries, model_queries, 300 )),np.array(GetAvgFeatureVecs( clean_train_reviews_passage, model_passage, 300 ))))   
    X_train[np.isnan(X_train)]=0
    y_train = np.array(train_data["relevancy"]).reshape(len(train_data),1)

    X_validation = np.hstack((np.array(GetAvgFeatureVecs( clean_validation_reviews_queries, model_queries, 300 )),np.array(GetAvgFeatureVecs( clean_validation_reviews_passage, model_passage, 300 ))))  
    X_validation[np.isnan(X_validation)]=0 
    y_validation = np.array(validation_data["relevancy"]).reshape(len(validation_data),1)

    return X_train,y_train,X_validation,y_validation


#Function B:The following functions are used to implement the training algorithm for a handwritten logistic regression model
#==========================================================================================================#
def Sigmoid(z):

    a = 1/(1+np.exp(-z))

    return a

def Propagate(w, b, X, Y):
    """
    This function is used to calculate the gradients of X and Y after forward and backward propagation based on a given weight w and offset b    
    """
    m = X.shape[1]
    # forward propagation 
    A = Sigmoid(np.dot(w.T,X)+b) 
    cost = -(np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)))/m                 
    # backpropagation
    dZ = A-Y
    dw = (np.dot(X,dZ.T))/m
    db = (np.sum(dZ))/m

    return dw, db, cost

def LogisticModel(X_train, Y_train, learning_rate=0.1, num_iterations=1000):

    dim = X_train.shape[0]
    W = np.zeros((dim,1))
    b = 0

    costs=[]

    #Gradient descent, iteratively find the model parameters
    for i in range(num_iterations):

        dw, db, cost = Propagate(W,b,X_train,Y_train)
        W = W - learning_rate*dw
        b = b - learning_rate*db
        #if i%100 == 0:
        print ("Cost after iteration %i: %f" %(i, cost))
        costs.append(cost)

    b = np.array(b).reshape((1))

    return W,b,costs

#Function C:The following functions are used to implement predictions on validation data
#==========================================================================================================#
def Predict(w,b,X):

    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    A = Sigmoid(np.dot(w.T,X)+b)

    for  i in range(m):

        if A[0,i]>0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0

    return Y_prediction

#Function D:The following functions are used to implement the evaluation of the training results
#==========================================================================================================#
def InitializeData(y_validation_prediction):

    validation_data = pd.read_csv( "raw/validation_data.tsv", header=0, delimiter="\t", quoting=3 )[:500000]
    all_data = validation_data
    all_data['score']=y_validation_prediction

    query_data = all_data.loc[:,['qid','queries']].drop_duplicates(subset='qid').reset_index(drop=True)
    query_data['ave_precision']=0.0
    query_data['ave_precision']=query_data['ave_precision'].astype(float)
    query_data['NDCG']=0.0
    
    return all_data,query_data
    
def GenerateSortedDataFrame(all_data,algoname):
    '''
    This function sorts each record in validation_data according to the predicted correlation of the logistic regression model
    '''

    query_data = all_data.loc[:,['qid','queries']].drop_duplicates(subset='qid').reset_index(drop=True)
    query_data['ave_precision']=0
    query_data['NDCG']=0
    np_query_data = np.array(query_data)

    sorted_DF = pd.DataFrame(columns=['qid','pid','score','relevancy','A','algoname','rank'])
    # for each query vector
    for query in tqdm(np_query_data):
        #Select the first 100 of the corresponding 1000 Passages (the Passages of some query vectors are less than 100)
        query_records = []
        query_passages = np.array(all_data.loc[all_data['qid']==query[0]])#[:100]
        #Calculate the cosine similarity between the query vector and its corresponding 100 (some less than 100) passages
        for query_passage in query_passages:
            
            query_records.append([query[0],query_passage[1],query_passage[5],query_passage[4]])

        # Sort in reverse order according to the score, and store the result in a DataFrame
        query_records = sorted(query_records, key = lambda x:-x[2])

        sorted_DF_query_records = pd.DataFrame(query_records, columns=['qid','pid','score','relevancy'])
        sorted_DF_query_records['A']='A1'
        sorted_DF_query_records['algoname']=algoname
        sorted_DF_query_records['rank']=sorted_DF_query_records.index+1
        sorted_DF = pd.concat([sorted_DF, sorted_DF_query_records]).reset_index(drop=True)

    return sorted_DF


if __name__ == '__main__':

    #Part A: The following code is used to generate the data X_train, y_train, X_validation, y_validation in vector format required for training according to the Word2Vec algorithm (this part takes about 31h36min)
    #==================================================== ===================================================== =======#

    train_data = pd.read_csv( "raw/train_data.tsv", header=0, delimiter="\t", quoting=3 )
    validation_data = pd.read_csv( "raw/validation_data.tsv", header=0, delimiter="\t", quoting=3 )

    # To save training time, select about 50% of all data for training
    train_data = train_data[:2000000]
    validation_data = validation_data[:500000]

    # Confirm the number of selected data and the frequency distribution of the correlation
    print("Read %d labeled train reviews, %d labeled validation reviews, " % (train_data["passage"].size, validation_data["passage"].size))
    print(train_data['relevancy'].value_counts())
    print(validation_data['relevancy'].value_counts())

    model_queries,model_passage = TrainingVectorModels(train_data)
    X_train,y_train,X_validation,y_validation = CalculateAverageVector(train_data,validation_data,model_queries,model_passage)
    
    # X_train(2000000,600),y_train(2000000,1),X_validation(500000,600),y_validation(500000,1)
    print(X_train.shape)
    print(y_train.shape)
    print(X_validation.shape)
    print(y_validation.shape)

    # Save the generated data in vector format to four files in txt format
    np.savetxt("TrainData_File/X_train_task2.txt", X_train,fmt='%f',delimiter=',')
    np.savetxt("TrainData_File/y_train_task2.txt", y_train,fmt='%f',delimiter=',')
    np.savetxt("TrainData_File/X_validation_task2.txt", X_validation,fmt='%f',delimiter=',')
    np.savetxt("TrainData_File/y_validation_task2.txt", y_validation,fmt='%f',delimiter=',')

    #Part B: The following code trains the logistic regression model based on the X_train and y_train data, and obtains the parameters W and offset b of the trained model (this part takes about 20min)
    #==================================================== ===================================================== =======#

    print("Reading x train data")
    X_train= np.loadtxt("TrainData_File/X_train_task2.txt",delimiter=',')
    print("Reading y train data")
    y_train= np.loadtxt("TrainData_File/y_train_task2.txt",delimiter=',')

    print("Fitting a Logistic Regression Model to labeled training data...")
    learning_rate=0.1
    num_iterations=100

    # Fit a Logistic Regression Model to the training data
    W,b,costs = LogisticModel(X_train.T,y_train,learning_rate,num_iterations)

    np.savetxt("TrainData_File/task2_W.txt", W,fmt='%f',delimiter=',')
    np.savetxt("TrainData_File/task2_b.txt", b,fmt='%f',delimiter=',')

    #Part C: The following code predicts the validation data X_validation based on the parameters W and b of the logistic regression model (this part takes about 2min)
    #==================================================== ===================================================== =======#

    W = np.loadtxt("TrainData_File/task2_W.txt",delimiter=',').reshape((600, 1))
    b = np.loadtxt("TrainData_File/task2_b.txt",delimiter=',')

    X_validation = np.loadtxt("TrainData_File/X_validation_task2.txt",delimiter=',')
    y_validation = np.loadtxt("TrainData_File/y_validation_task2.txt",delimiter=',')
    
    #Use X_validation data to make predictions
    y_validation_prediction = Predict(W,b,X_validation.T).T

    #Part D: The following code is used to sort and evaluate the validation data (this part takes about 3min)
    #==================================================== ===================================================== =======#

    #Initialize all_data containing all data records, and query_data containing all queries
    all_data, query_data = InitializeData(y_validation_prediction)

    #According to the prediction correlation based on LogisticRegressionModel in all_data, return the sorted all_data, namely LogisticRegression_Df
    LogisticRegression_Df = GenerateSortedDataFrame(all_data,'LR')

    #Save the sorting results to the LR.txt file according to the format requirements
    order = ['qid', 'A', 'pid', 'rank', 'score','algoname']
    LogisticRegression_Df_record = LogisticRegression_Df[order]
    LogisticRegression_Df_record = LogisticRegression_Df_record.query('rank<=100')
    LogisticRegression_Df_record.to_csv('LR.txt',sep='\t',index=False,header=False)

    #Calculate AP and NDCG corresponding to each query
    query_data = t1.CalculateAveragePrecision(query_data,LogisticRegression_Df)
    query_data = t1.CalculateNDCG(query_data,LogisticRegression_Df)

    print("The MAP of logistic regression model is")
    print(query_data['ave_precision'].mean())

    print("The NDCG of logistic regression model is")
    print(query_data['NDCG'].mean())

    #Part E: The following code is used to plot the convergence of the loss function for different learning rates (this part takes about 20min)
    #==================================================== ===================================================== =======#

    W,b,costs0 = LogisticModel(X_train.T,y_train,10,num_iterations)
    W,b,costs1 = LogisticModel(X_train.T,y_train,1,num_iterations)
    W,b,costs2 = LogisticModel(X_train.T,y_train,0.5,num_iterations)
    W,b,costs3 = LogisticModel(X_train.T,y_train,0.1,num_iterations)
    W,b,costs4 = LogisticModel(X_train.T,y_train,0.01,num_iterations)

    x = range(100)

    plt.plot(x,costs0,label='learning_rate = 10')
    plt.plot(x,costs1,label='learning_rate = 1')
    plt.plot(x,costs2,label='learning_rate = 0.5')
    plt.plot(x,costs3,label='learning_rate = 0.1')
    plt.plot(x,costs4,label='learning_rate = 0.01')
    plt.legend(loc = 'upper right')
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss cost value")
    plt.show()   
