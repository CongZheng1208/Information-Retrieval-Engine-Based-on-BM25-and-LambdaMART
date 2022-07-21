import numpy as np
import xgboost as xgb
import EvaluationMetricGenerator as t1
import LogisticRegression as t2
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':

    #Part A: The following code is used to generate training data and validation data in the format required by the xgboost library (this part takes about 33min)
    #==================================================== ===================================================== =======#

    train_data = pd.read_csv( "raw/train_data.tsv", header=0, delimiter="\t", quoting=3 )
    validation_data = pd.read_csv( "raw/validation_data.tsv", header=0, delimiter="\t", quoting=3 )

    train_data = train_data[:2000000]
    validation_data = validation_data[:500000]

    X_train= np.loadtxt('TrainData_File/X_train_task2.txt',delimiter=',')
    y_train= np.loadtxt('TrainData_File/y_train_task2.txt',delimiter=',')
    X_validation = np.loadtxt('TrainData_File/X_validation_task2.txt',delimiter=',')
    y_validation = np.loadtxt('TrainData_File/y_validation_task2.txt',delimiter=',')

    # The following code is used to generate training data in the format required by the xgboost library
    train_file = open("TrainData_File/task3_train_data.txt", 'w')

    for i in tqdm(range(X_train.shape[0])):

        train_file.write(str(y_train[i])+' ')
        train_file.write('qid'+':'+str(train_data.iloc[i]["qid"])+' ')

        for j in range(X_train.shape[1]):
        
            train_file.write(str(j)+':'+str(X_train[i][j]))
            train_file.write(' ')
    
        train_file.write('\n')
    
    train_file.close()

    # The following code is used to generate validation data in the format required by the xgboost library
    validation_file = open("TrainData_File/task3_validation_data.txt", 'w')

    for i in tqdm(range(X_validation.shape[0])):

        validation_file.write(str(y_validation[i])+' ')
        validation_file.write('qid'+':'+str(validation_data.iloc[i]["qid"])+' ')

        for j in range(X_validation.shape[1]):
        
            validation_file.write(str(j)+':'+str(X_validation[i][j]))
            validation_file.write(' ')
    
        validation_file.write('\n')
        
    validation_file.close()


    #Part B: The following code is used to train the LambdaMART model based on the xgboost library (this part takes about 16min)
    #==================================================== ===================================================== =======#

    training_data = xgb.DMatrix("TrainData_File/task3_train_data.txt")
    validation_data = xgb.DMatrix("TrainData_File/task3_validation_data.txt")
    param = {'max_depth':6, 'eta':0.3, 'objective':'rank:pairwise'}
    LambdaMART_model = xgb.train(param, training_data)

    #Part C: The following code predicts, sorts and evaluates the validation data X_validation based on the LambdaMART model (this part takes about 1min)
    #==================================================== ===================================================== =======#
    y_validation_prediction = LambdaMART_model.predict(validation_data)

    #Part D: The following code is used to sort and evaluate the validation data (this part takes about 4min)
    #==================================================== ===================================================== =======#

    #Initialize all_data containing all data records, and query_data containing all queries
    all_data, query_data = t2.InitializeData(y_validation_prediction)
    
    #According to the prediction correlation based on LambdaMARTModel in all_data, return the sorted all_data, namely LambdaMART_Df
    LambdaMART_Df = t2.GenerateSortedDataFrame(all_data,'LM')

    #Save the sorting results to the LM.txt file according to the format requirements
    order = ['qid', 'A', 'pid', 'rank', 'score','algoname']
    LambdaMART_Df_record = LambdaMART_Df[order]
    LambdaMART_Df_record = LambdaMART_Df_record.query('rank<=100')
    LambdaMART_Df_record.to_csv('LM.txt',sep='\t',index=False,header=False)

    #Calculate AP and NDCG corresponding to each query
    query_data = t1.CalculateAveragePrecision(query_data,LambdaMART_Df)
    query_data = t1.CalculateNDCG(query_data,LambdaMART_Df)

    print("The MAP of LambdaMART Model is")
    print(query_data['ave_precision'].mean())

    print("The NDCG of LambdaMART Model is")
    print(query_data['NDCG'].mean())