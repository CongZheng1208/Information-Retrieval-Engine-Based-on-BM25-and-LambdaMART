import pandas as pd
import math
import numpy as np
from tqdm import tqdm

#Function A:The following functions are used to implement the BM25 model
#==========================================================================================================#
def TextPreprocessing(original_text):
    '''
    This function is used to get the text from the specified file path and return the text after text preprocessing
    '''
    abnormal_symbol_collection = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    
    #Replace all uppercase in text with lowercase
    lower_txt=original_text.lower()
    #Remove all non-text symbols from the text
    for asc in abnormal_symbol_collection:
        lower_txt=lower_txt.replace(asc," ")

    # break the text into tokens
    tokens = lower_txt.split()

    #return the tokens after text preprocessing
    return tokens

def GenerateInvertedIndex(np_passage_data):
    '''
    #This function is used to generate its corresponding inverted index in the vocabulary for a given Passage library
    '''
    invert_index={}
    #For each passport in the passport library
    for passage_item in tqdm(np_passage_data):
        #For the token split into each line of passport

        processed_passage = TextPreprocessing(passage_item[1])

        for token in set(processed_passage):
            #if the token is already in the dictionary
            if token in invert_index:
                invert_index[token].append((passage_item[0],processed_passage.count(token),passage_item[2]))
            else:
                invert_index[token] = [(passage_item[0],processed_passage.count(token),passage_item[2])]

    return invert_index

def CalculateBM25Similarity(inverted_index,query,passage):
    '''
    This function is used to calculate the bm25 similarity between two vectors
    '''
    bm25_value = 0
    #N is the total number of passes
    N = 955211
    #avdl is the average length of all passages
    avdl=56.82515067351611
    #dl is the length of the passage
    dl = len(passage[1].split())
    k1 = 1.2
    k2 = 100
    b = 0.75

    #For each term in query_vec
    for term in query[1].split():
        
        #if the term is contained in the inverted index
        if term in inverted_index.keys():
            #n is the number of passages that contain the term
            n = len(inverted_index[term])
            #f is the number of times the term appears in the passage corresponding to passage_vec
            f = passage[1].split().count(term)
            #qf is the number of times the term appears in the query corresponding to query_vec
            qf = query[1].split().count(term)

            binary_independent_model = math.log((N-n+0.5)/(n+0.5))
            query_passage_weight = f*(k1+1)/(k1*(1-b+b*dl/avdl)+f)
            query_weight = qf*(k2+1)/(k2+qf)
            bm25_value=bm25_value+binary_independent_model*query_passage_weight*query_weight
        else:
            continue

    return bm25_value

def BM25Model(all_data):
    '''
    Based on the read verification data, this function sorts it according to the BM25 algorithm and returns the sorted result
    '''

    #According to the uniqueness of pid, filter out each unique pid and its corresponding passage, and reset the serial number
    passage_data = all_data.loc[:,['pid','passage','relevancy']].drop_duplicates(subset='pid').reset_index(drop=True)
    passage_data['word_count']=passage_data['passage'].apply(lambda x:len(str(x).split(" ")))
    np_passage_data = np.array(passage_data)

    #Get the inverted index
    inverted_index = GenerateInvertedIndex(np_passage_data)

    query_data = all_data.loc[:,['qid','queries','relevancy']].drop_duplicates(subset='qid').reset_index(drop=True)
    np_query_data = np.array(query_data)

    bm25_Df = pd.DataFrame(columns=['qid','pid','score','relevancy'])
    # for each query vector
    for query in tqdm(np_query_data):
        #Select the first 100 of the corresponding 1000 Passages (the Passages of some query vectors are less than 100)
        query_records = []
        query_passages = np.array(all_data.loc[all_data['qid']==query[0]][:100])
        #Calculate the cosine similarity between the query vector and its corresponding 100 (some less than 100) passages
        for query_passage in query_passages:
            passage = np.array([query_passage[1],query_passage[3]])
            bm25_similarity = CalculateBM25Similarity(inverted_index,query, passage)
            query_records.append([query[0],passage[0],bm25_similarity,query_passage[4]])

        # Sort in reverse order according to the Cos similarity of tfidf, and score the result in a DataFrame
        query_records = sorted(query_records, key = lambda x:-x[2])
        bm25Df_query_records = pd.DataFrame(query_records, columns=['qid','pid','score','relevancy'])
        bm25_Df = pd.concat([bm25_Df, bm25Df_query_records]).reset_index(drop=True)

    return bm25_Df

#Function B:The following functions are used to implement the evaluation indicators MAP and NDCG
#==========================================================================================================#
def CalculateAveragePrecision(query_data,dataframe):
    '''
    This function is used to calculate the AveragePrecision value corresponding to each query in query_data and add it as a new column of query_data
    '''
    np_query_data = np.array(query_data)
    #for each query
    for row in tqdm(np_query_data):
        ave_precision = 0.0
        count = 1.0
       
        #qid_data is the search records corresponding to the query qid
        qid_data = dataframe[(dataframe['qid'] == row[0])]
        qid_data.reset_index(drop=True, inplace=True)

        #For each retrieved record, check whether it is relevant, if relevant, accumulate the accuracy
        for index,row1 in qid_data.iterrows():
            if row1['relevancy'] == 1.0:
                ave_precision += count/np.double(index+1)
                count += 1.0
            else:
                continue

        #If all the query data corresponding to qid contains at least one related file, it will be included in the AP
        if 1.0 in qid_data['relevancy'].values:
            N=np.double(qid_data.groupby(['relevancy']).size()[1.0])
            row[2] = ave_precision/N
        else:
            row[2] = 0.0 
       
    query_data = pd.DataFrame(np_query_data)
    query_data.columns = ['qid','queries','ave_precision','NDCG']

    return query_data

def CalculateNDCG(query_data,dataframe):
    '''
    This function is used to calculate the NDCG value corresponding to each query in query_data and add it as a new column of query_data
    '''
    np_query_data = np.array(query_data)

    #for every query
    for row in tqdm(np_query_data):
        DCG = 0.0

        #qid_data is the retrieval records corresponding to the query qid
        qid_data = dataframe[(dataframe['qid'] == row[0])]
        qid_data.reset_index(drop=True, inplace=True)
        #For each of the retrieved records, the DCG value is calculated
        for index,row1 in qid_data.iterrows():
            DCG += (2**row1['relevancy'] -1)/math.log(index+2,2)

        row[3] = DCG
    
    query_data = pd.DataFrame(np_query_data)
    query_data.columns = ['qid','query','ave_precision','NDCG']

    #normalize
    optDCG = query_data['NDCG'].max()

    if optDCG != 0.0:
        query_data['NDCG'] = query_data['NDCG']/optDCG
    else:
        query_data['NDCG'] = 0.0

    return query_data



if __name__ == '__main__':
    all_data=pd.read_csv('../raw/validation_data.tsv', sep='\t')

    query_data = all_data.loc[:,['qid','queries']].drop_duplicates(subset='qid').reset_index(drop=True)
    query_data['ave_precision']=0
    query_data['NDCG']=0
   
    bm25_Df = BM25Model(all_data)

    query_data = CalculateAveragePrecision(query_data,bm25_Df)
    query_data = CalculateNDCG(query_data,bm25_Df)

    print("The MAP of BM25 Model is")
    print(query_data['ave_precision'].mean())

    print("The NDCG of BM25 Model is")
    print(query_data['NDCG'].mean())













   
        

