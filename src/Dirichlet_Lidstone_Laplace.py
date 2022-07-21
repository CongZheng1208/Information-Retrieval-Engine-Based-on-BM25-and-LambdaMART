import math
import numpy as np
import pandas as pd
import InvertedIndexGenerator as t2
import numpy as np

def CalculateLaplaceSimilarity(inverted_index,query,passage):
    '''
    This function is used to calculate the Laplace similarity between two vectors
    '''
    laplace_value = 0
    #dl is the length of the passage
    dl = len(passage[1].split())
    # v is the length of the vocabulary
    v = len(inverted_index)
    #For each term in query_vec
    for term in query[1].split():
        #f is the number of times the term appears in the passage corresponding to passage_vec
        f = passage[1].split().count(term)
        laplace_value=laplace_value+math.log((f+1)/(dl+v))

    return laplace_value

def CalculateLidstoneSimilarity(inverted_index,query,passage):
    '''
    This function is used to calculate the bm25 similarity between two vectors
    '''
    lidstone_value = 0
    #dl is the length of the passage
    dl = len(passage[1].split())
    # v is the length of the vocabulary
    v = len(inverted_index)
    epsilon = 0.1
    #For each term in query_vec
    for term in query[1].split():
        
        #f is the number of times the term appears in the passage corresponding to passage_vec
        f = passage[1].split().count(term)
        tf = f/dl
        lidstone_value=lidstone_value+math.log((tf+epsilon)/(dl+epsilon*v))

    return lidstone_value


def CalculateDirichletSimilarity(inverted_index,query,passage):
    '''
    This function is used to calculate the bm25 similarity between two vectors
    '''
    dirichlet_value = 0
    #dl is the length of the passage
    dl = len(passage[1].split())
    #pl is the total number of words in all passages
    pl=13230272
    mu = 50
    #For each term in query_vec
    for term in query[1].split():

        if term in inverted_index.keys():
            #f is the number of times the term appears in the passage
            f = passage[1].split().count(term)
            #n is the number of passages that contain the term
            n = len(inverted_index[term])
            dirichlet_value=dirichlet_value+math.log((dl*f)/(dl*(dl+mu))+(mu*n)/(pl*(dl+mu)))
        else:
            continue

    return dirichlet_value


if __name__ == '__main__':
    #Read data in DataFrame format and add headers
    all_data=pd.read_csv('raw/candidate_passages_top1000.tsv', sep='\t', header= None)
    all_data.columns = ['qid','pid','query','passage']
    #According to the uniqueness of pid, filter out each unique pid and its corresponding passage, and reset the serial number
    passage_data = all_data.loc[:,['pid','passage']].drop_duplicates(subset='pid').reset_index(drop=True)
    passage_data['word_count']=passage_data['passage'].apply(lambda x:len(str(x).split(" ")))
    np_passage_data = np.array(passage_data)
    #Get the inverted index
    inverted_index = t2.GenerateInvertedIndex(np_passage_data)
   
    query_data=pd.read_csv('raw/test-queries.tsv', sep='\t', header= None)
    query_data.columns = ['qid','query']
    np_query_data = np.array(query_data)
    
    laplaceDf = pd.DataFrame(columns=['qid','pid','store'])
    lidstoneDf = pd.DataFrame(columns=['qid','pid','store'])
    dirichletDf = pd.DataFrame(columns=['qid','pid','store'])

    # for each query vector
    for query in np_query_data:
        #Select the first 100 of the corresponding 1000 Passages (the Passages of some query vectors are less than 100)
        query_records0 = []
        query_records1 = []
        query_records2 = []
        query_passages = np.array(all_data.loc[all_data['qid']==query[0]][:100])
        #Calculate the cosine similarity between the query vector and its corresponding 100 (some less than 100) passages
        for query_passage in query_passages:
            passage = np.array([query_passage[1],query_passage[3]])

            laplace_similarity = CalculateLaplaceSimilarity(inverted_index,query, passage)
            query_records0.append([query[0],passage[0],laplace_similarity])
            lidstone_similarity = CalculateLidstoneSimilarity(inverted_index,query, passage)
            query_records1.append([query[0],passage[0],lidstone_similarity])
            dirichlet_similarity = CalculateDirichletSimilarity(inverted_index,query, passage)
            query_records2.append([query[0],passage[0],dirichlet_similarity])

        # Sort in reverse order according to the Cos similarity of tfidf, and store the result in a DataFrame
        query_records0 = sorted(query_records0, key = lambda x:-x[2])
        query_records1 = sorted(query_records1, key = lambda x:-x[2])
        query_records2 = sorted(query_records2, key = lambda x:-x[2])

        laplaceDf_query_records = pd.DataFrame(query_records0, columns=['qid','pid','store'])
        lidstoneDf_query_records = pd.DataFrame(query_records1, columns=['qid','pid','store'])
        dirichletDf_query_records = pd.DataFrame(query_records2, columns=['qid','pid','store'])
        
        laplaceDf = pd.concat([laplaceDf, laplaceDf_query_records]).reset_index(drop=True)
        lidstoneDf = pd.concat([lidstoneDf, lidstoneDf_query_records]).reset_index(drop=True)
        dirichletDf = pd.concat([dirichletDf, dirichletDf_query_records]).reset_index(drop=True)

    #Save the bm25 values ​​corresponding to about 100 Passages (19290 in total) corresponding to 200 queries in csv format
    laplaceDf.to_csv('laplace.csv',index=False,header=False)
    #Save the bm25 values ​​corresponding to about 100 Passages (19290 in total) corresponding to 200 queries in csv format
    lidstoneDf.to_csv('lidstone.csv',index=False,header=False)
    #Save the bm25 values ​​corresponding to about 100 Passages (19290 in total) corresponding to 200 queries in csv format
    dirichletDf.to_csv('dirichlet.csv',index=False,header=False)