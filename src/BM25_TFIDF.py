import math
import numpy as np
import pandas as pd
import InvertedIndexGenerator as t2
import numpy as np
import progressbar

bar = progressbar.ProgressBar(widgets=[
    progressbar.Percentage(),
    ' (', progressbar.SimpleProgress(), ') ',
    ' (', progressbar.AbsoluteETA(), ') ',])

def GenerateTfIdfVectors(inverted_index,passage_data):
    '''
    This function is used to calculate the tf-idf value of the vocabulary based on the inverted sequence
    '''
    tf_idf_vectors={}

    #For each word in the vocabulary and its corresponding inverted index
    for term, invert_indexs in bar(inverted_index.items()):
        #Use the number of times in the passage that the word appears and the total number of passages to calculate idf
        idf = math.log(passage_data.shape[0]/len(invert_indexs),10)
        
        #For every passage where the word appears
        for invertindex in invert_indexs:
            #Use the number of times the word appears in the passage and the length of the passage itself to calculate tf
            tf = invertindex[1]/invertindex[2]
            if invertindex[0] in tf_idf_vectors:
                tf_idf_vectors[invertindex[0]].append((term,tf*idf))
            else:
                tf_idf_vectors[invertindex[0]] = [(term,tf*idf)]
    
    return tf_idf_vectors

def GenerateTfIdfQueryVectors(inverted_index, passage_vector,np_query_data):
    '''
    This function is used to calculate the tf-idf value of the vocabulary based on the inverted sequence
    '''
    tf_idf_query_vectors={}
    # For each query in the query dataset
    for query in np_query_data:
        qid = query[0]
        split_query_content = query[1].split()

        tf_idf_query_vectors[qid]=[]
        #For each word in the query
        for token in split_query_content:
            # If the word exists in the inverted index of the text library, calculate the tf and idf corresponding to the word and store it in the query vector
            if token in inverted_index.keys():
                tf = split_query_content.count(token)/len(split_query_content)
                idf = math.log(len(passage_vector)/ len(inverted_index[token]),10)
                tf_idf_query_vectors[qid].append((token,tf*idf))
            else:
                continue
    return tf_idf_query_vectors


def CalculateCosSimilarity(Vector1, Vector2):
    '''
    This function is used to calculate the cosine similarity between two vectors
    '''
    dot_prob = 0
    for vector1 in Vector1:
        for vector2 in Vector2:
          if vector1[0]==vector2[0]:
              dot_prob += float(vector1[1])*float(vector2[1])

    mag_1 = math.sqrt(np.sum([float(vector1[1])**2 for vector1 in Vector1]))
    mag_2 = math.sqrt(np.sum([float(vector2[1])**2 for vector2 in Vector2]))
    # If the length of one of the vectors is 0, do not calculate and return 0 directly
    if mag_1==0 or mag_2==0:
        return 0
    return dot_prob / (mag_1 * mag_2)

def CalculateBM25Similarity(inverted_index,query,passage):
    '''
    This function is used to calculate the bm25 similarity between two vectors
    '''
    bm25_value = 0
    #N is the total number of passes
    N = 182469
    #avdl is the average length of all passages
    avdl=57.24898475905496
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
    #Based on the inverted index, get the TfIdf vector of the passage
    passage_vector = GenerateTfIdfVectors(inverted_index,np_passage_data)
   
    query_data=pd.read_csv('raw/test-queries.tsv', sep='\t', header= None)
    query_data.columns = ['qid','query']
    np_query_data = np.array(query_data)
    #Based on the TfIdf vector of Passage, get the TfIdf vector of all queries
    query_vector = GenerateTfIdfQueryVectors(inverted_index,passage_vector,np_query_data)

    tfidfDf = pd.DataFrame(columns=['qid','pid','store'])
    # for each query vector
    for query_id, query_vec in query_vector.items():
        #Select the first 100 of the corresponding 1000 Passages (the Passages of some query vectors are less than 100)
        query_records = []
        query_passages = np.array(all_data.loc[all_data['qid']==query_id][:100])

        #Calculate the cosine similarity between the query vector and its corresponding 100 (some less than 100) passages
        for query_passage in query_passages:
            pid = query_passage[1]
            passage_vec = passage_vector[pid]
            cos_similarity = CalculateCosSimilarity(query_vec,passage_vec)
            query_records.append([query_id,pid,cos_similarity])

        # Sort in reverse order according to the Cos similarity of tfidf, and store the result in a DataFrame
        query_records = sorted(query_records, key = lambda x:-x[2])
        tfidfDf_query_records = pd.DataFrame(query_records, columns=['qid','pid','store'])
        tfidfDf = pd.concat([tfidfDf, tfidfDf_query_records]).reset_index(drop=True)
    # Save the TfIdf cosine similarity values corresponding to about 100 Passages corresponding to 200 queries (total 19290) in csv format
    tfidfDf.to_csv('tfidf.csv',index=False,header=False)


    bm25Df = pd.DataFrame(columns=['qid','pid','store'])
    # for each query vector
    for query in np_query_data:
        #Select the first 100 of the corresponding 1000 Passages (the Passages of some query vectors are less than 100)
        query_records = []
        query_passages = np.array(all_data.loc[all_data['qid']==query[0]][:100])
        #Calculate the cosine similarity between the query vector and its corresponding 100 (some less than 100) passages
        for query_passage in query_passages:
            passage = np.array([query_passage[1],query_passage[3]])
            bm25_similarity = CalculateBM25Similarity(inverted_index,query, passage)
            query_records.append([query[0],passage[0],bm25_similarity])

        # Sort in reverse order according to the Cos similarity of tfidf, and store the result in a DataFrame
        query_records = sorted(query_records, key = lambda x:-x[2])
        bm25Df_query_records = pd.DataFrame(query_records, columns=['qid','pid','store'])
        bm25Df = pd.concat([bm25Df, bm25Df_query_records]).reset_index(drop=True)

    #Save the bm25 values ​​corresponding to about 100 Passages (19290 in total) corresponding to 200 queries in csv format
    bm25Df.to_csv('bm25.csv',index=False,header=False)
    


    

        


            


        
  







