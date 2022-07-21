import pandas as pd
import TextPreprocessing as t1
import numpy as np
import progressbar

bar = progressbar.ProgressBar(widgets=[
    progressbar.Percentage(),
    ' (', progressbar.SimpleProgress(), ') ',
    ' (', progressbar.AbsoluteETA(), ') ',])

def GenerateInvertedIndex(np_passage_data):
    '''
    #This function is used to generate its corresponding inverted index in the vocabulary for a given Passage library
    '''
    invert_index={}
    #For each passport in the passport library
    for passage_item in bar(np_passage_data):
        #For the token split into each line of passport

        processed_passage = t1.TextPreprocessing(passage_item[1])

        for token in set(processed_passage):
            #if the token is already in the dictionary
            if token in invert_index:
                invert_index[token].append((passage_item[0],processed_passage.count(token),passage_item[2]))
            else:
                invert_index[token] = [(passage_item[0],processed_passage.count(token),passage_item[2])]

    return invert_index

if __name__ == '__main__':
    #Read data in DataFrame format and add headers
    all_data=pd.read_csv('../raw/candidate_passages_top1000.tsv', sep='\t', header= None)
    all_data.columns = ['qid','pid','query','passage']
    #According to the uniqueness of pid, filter out each unique pid and its corresponding passage, and reset the serial number
    passage_data = all_data.loc[:,['pid','passage']].drop_duplicates(subset='pid').reset_index(drop=True)

    passage_data['word_count']=passage_data['passage'].apply(lambda x:len(str(x).split(" ")))
    np_passage_data = np.array(passage_data)

    #Get the inverted index
    invertedindex = GenerateInvertedIndex(np_passage_data)
    print(len(np_passage_data))
    print(np.mean(passage_data['word_count']))