import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer

def TextPreprocessing(original_text) :
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
    
    '''
    # Perform stemmering and lemmatizing operations
    stemmer = SnowballStemmer("english")
    wnl = WordNetLemmatizer()
    for i in range(len(tokens)):
        tokens[i] = stemmer.stem(wnl.lemmatize(tokens[i]))
    '''

    #return the tokens after text preprocessing
    return tokens

def StatisticalFrequency(tokens):
    '''
    This function is used to calculate the word frequency corresponding to the given tokens and returns a vocabulary
    '''
    tokens_frequency={}

    # Traverse the tokens in tokens, and record their word frequency
    for token in tokens :
        tokens_frequency[token]=tokens_frequency.get(token,0)+1
    
    #sort tokens_frequency
    sorted_tokens_frequency = list(tokens_frequency.items())
    sorted_tokens_frequency.sort(key=lambda x:x[1],reverse=True )

    #Convert the generated sorted_tokens_frequency to DataFrame form
    vocabulary=pd.DataFrame(sorted_tokens_frequency,
                            columns = ['terms','count'])
        
    #Calculate the probability of occurrence corresponding to each term (normalized probability)
    vocabulary['normalised frequency'] = vocabulary['count']/len(tokens)

    #Plot the change in normalized probability with frequency ranking
    plt.figure(figsize=(4, 4), dpi=70)
    plt.plot(vocabulary.index+1, vocabulary['normalised frequency'])
    plt.xlabel('Frequency Ranking')
    plt.ylabel('Normalised Frequency')
    plt.show()

    return vocabulary

if __name__ == '__main__':
    #Get the processed text
    original_text=open("../raw/passage_collection_new.txt").read()
    tokens=TextPreprocessing(original_text)

    #Convert tokens to terms
    vocabulary = StatisticalFrequency(tokens)
    terms = vocabulary['terms']

    Hn = 0
    for i in range(len(tokens)):
        Hn += 1/(i+1)

    #Draw the normalized probability of text and the comparison loglog diagram of Zipf's law
    plt.figure(figsize=(4, 4), dpi=70)
    plt.loglog(vocabulary.index+1, 1/((vocabulary.index+1)*Hn), label = 'Actual data')
    plt.loglog(vocabulary.index+1, vocabulary['normalised frequency'], label = 'Zipf\'s law')

    plt.legend()
    plt.xlabel('Frequency Ranking')
    plt.ylabel('Normalised Frequency')
    plt.show()