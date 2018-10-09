import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer



file1 = pd.read_csv('/Users/adityakumar/Desktop/file.csv')

### The csv file which i am getting when i am using twint also contains cloumns which we don't need like retweets, link, etc. so first we need to remove this


col_to_drop = ['date', 'time', 'timezone', 'replies', 'retweets', 'likes', 'location', 'hashtags', 'link', 'retweet', 'user_rt', 'mentions']
file1 = file1.drop(col_to_drop, axis=1)


file1 = file1.drop_duplicates('id', 'first')
file1 = file1.drop_duplicates('tweet', 'first')
file1 = file1.reset_index(drop=True)


file1['label'] = ' '      # here i added a new column 'label' for labelling tweets
file1 = file1[['id', 'user_id', 'username', 'label', 'tweet']] # rearranging columns



file1.to_csv('/Users/adityakumar/Desktop/tweets.csv', sep=',') # saving the file for manual labelling or annotation


 # Preprocessing tweets

### After manual annotation we will first add new column 'preprocessed tweet'


file2 = pd.read_csv('/Users/adityakumar/Desktop/tweets.csv')




file2 = file2.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis = 1)



file2['preprocessed tweet'] = ' '




file2['label'].replace(' ', np.nan, inplace=True) ## for tweets which are not in english or tweets with the
file2 = file2.dropna()                           ## sentence as the previous tweet i left unlabelled those tweets 
file2 = file2.reset_index(drop=True)    ## this is because the features for those tweets will be same and will 
                                        ## not contribute to the model.



stop = set(stopwords.words('english'))  # nltk provides a set of stopwords here we are loading those words 
                                         # which we will use in removing stopwords from tweets




porter = PorterStemmer() #There are many stemming algorithms, although a popular and long-standing 
                        #method is the Porter Stemming algorithm. 
                        #This method is available in NLTK via the PorterStemmer class here we are calling that class



def user(text):                 # this function will be used to remove '@user' from tweets because '@user' will not 
    textlist = []               # contribute to the sentiment of tweets
    for ix in text.split():
        if ix[:1]=='@':
            pass
        else:
            textlist.append(ix)
    return ' '.join(textlist)




for ix in range(file2.shape[0]):
    text1 = file2.loc[ix]['tweet']
    text1 = re.sub(r'http\S+', '', text1) # here removing links from tweets because links can't contribute to the
                                        # sentiment of tweets

    text1 = user(text1)   # calling user function to remove '@user' from the tweets
    
    
    tokens = word_tokenize(text1)    # here we are removing non-english words from sentence by checking whether
    words = [word for word in tokens if word.isalpha()] # it is alphabet or not if alphabet then it's in otherwise
    text1 = ' '.join(words)     # remove from tweet
    
    
    stopw = [i for i in text1.lower().split() if i not in stop]  # here removing stopwords from tweet by checking
    text1 = ' '.join(stopw) # it's availability in nltk set of stopwords if this word in set of stopwords then
                                # skip that word.
    
    token1 = word_tokenize(text1)      # here stemming the tweet by porter stemming algorithm
    stemmed = [porter.stem(word) for word in token1]
    text1 = ' '.join(stemmed)
    
    
    file2.at[ix, 'preprocessed tweet']= text1  # now adding that preprocessed tweet to column

    



file2 = file2.drop(['id', 'user_id'],axis=1)# removing tweet id and user id because we don't need



file2['S.No.'] = ' '
for ix in range(file2.shape[0]):
    file2.at[ix,'S.No.']=ix+1




file2 = file2[['S.No.', 'username', 'label','tweet','preprocessed tweet' ]]




file2.to_csv('/Users/adityakumar/Desktop/tweets.csv')

