
# coding: utf-8

# In[4]:

import codecs
import pandas as pd
import itertools as it

import re

import fileinput

from pandas import DataFrame
import os

import spacy
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence

import en_core_web_sm
nlp = spacy.load('en')

import csv

# In[16]:

all_user_filepath = "collapsed_user_20000.txt"#"collapsed_business_rest_subset.txt"



# Setup and define function for NLP

# In[23]:

#helper functions from modern nlp in python
def punct_space(token):
    """
    helper function to eliminate tokens
    that are pure punctuation or whitespace
    """
    
    return token.is_punct or token.is_space

def line_review(filename):
    """
    generator function to read in reviews from the file
    and un-escape the original line breaks in the text
    """
    
    with codecs.open(filename, encoding='utf_8') as f:
        for review in f:
            yield review.replace('\\n', '\n')
            
def lemmatized_sentence_corpus(filename):
    """
    generator function to use spaCy to parse reviews,
    lemmatize the text, and yield sentences
    """
    
    for parsed_review in nlp.pipe(line_review(filename),
                                  batch_size=10000, n_threads=4):
        
        for sent in parsed_review.sents:
            yield u' '.join([token.lemma_ for token in sent
                             if not punct_space(token)])


# In[24]:

def get_sample_review(review_txt_filepath, review_number):
    """
    retrieve a particular review index
    from the reviews file and return it
    """
    
    return list(it.islice(line_review(review_txt_filepath),
                          review_number, review_number+1))[0]


# In[25]:

from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore

import pyLDAvis
import pyLDAvis.gensim
import warnings
import _pickle as pickle


# In[26]:

lda_model_filepath = 'lda_model_eat_30'
trigram_dictionary_filepath = 'trigram_dict_eat_30.dict'
trigram_model_filepath = 'trigram_model_all_eat_30'
bigram_model_filepath = 'bigram_model_all_eat_30'


# In[27]:

lda = LdaMulticore.load(lda_model_filepath)
trigram_dictionary = Dictionary.load(trigram_dictionary_filepath)
trigram_model = Phrases.load(trigram_model_filepath)
bigram_model = Phrases.load(bigram_model_filepath)


all_numbers = list(range(0,50))
df_all_numbers = pd.DataFrame(columns =["topic_number"])
for topic_number in all_numbers:
    df_all_numbers = df_all_numbers.append({
     "topic_number": topic_number
      }, ignore_index=True)



# In[32]:

#output_file = open('review_bus_lda.txt','w')


# In[ ]:
with codecs.open('output_file_user_20000.txt', 'w', encoding='utf_8') as out_file:

	with codecs.open(all_user_filepath, encoding='utf_8') as review_file:

		for line in review_file:
			# reduce text if too long
			review_text = line

			if len(review_text) > 100000:
    				review_text = review_text[:100000]
			
			# parse the review text with spaCy			
			parsed_review = nlp(review_text)
		 
			# lemmatize the text and remove punctuation and whitespace
			unigram_review = [token.lemma_ for token in parsed_review
				      if not punct_space(token)]
		    
			# apply the first-order and secord-order phrase models
			bigram_review = bigram_model[unigram_review]
			trigram_review = trigram_model[bigram_review]
		    
			# remove any remaining stopwords
			trigram_review = [term for term in trigram_review
				      if not term in spacy.lang.en.stop_words.STOP_WORDS]
		    
			# create a bag-of-words representation
			review_bow = trigram_dictionary.doc2bow(trigram_review)
		    
			 # create an LDA representation
			review_lda = lda[review_bow]
		    
			# sort with the most highly related topics first
			review_lda = sorted(review_lda, key=lambda review_lda: -review_lda[1])
		
			df = pd.DataFrame(columns=["topic_number", "freq"])

			for topic_number, freq in review_lda:
			    df = df.append({
			    "topic_number": topic_number,
			    "freq":  round(freq, 4)
			    }, ignore_index=True)
			    #merge with complete topic list and replace na with zero
			df_full = pd.merge(df_all_numbers, df, how='left', on=['topic_number'])
			df_full = df_full.fillna(0)

		
			one_row = df_full['freq']
		
			out = one_row.values.tolist()
			#out.insert(0, bus)
			out2 = str(out)
			out3 = re.sub(r"[\[ | \]]", "", out2)
			    
			    #print(busi)
			out_file.write(out3 + '\n')
			#wr.writerow(out)    
			#output_file.write(out3)
			#output_file.write('\n')



