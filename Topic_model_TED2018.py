import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models.wrappers import LdaMallet
from gensim.models.coherencemodel import CoherenceModel
from gensim import similarities


import os.path
import re
import glob
import numpy as np
from pprint import pprint

import pandas as pd
import matplotlib.pyplot as plt


os.environ['mallet_home'] = 'mallet-2.0.8'
mallet_home = os.environ.get('mallet_home',None)
mallet_path = os.path.join(mallet_home, 'bin', 'mallet')
import nltk
nltk.download('stopwords')

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

def load_data_from_dir(path):
    file_list = glob.glob(path + '/*.txt')

    # create document list:
    documents_list = []
    file_name=[]
    for filename in file_list:
        file_name.append(filename)
        with open(filename, 'r', encoding='utf8') as f:
            text = f.read()
            f.close()
            documents_list.append(text)            
    print("Total Number of Documents:",len(documents_list))
    file_name = [w.replace('ted-transcripts/transcripts\\', '') for w in file_name]
    return documents_list,file_name

def preprocess_data(doc_set,extra_stopwords = {}):
    # adapted from https://www.datacamp.com/community/tutorials/discovering-hidden-topics-python
    # replace all newlines or multiple sequences of spaces with a standard space
    doc_set = [re.sub('\s+', ' ', doc) for doc in doc_set]
    # initialize regex tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    # create English stop words list
    en_stop = set(stopwords.words('english'))
    #stemmer
    ps= PorterStemmer()
    #lemmatizer
    lemmatizer = WordNetLemmatizer() 
    # add any extra stopwords
    if (len(extra_stopwords) > 0):
        en_stop = en_stop.union(extra_stopwords)
    
    # list for tokenized documents in loop
    texts = []
    # loop through document list
    for i in doc_set:
        # clean and tokenize document string
        raw = i.lower()
        raw = lemmatizer.lemmatize(raw,pos='v')
        raw = ps.stem(raw)
        tokens = tokenizer.tokenize(raw)
        tokens = nltk.pos_tag(tokens)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # add tokens to list
        texts.append(stopped_tokens)
    return texts
def prepare_corpus(doc_clean):
    # adapted from https://www.datacamp.com/community/tutorials/discovering-hidden-topics-python
    # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
    dictionary = corpora.Dictionary(doc_clean)
    
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

    return dictionary,doc_term_matrix

def main():
	# adjust the path below to wherever you have the transcripts2018 folder
	document_list,file_name = load_data_from_dir("ted-transcripts/transcripts/")
	print(len(document_list))
	# I've added extra stopwords here in addition to NLTK's stopword list - you could look at adding others.
	doc_clean = preprocess_data(document_list,{'laughter','applause'})

	dictionary, doc_term_matrix = prepare_corpus(doc_clean)

	number_of_topics=0 # adjust this to alter the number of topics
	# words=20 #adjust this to alter the number of words output for the topic below

	# runs LDA using Mallet from gensim using the number_of_topics specified above - this might take a couple of minutes
	# you can create additional variables eg ldamallet to store models with different numbers of topics
	# ldamallet = LdaMallet(mallet_path, corpus=doc_term_matrix, num_topics=number_of_topics, id2word=dictionary)
	# gensimmodel = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldamallet)
	# coherencemodel = CoherenceModel(model=gensimmodel, texts=doc_clean, dictionary=dictionary, coherence='c_v')
	# coherence_lda = coherencemodel.get_coherence()
	# print('\nCoherence Score: ', coherence_lda)

	min_k = 5
	max_k = 15
	intervals = 5
	coherences = {}
	coherence_lda = {}
	max_coherence = 0
	for i in range(min_k, max_k, intervals):
		ldamallet = LdaMallet(mallet_path, corpus=doc_term_matrix, num_topics=i, id2word=dictionary)
		gensimmodel = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldamallet)
		coherences[i] = CoherenceModel(model=gensimmodel, texts=doc_clean, dictionary=dictionary, coherence='c_v')
		coherence_lda[i] = coherences[i].get_coherence()
		#identify best coherence score and save the model
		if coherence_lda[i]>max_coherence:
			max_coherence = coherence_lda[i]
			ldamalletbest = ldamallet
			gensimmodelbest = gensimmodel
			coherencebest = coherences[i]
			number_of_topics = i

	for k in coherence_lda:
		print('\nCoherence Score for topic count ',k,':',coherence_lda[k])

	print ('best coherence:',max_coherence)

	ldamalletbest.show_topics(num_topics=number_of_topics,num_words=20)
	ldamalletbest.print_topics()
	# convert the coherence scores to a pandas dataframe
	df = pd.DataFrame.from_dict(coherence_lda, orient='index', columns=['Coherence'])
	df['Topics'] = df.index

	# plot the result
	df.plot(kind='line', x='Topics', y='Coherence')
	plt.show()

	text_name ='2018-03-03-kriti_sharma_how_to_keep_human_biases_out_of_ai.txt' #name of file need to be checked
	#text_name ='2012-09-14-timothy_bartik_the_economic_case_for_preschool.txt' #name of file need to be checked

	doc_id = file_name.index(text_name) # index of document to explore
	print(file_name[3138])

	document_topics = gensimmodelbest.get_document_topics(doc_term_matrix[doc_id]) # substitute other models here
	document_topics = sorted(document_topics, key=lambda x: x[1], reverse=True) # sorts document topics

	model_doc_topics = gensimmodelbest.get_document_topics(doc_term_matrix) # substitute other models here
	lda_index = similarities.MatrixSimilarity(model_doc_topics.corpus)
    
	# query for our doc_id from above
	similarity_index = lda_index[doc_term_matrix[doc_id]]
	# Sort the similarity index
	similarity_index = sorted(enumerate(similarity_index), key=lambda item: -item[1])

	for i in range(1,6):
		document_id, similarity_score = similarity_index[i]
		print('Document Index: ',document_id)
		print('Document Name: ',file_name[document_id])
		print('Similarity Score',similarity_score)
		print(re.sub('\s+', ' ', document_list[document_id][:500]), '...') # preview first 500 characters
		print()

	# print('Loop done')
	# # convert the coherence scores to a pandas dataframe
	# df = pd.DataFrame.from_dict(coherence_lda, orient='index', columns=['Coherence'])
	# df['Topics'] = df.index
	# # plot the result
	# df.plot(kind='line', x='Topics', y='Coherence')
	# print('Done')
if __name__ == "__main__":
    main()