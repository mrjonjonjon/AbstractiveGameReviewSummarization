import pandas as pd
import util
import test_data
import pprint
#PIPELINE:

#embed sentences with tfidf
#cluster them with kmeans
#assign each sentence an aspect based on keyword matching
#train classififiers on the above
from nltk.tokenize import sent_tokenize

# Read the CSV file
data_frame = pd.read_csv('CleanData.csv',nrows=5000)#24,141,203 lines



#correct grammar and spelling
#x=util.correct_sentences(x)

x=util.filter_reviews(data_frame)['review'].to_list()
#decompose reviews into sentences
x=[util.sent_tokenize(z) for z in x]
x=[item for sublist in x for item in sublist]

#cluster sentences based on LDA
c=util.cluster_sentences(x,print_clusters=False)

#create a dictionary from aspect to list of sentences in that aspect
allsents_to_aspect={}
for cluster_id,tokenized_sentences in c.items():
    sent_to_aspect=util.get_cluster_aspects(tokenized_sentences,util.set_to_aspect(test_data.keyword_to_list))
    allsents_to_aspect.update(sent_to_aspect)
    
aspect_to_sents = util.categorize_sentences(allsents_to_aspect)
pos,neg = util.pos_neg_sentiment(aspect_to_sents['Gameplay'])
print('pos::::',pos)
print('neg::::',neg)