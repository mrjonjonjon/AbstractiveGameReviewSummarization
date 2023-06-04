
import gensim
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from gensim.corpora import Dictionary
from gensim.models import LdaModel    
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from collections import defaultdict
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import openai
import pandas as pd
import re
from wtpsplit import WtP
nltk.download('vader_lexicon')

#------PREPROCESSING--------#

def correct_sentences(sentences):
    '''takes a list of sentences(non tokenized) and corrects spelling and grammar using openai's text-davinci'''

    # Set up your OpenAI API credentials
    openai.api_key = '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'
    fixed_sentences = []
    for sentence in sentences:
        # Construct the prompt for language model
        prompt = f"Correct the following sentence: '{sentence}'\n\nFixed sentence:"

        # Generate corrected text using the language model
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            max_tokens=50,
            n=1,
            stop=None,
            temperature=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        # Extract the corrected text from the API response
        corrected_text = response.choices[0].text.strip()

        fixed_sentences.append(corrected_text)

    return fixed_sentences



def tokenize_sentence(sentence):
    '''tokenizes a single sentence.returns a list of tokens'''
    return word_tokenize(sentence)


def sent_tokenize(text):
    '''takes a string consisting of multiple sentences and returns a list of sentences'''
    try:
        #fixed_sentences = nltk.sent_tokenize(text.replace(".", ". ").replace("!", "! ").replace("?", "? "))
        wtp = WtP("wtp-bert-mini")
        # returns ["This is a test", "This is another test."]
        fixed_sentences=wtp.split(text)
    except Exception as e:
        print('ddd',text)
    return fixed_sentences
    

def tokenize_sentences(sentences):
    '''tokenizes a list of sentences'''
    #tokenized_sentences = [sentence.lower().split() for sentence in sentences]
    return [word_tokenize(sentence) for sentence in sentences]

def remove_stopword_tokenized_sentences(tokenized_sentences):
    '''removes stopwords from a list of tokenzied sentences(sentence = list of words)'''
    stopword_list = set(stopwords.words('english'))
    stopped_sentences = [
        [word for word in sentence if word not in stopword_list]
        for sentence in tokenized_sentences
    ]
    return stopped_sentences
    

def lemmatize_tokens(tokens):
    '''takes a tokenlist representing a sentence and returns a lemmatized tokenlist'''
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None
    lemmatizer = WordNetLemmatizer()
    tagged_tokens = nltk.pos_tag(tokens)  # Perform POS tagging
    lemmatized_tokens = []
    
    for token, tag in tagged_tokens:
        pos = get_wordnet_pos(tag)  # Map POS tag to WordNet POS tag
        if pos:  # If a valid WordNet POS tag is obtained
            lemma = lemmatizer.lemmatize(token, pos=pos)  # Lemmatize the token
        else:
            lemma = lemmatizer.lemmatize(token)  # Use default lemmatization without POS tag
        lemmatized_tokens.append(lemma)
    
    return lemmatized_tokens



#------END PREPROCESSING------------#


def set_to_aspect(aspect_to_set):
    '''
    takes a dict from an aspect word to a set and returns a dict from a set of words to an aspect word
    '''
    ans={}
    for word,set in aspect_to_set.items():
        ans[frozenset(set)]=word
    return ans

def categorize_sentences(original_dict):
    '''takes a dict from sentence to aspect and returns a dict from aspect to a list of sentences in that aspect'''
    categorized_dict = {}

    for sentence, category in original_dict.items():
        if category not in categorized_dict:
            categorized_dict[category] = []

        categorized_dict[category].append(sentence)

    return categorized_dict


def get_aspect(sentence,set_to_aspect):
    '''
    takes list of words and returns a set of all the string aspects to which it belongs
    '''
    aspect_matches=set()
    for word in sentence:
        for s,aspect in set_to_aspect.items():
            if word in s:
                aspect_matches.add(aspect)

    return aspect_matches

def get_cluster_aspects(tokenized_sentences,set_to_aspect):
    '''
    takes a list of tokenized sentences (presumably from the same LDA cluster) and returns a dictionary from each sentence to the aspect
    '''
    sents_to_aspect={}
    aspect_counts = defaultdict(int)
    
    remaining_sents=[]#0 matching aspects
    for tokenized_sentence in tokenized_sentences:
        aspects = get_aspect(tokenized_sentence,set_to_aspect)
        #print(aspects)
        if len(aspects)==1:
            aspect = aspects.pop()
            sents_to_aspect[tuple(tokenized_sentence)]=aspect
            aspect_counts[aspect]+=1
        elif len(aspects)==0:
            remaining_sents.append(tokenized_sentence)
        else:
            #TODO: FIX THIS LATER. SUPPOSED TO DISCARD
            aspect = aspects.pop()
            sents_to_aspect[tuple(tokenized_sentence)]=aspect
            aspect_counts[aspect]+=1
    #sometimes aspect_counts is empty because there were no keyword matches in the cluster. might need to expand keywordsets
    if len(aspect_counts)==0:
        for tokenized_sentence in remaining_sents:
            sents_to_aspect[tuple(tokenized_sentence)]='zero_match'
    else:
        majority_aspect = max(aspect_counts, key=aspect_counts.get)

        for tokenized_sentence in remaining_sents:
            sents_to_aspect[tuple(tokenized_sentence)]=majority_aspect
            
    return sents_to_aspect
        
def cluster_sentences(sentences,print_clusters=False):
        '''
        takes a list of sentences and returns a dictionary from  cluster id to a list of tokenized sentences in that cluster.
        clustering is based on LDA only
        '''
        # Tokenize sentences
        tokenized_sentences = tokenize_sentences(sentences)
        
        # Remove stopwords
        #tokenized_sentences = remove_stopword_tokenized_sentences(tokenized_sentences)

        # Create a dictionary mapping tokens to numeric IDs
        dictionary = Dictionary(tokenized_sentences)

        # Convert tokenized sentences into a Bag of Words (BoW) representation
        bow_corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_sentences]

        # Train the LDA model
        num_topics = 5  # Number of topics/clusters
        lda_model = LdaModel(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=20,iterations=10)

        # Get the topic distribution for each sentence
        sentence_topics = [lda_model[bow] for bow in bow_corpus]

        # Initialize the cluster data structure
        cluster_data = {i: [] for i in range(num_topics)}

        # Populate the cluster data structure with sentences
        for i, sentence_topic in enumerate(sentence_topics):
            dominant_topic = max(sentence_topic, key=lambda x: x[1])[0]  # Identify the dominant topic for the sentence
            cluster_data[dominant_topic].append(tokenized_sentences[i])    
            
        # Print the sentences in each cluster
        if print_clusters==True:
            for topic_id, cluster_sentences in cluster_data.items():
                print(f"Cluster {topic_id}: [{lda_model.show_topic(topic_id,topn=3)}]")
                for sentence in cluster_sentences:
                    print(sentence)
                print()
        return cluster_data


def expand_keywords_with_synonyms(keywords, max_expansions=10):
    expanded_keywords = set(keywords)  # Initialize with original keywords

    for keyword in keywords:
        synonyms = set()
        for syn in wordnet.synsets(keyword, pos=wordnet.NOUN):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
                if len(synonyms) >= max_expansions:  # Check if maximum expansions reached
                    break
            if len(synonyms) >= max_expansions:  # Check if maximum expansions reached
                break
        expanded_keywords.update(synonyms)

    return expanded_keywords


def pos_neg_sentiment(sentences):
    '''takes a list of tokenized sentences and returns the top 10% positive and top 1)% negative sentences, correcting them using gpt'''

    def calculate_sentiment_score(tokens):
        sentence = ' '.join(tokens)
        sid = SentimentIntensityAnalyzer()
        sentiment_scores = sid.polarity_scores(sentence)
        return sentiment_scores['compound']
    
    sentiment_scores = [calculate_sentiment_score(tokens) for tokens in sentences]

    sentences_with_scores = list(zip(sentences, sentiment_scores))
    sorted_sentences = sorted(sentences_with_scores, key=lambda x: x[1])

    num_sentences = len(sentences)
    num_positive_sentences = min(int(0.1 * num_sentences),5)  # Top 10% positive sentences
    num_negative_sentences = min(int(0.1 * num_sentences),5)  # Top 10% negative sentences

    top_positive_sentences = [sentence for sentence, score in sorted_sentences[-num_positive_sentences:]]
    top_negative_sentences = [sentence for sentence, score in sorted_sentences[:num_negative_sentences]]

  #  print("Top positive sentences:")
    #for sentence in top_positive_sentences:
     #   print(' '.join(sentence))

    #print("\nTop negative sentences:")
    #for sentence in top_negative_sentences:
    #    print(' '.join(sentence))
    return correct_sentences(top_positive_sentences),correct_sentences(top_negative_sentences)


def filter_reviews(data_frame):
    '''takes top 10% of most helpful reviews that are at least 100 words long'''
    # Read the Steam reviews dataframe
    data_frame = pd.read_csv('CleanData.csv', nrows=5000)

    # Calculate the threshold for the top 10% of helpful reviews
    threshold = data_frame['weighted_vote_score'].quantile(0.9)

    # Filter the dataframe to include only the top 10% of helpful reviews
    top_reviews = data_frame[data_frame['weighted_vote_score'] >= threshold]

    # Filter reviews by length and language
    min_review_length = 100  # Specify the minimum required length of a review
    filtered_reviews = top_reviews[
        (top_reviews['review'].str.len() >= min_review_length)
    ]

    # Apply language detection and filter out non-English reviews
    filtered_reviews = filtered_reviews[
        filtered_reviews['review'].apply(lambda x: len(x) >= min_review_length)
    ]
    return filtered_reviews

def replace_punctuation_with_unicode(text):
    # Find all occurrences of punctuation within quotation marks
    pattern = r'([`"\'‘’“”„«»])((?:(?!\1).)*)\1'
    matches = re.findall(pattern, text)

    # Replace punctuation with corresponding Unicode values
    for match in matches:
        quote_mark = match[0]
        content = match[1]
        replaced_text = ""
        for char in content:
            if char in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~':
                replaced_text += f'\\u{ord(char):04x}'
            else:
                replaced_text += char
        text = text.replace(f'{quote_mark}{content}{quote_mark}', f'{quote_mark}{replaced_text}{quote_mark}')

    return text


def pretty_print_dict(dictionary):
    formatted_dict = {str(key): value for key, value in dictionary.items()}
    print(json.dumps(formatted_dict, indent=4))

