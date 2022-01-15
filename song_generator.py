# python3 -m spacy download en_core_web_sm

import pandas as pd
from collections import defaultdict, Counter
import numpy as np
import lda, spacy
from math import log10
import sys
import ssl
from gensim.models import Word2Vec
import gensim.models.word2vec as w2v
import spacy
from spacy import displacy
import en_core_web_sm
import sklearn.manifold

#just to be sure I'm not missing any imports: 

import gensim
import gensim.models.word2vec as w2v
import multiprocessing
import os
import re
import pprint
import sklearn.manifold
import matplotlib.pyplot as plt







import nltk
from nltk.corpus import stopwords

def import_data(file):
    return pd.read_csv(file)

def cleaning(data):
    if 'ALink' in data:
        data['ALink'] = data['ALink'].str[1:-1]
        data['ALink'] = data['ALink'].str.replace('-', ' ')
        data = data.drop_duplicates(['ALink', 'SName'])
        data = data.dropna(axis='index', how='any')
        return data
    return data

def tokenize(text):
    return ''.join(filter(chars.__contains__, text)).split(' ')

def preprocess(tokens):
    result = []
    for token in tokens:
        if token not in stop_words:
            result.append(token.lower())
    return result

def create_inverted_index(inverted_index):
    for id in sorted(keys):
        temporary_dataframe = clean_data[clean_data['ALink'] == id[0]]
        temporary_dataframe = temporary_dataframe[temporary_dataframe['SName'] == id[1]]
        term_set = set(preprocess(tokenize(temporary_dataframe['Lyric'].values[0])))
        for term in term_set:
            inverted_index[term].append(id)
    return inverted_index

def create_short_term_list():
    short_term_list = [t for t in list(inverted_index.keys())]
    return short_term_list

def create_doc_term_matrix(doc_term_matrix):
    # Make a temporary dictionary object for fast lookup:
    short_term_map = {}
    for i in range(len(short_term_list)):
        short_term_map[short_term_list[i]] = i

    for d in range(0,len(doc_list)):
        key = doc_list[d]
        temporary_dataframe = clean_data[clean_data['ALink'] == key[0]]
        temporary_dataframe = temporary_dataframe[temporary_dataframe['SName'] == key[1]]
        for token in preprocess(tokenize(temporary_dataframe['Lyric'].values[0])):
            if (token in short_term_map):
                doc_term_matrix[d,short_term_map[token]] = True
    return doc_term_matrix

def show_topics(m):
    for i, topic_dist in enumerate(m.topic_word_):
        topic_words = np.array(short_term_list)[np.argsort(topic_dist)][:-11:-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))

def preprocess_query(query):
    return preprocess(tokenize(query))

def predict(query):

    query_term_matrix = np.zeros((1, len(short_term_list)), np.bool_)

    # Make a temporary dictionary object for fast lookup:
    short_term_map = {}
    for i in range(len(short_term_list)):
        short_term_map[short_term_list[i]] = i

    for token in query:
        if (token in short_term_map):
            query_term_matrix[0,short_term_map[token]] = True
    return query_term_matrix

def show_top_papers_on_topic(topic, query, artist):
    output_limit = 0
    result = []

    for d in sorted(doc_list_indexes, key=lambda i:doc_topic[i,topic], reverse=True):
        if output_limit == 10:
            break
        # if artist is specified, only check lyrics of songs by that artist
        if artist:
            if doc_list[d][0] != artist:
                continue
        lyric = preprocess(tokenize(return_lyric(doc_list[d][0],doc_list[d][1])))
        count = 0
        for word in query:
            if word in lyric:
                count +=1
        if count > 0:
            result.append(doc_list[d])
            output_limit += 1

    return result

def return_lyric(artist, song):
    artist_data = clean_data[clean_data['ALink']==artist]
    song_data = artist_data[artist_data['SName']==song]
    return song_data['Lyric'].values[0]

def tf(t,d):
    return float(tf_matrix[d][t])

def df(t):
    return float(len(inverted_index[t]))

def idf(t):
    return log10((num_documents + 1)/(df(t) + 1))

def tfidf(t,d):
    return tf(t,d) * idf(t)


def or_merge(sorted_list1, sorted_list2):
    merged_list = []
    list1 = list(sorted_list1)
    list2 = list(sorted_list2)
    while (True):
        if (not list1):
            merged_list.extend(list2)
            break
        if (not list2):
            merged_list.extend(list1)
            break
        if (list1[0] < list2[0]):
            merged_list.append(list1[0])
            list1.pop(0)
        elif (list1[0] > list2[0]):
            merged_list.append(list2[0])
            list2.pop(0)
        else:
            merged_list.append(list1[0])
            list1.pop(0)
            list2.pop(0)
    return merged_list

def score_ntn_nnn(query_words, doc_id):
    score = 0
    for t in query_words:
        score += tfidf(t,doc_id)
    return score

def query_ntn_nnn(query_words):
    # query_words = preprocess(tokenize(query_string))
    first_word = query_words[0]
    remaining_words = query_words[1:]
    or_list = inverted_index[first_word]
    result = []
    for t in remaining_words:
        or_list = or_merge(or_list, inverted_index[t])
    for song_id in sorted(or_list, key=lambda i: score_ntn_nnn(query_words,i), reverse=True)[:10]:
        result.append(song_id)
        #print(song_id, ' score: ', score_ntn_nnn(query_words,song_id))

    return result

def weighted_average_output(lda_list, tfidf_list, w2v_list):
    position = sorted(list(range(10)), reverse = True)
    result1 = list(zip(lda_list, position))
    result2 = list(zip(tfidf_list, position))
    result3 = list(zip(w2v_list, position))
    result1.extend(result2)
    result1.extend(result3)
    
    test1 = defaultdict(list)
    for item in result1:
        if item[0] in test1:
            test1[item[0]][0] += item[1]
        else:
            test1[item[0]].append(item[1])
    final_output = sorted(test1.items(), key=lambda kv: kv[1], reverse = True)
    
    return final_output[:10]

# checks if the query contains a person. If a person is found, this entity is
# returned in the 'artist' variable. If no person is found, this variable remains
# None. If a person is found, they are removed from the query and the query
# is returned without them. If there is nothing more to the query than the artist
# this function simply prints the top 10 songs from that artist. Otherwise we
# search the remaining words against lyrics from songs only by that artist.
def check_if_person(query):
    nlp = en_core_web_sm.load()
    entities = nlp(query)
    print([(X.text, X.label_) for X in entities.ents])
    query = preprocess_query(query)
    new_query = query
    artist = None
    for X in entities.ents:
        # if there are entities in the query, determine if the entity is a person i.e. artist
        if X.label_ == 'PERSON':
            artist = X.text
            # remove the entity text from the query
            new_query = remove_person_from_query(X.text, query)
            # if there is no extra text apart from the artist, print the first
            # 10 songs of this artist as a result
            if not new_query:
                print('\nSong recommendations are:\n')
                artist_data = clean_data[clean_data['ALink']==X.text.lower()]
                i = 0
                for index, row in artist_data.iterrows():
                    if i < 10:
                        print(row['SName'], 'by ', row['ALink'])
                    i += 1
                print('------------------')
            break;
    return [artist, new_query]

def remove_person_from_query(entity_text, query):
    entity_text = tokenize(entity_text)
    new_query = []
    for word in query:
        if word not in entity_text:
            new_query.append(word)
    return new_query


#------------------BEGIN CLEANING OF DATA, CREATING MODELS --------------------#

data = import_data('lyrics-data.csv') # different file names: lyrics-data.csv | artists-data.csv
clean_data = cleaning(data) # "AC/DC" is displayed as "ac dc"
clean_data = clean_data #[:10000] #temporary reducing the dataset size for testing purposes
#clean_data

chars = set('abcdefghijklmnopqrstuvwxyz ')
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

keys = list(zip(clean_data['ALink'], clean_data['SName']))
sorted(keys)

inverted_index = defaultdict(list)
inverted_index = create_inverted_index(inverted_index)

tf_matrix = defaultdict(Counter)

doc_list = keys

all_lyrics = []

for doc_id in doc_list:
    tokens = preprocess(tokenize(return_lyric(doc_id[0], doc_id[1])))
    all_lyrics.append(tokens)
    tf_matrix[doc_id] = Counter(tokens)

# word2vec = Word2Vec(all_lyrics, min_count=2)

num_documents = float(len(doc_list))

short_term_list = create_short_term_list()

doc_term_matrix = np.zeros((len(doc_list), len(short_term_list)), np.bool_)
doc_term_matrix = create_doc_term_matrix(doc_term_matrix)

songtext_generator_model = lda.LDA(n_topics=100, n_iter=200, random_state=1)
songtext_generator_model.fit(doc_term_matrix);

doc_topic = songtext_generator_model.doc_topic_
doc_list_indexes = list(range(0,len(doc_list)))

# show_topics(songtext_generator_model)



# --------------Building Word2Vec model: --------------------------------------------------------------

text_corpus = all_lyrics    #all_lyrics is an array with arrays of tokens in it. Some tokens are  == ''

num_features = 100
# Minimum word count threshold.
min_word_count = 1


import multiprocessing
cpus = multiprocessing.cpu_count()


context_size = 9 # Context window size


downsampling = 1e-1

# Seed for the RNG, to make the results reproducible.
#random number generator
#deterministic, good for debugging
seed = 1
load_model = True
if not load_model:

    songs2vec = w2v.Word2Vec(
        sg=1,
        seed=seed,
        workers=cpus,
        size=num_features,
        min_count=min_word_count,
        window=context_size,
        sample=downsampling
    )
    print("Building vocab ...")
    songs2vec.build_vocab(text_corpus)
    print(len(text_corpus))    
    
    import time
    start_time = time.time()
    
    token_count = sum([len(sentence) for sentence in text_corpus])
    
    print("Done Building vocab: ...")
    print("--- %s seconds ---" % (time.time() - start_time))
    
    
    print("Starting Training: ...")
    
    # =============================================================================
    
    
    songs2vec.train(text_corpus,total_examples = token_count, epochs = songs2vec.epochs)
    print("Training done")
    
    save_path = r"C:\Users\XHK\Desktop\VU2020\Periode2\WDPS\word2vec\SongGenW2V_small.w2v"
    
    print('Saving trained w2v model to: ', save_path)
    songs2vec.save(save_path)
    print('Saving Done...')

if load_model:
    print('Loading pre-trained model from file...')
    songs2vec = w2v.Word2Vec.load("SongGenW2V_BIGGEST.w2v")
    print('Loading done')
# function adapted from https://www.kaggle.com/aayushchadha/using-word2vec-to-find-similar-songs 
def songVector(row, model_size = num_features):
    vector_sum = np.zeros(model_size)
    words = row.lower().split()
    for word in words:
        vector_sum = vector_sum + songs2vec[word]
    vector_sum = vector_sum.reshape(1,-1)
    normalised_vector_sum = sklearn.preprocessing.normalize(vector_sum)
    return normalised_vector_sum



load_vectorized_songs = False
if not load_vectorized_songs:
    
    songs = pd.DataFrame()
    joined_lyrics = []
    for sentence in all_lyrics:
        joined_lyrics.append(' '.join(sentence))
    
    songs['lyrics'] = joined_lyrics
    
    import time
    start_time = time.time()
    
    print("Vectorizing songs......")
    
    songs['song_vectors'] = songs['lyrics'].apply(songVector)
    songs.to_csv(r'song_vectors_big_test.csv')
    
    print('Done vectorizing')
    print('pickling...')
    songs.to_pickle(r"song_vectors_PICKLED_BIG", compression='infer', protocol=5)

if load_vectorized_songs:
    print('Loading vectorized songs from file...')
    songs = pd.DataFrame()
    loaded = pd.read_csv("song_vectors_BIGGEST.csv")
    songs['song_vectors'] = loaded['song_vectors']
    print('Loading done')
from sklearn.metrics.pairwise import cosine_similarity

def tokenize_query(query_string):
    query_words = preprocess(tokenize(query_string))
    return query_words

def return_similarity(song_vector, query_vector):
    simi = cosine_similarity(query_vector,song_vector)
    
    
    return simi

def get_topn(similarity_vector, n=10):
    top = []
    similarity_vector = np.array(similarity_vector)
    for i in range(n):
        max_index = np.argmax(similarity_vector)
        top.append((clean_data.iloc[max_index]['ALink'],clean_data.iloc[max_index]['SName']))          #    top.append(clean_data.iloc[max_index])
        similarity_vector[max_index] = 0   #set to 0 so the next max can be found
    
    return top
    


def get_similar_songs(query, list_size = 10):
    query = tokenize_query(query)
    query_vector = []
    valid_query = []
    for i in query:
        try:
            in_vocab = songs2vec[i]   #to check if all words in query are in vocabulary
            valid_query.append(i)
        except KeyError:
            print('word %s not in vocabulary' % i)
    query_vector = songVector(' '.join(valid_query))    #taking the average vector of the query 
    #simi = cosine_similarity(songs2vec['hello'].reshape(1,-1),songs2vec['bye'].reshape(1,-1))
    
    
    sim_vec = [] 
# =============================================================================
#     for i in range(len(songs['song_vectors'].values)):
#         sim_vec.append(return_similarity(songs['song_vectors'].iloc[i], query_vector))
# =============================================================================
    sim = pd.DataFrame()
    sim['sim'] = songs['song_vectors'].apply(return_similarity, query_vector = query_vector)
    topn = get_topn(sim['sim'], list_size)
# =============================================================================
#     for song_vector in songs['song_vectors']:
#         simi = cosine_similarity(songs2vec[query].reshape(1,-1),song_vector)
#         temp.append(simi)
# =============================================================================
    return topn
   

#----------------NOW LOOP REQUESTING SEARCH INPUT-------------------------#

while True:
    query = input("\n\nType search request: ")

    # first check named entity recognition for PERSON if searching for artist
    nre_result = check_if_person(query)
    # retrieve artist if any found, else None
    artist = nre_result[0]
    # if artist found, returns query without artist, else original query
    # i.e. adam lambert ghost town would return query = ['ghost', 'town'] with
    # adam lambert removed since it's no longer relevant
    query = nre_result[1]

    # if there is no query it means the only search was the name of the artist
    # and we have already printed 10 songs by that artist, so ask for next input
    if not query:
        continue
    
# =============================================================================
#     query = preprocess_query(query)
#     temp_query = query
#     for item in query:
#         try:
#             sim_words = songs2vec.wv.most_similar(item, topn =5)
#             #print(sim_words)    #for testing
#             for i in range(5):
#                 temp_query.append(sim_words[i][0])
#             print(temp_query)
#         except KeyError:
#             print('word %s not in vocabulary' % item)
# =============================================================================

    # nlp = en_core_web_sm.load()

    # query = temp_query

    # query = 'thunderstruck'
    prediction = predict(query) # 'ghost town', 'thunderstruck',
    result = songtext_generator_model.transform(prediction)
    #np.argmax(result)

    # print(' LDA model results:')
    lda_result = show_top_papers_on_topic(np.argmax(result), query, artist)
    # print(lda_result)
    #print('\n', 'TF-IDF result:')
    tfidf_result = query_ntn_nnn(query)
    
    w2v_result = get_similar_songs(query)

    combined_output = weighted_average_output(lda_result, tfidf_result, w2v_result)
    print('\nSong recommendations are:\n')
    for item in combined_output:
        print(item[0][1],'by ',item[0][0])
    print('------------------')
