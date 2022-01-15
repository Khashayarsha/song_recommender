# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 16:53:46 2020

@author: XHK
"""


from sklearn.model_selection import train_test_split
import time
import pandas as pd
import numpy as np
import gensim
import gensim.models.word2vec as w2v
import multiprocessing
import os
import re
import pprint
import sklearn.manifold
import matplotlib.pyplot as plt


file1 = r"entities_detected_azlyrics.csv"
file2 = r"entities_detected_songlyrics.csv"
file3 = r"entities_detected_vagalume.csv"


df1 = pd.read_csv(file1, usecols=[5],
                  header=None)
df2 = pd.read_csv(file2, low_memory=False,
                  usecols=[4],
                  lineterminator='\n',
                  header=None)
df3 = pd.read_csv(file3, low_memory=False,
                  usecols=[4],
                  lineterminator='\n',
                  header=None)


df1 = df1.iloc[2:]
df3 = df3.iloc[2:]  # ignore random stuff in the first rows


text_corpus = []

for song in df1[df1.columns[0]]:
    words = song.lower().split()
    text_corpus.append(words)

for song in df3[df3.columns[0]]:
    words = song.lower().split()
    text_corpus.append(words)


# Dimensionality of the resulting word vectors.
#
# arrived at 50 as an optimum
num_features = 50
# Minimum word count threshold
min_word_count = 1

num_workers = multiprocessing.cpu_count()
context_size = 7


downsampling = 1e-1

seed = 1

songs2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)

print("Building vocab ...")
songs2vec.build_vocab(text_corpus)
print(len(text_corpus))

start_time = time.time()

token_count = sum([len(sentence) for sentence in text_corpus])
print("Starting Training: ...")

# =============================================================================
songs2vec.train(text_corpus, total_examples=token_count,
                epochs=songs2vec.epochs)
#
# =============================================================================
# if not os.path.exists("trained"):
#     os.makedirs("trained")
#
# =============================================================================
save_path = r"C:\Users\XHK\Desktop\VU2020\Periode2\WDPS\word2vec\songs2vectors2.w2v"
songs2vec.save(save_path)
# =============================================================================
print("Done Training: ...")
print("--- %s seconds ---" % (time.time() - start_time))


#songs2vec = w2v.Word2Vec.load(os.path.join("trained", "songs2vectors.w2v"))
# songs2vec=gensim.downloader.load(save_path)
def songVector(row):
    vector_sum = 0
    words = row.lower().split()
    for word in words:
        vector_sum = vector_sum + songs2vec[word]
    vector_sum = vector_sum.reshape(1, -1)
    normalised_vector_sum = sklearn.preprocessing.normalize(vector_sum)
    return normalised_vector_sum


songs = pd.DataFrame()


start_time = time.time()

print("Vectorizing songs of df1 (AzLyrics)")

songs['song_vector1'] = df1[df1.columns[0]].apply(songVector)

print("Vectorizing songs of df3 (VagaLume)")
songs['song_vector2'] = df2[df2.columns[0]].apply(songVector)

song_vectors = []

train, test = train_test_split(songs, test_size=0.9)


for song_vector in train['song_vector1']:
    song_vectors.append(song_vector)
