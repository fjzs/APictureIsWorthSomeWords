#Public Glove embeddings
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import nltk

nltk.download('averaged_perceptron_tagger')


#Remove '\n'
def preprocess(text):
    text.replace('\n',' ')
    return text

#get list of nouns
def get_nouns(text):
    is_noun = lambda pos: pos[:2] == 'NN'
    tokenized = nltk.word_tokenize(text)
    nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
    return nouns

def summarizer_cluster_center(df,n_clusters):
    
    print("Getting word embeddings")
    glove_file = '../doc2img_data/glove.6B/glove.6B.100d.txt'
    tmp_file = get_tmpfile("test_word2vec.txt")

    _ = glove2word2vec(glove_file, tmp_file)
    glove_model = KeyedVectors.load_word2vec_format(tmp_file)
    
    print("Computing nouns and clustering")
    N = len(df)
    summaries = [None]*N
    for df_index, row in df.iterrows():
        text = row['text']
        
        #preprocessing
        text = preprocess(text)
        
        #extracting nouns
        nouns = list(set(get_nouns(text)))
        
        #getting embeddings wherever possible
        feature_indices = {}
        features = []
        index = 0
        for noun in nouns:
            try:
                vec = glove_model.get_vector(noun)
                feature_indices[index] = noun
                features.append(vec)
                index += 1
            except:
                continue
                
        #clustering based on embeddings
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(features)
        centers = kmeans.cluster_centers_
        yhat = kmeans.fit_predict(np.array(features))
        
        #computing clusters with maximum words
        counts = [(i,list(yhat).count(i)) for i in range(n_clusters)]
        largest_clusters = sorted(counts, key = lambda x: x[1], reverse=True)[0:2]
        
        '''
        closest_words = []
        #Computing closest 5 words to center in 2 largest clusters
        for (cluster_no,cluster_size) in largest_clusters:
            cluster_indices = np.where(yhat==cluster_no)[0]
            cluster_words = [feature_indices[i] for i in cluster_indices]
            cluster_features = np.array([features[i] for i in cluster_indices])
            closest_indices = np.argsort(np.linalg.norm(cluster_features - centers[cluster_no], axis=1))[0:5]
            closest_words += [feature_indices[cluster_indices[i]] for i in closest_indices]
        summaries[df_index] = " ".join(closest_words)

        #getting nouns closest to cluster centers
        '''
        center_words = []
        for cluster_no in range(n_clusters):
            cluster_indices = np.where(yhat==cluster_no)[0]
            cluster_words = [feature_indices[i] for i in cluster_indices]
            cluster_features = np.array([features[i] for i in cluster_indices])
            center_index = np.argmin(np.linalg.norm(cluster_features - centers[cluster_no], axis=1))
            center_word = cluster_indices[center_index]
            center_words.append(feature_indices[center_word])
        
        #summary is a space separated list of center nouns
        summaries[df_index] = " ".join(center_words)
        
  
    df["summary"] = summaries
    return df