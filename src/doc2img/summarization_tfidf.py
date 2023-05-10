
from doc2img.dataloader import get_raw_dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import string
import yaml



class SummarizerPoems:
    
    def __init__(self, max_docs:int=100, top_k:int=77):
        self.top_k = top_k
        self.df = get_raw_dataset(max_examples=max_docs)
        self.summary = [None] * len(self.df) # this is a list of tokens
        self.__summarize()

    def __summarize(self):
        
        # Preprocess the docs
        docs = self.df["text"].tolist()
        punctuation_set = list(string.punctuation)
        stopset = list(set(stopwords.words('english') + punctuation_set))
        count = CountVectorizer(tokenizer=word_tokenize, stop_words=stopset, analyzer="word")
        bag_array = count.fit_transform(docs).toarray()
        
        # Assemble the dictionary in both directions
        vocabulary_to_index = count.vocabulary_
        index_to_vocabulary = [None] * len(vocabulary_to_index)
        for keyword in vocabulary_to_index:
            index = vocabulary_to_index[keyword]
            index_to_vocabulary[index] = keyword

        # Generate the tf-idf
        tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=False)
        tfidf_matrix = tfidf.fit_transform(bag_array).toarray()

        # Now get the top k keywords from each doc
        for i in range(len(tfidf_matrix)):
            row = tfidf_matrix[i]
            top_indices = np.argpartition(row, -self.top_k)[-self.top_k:]
            reduced_text = [index_to_vocabulary[j] for j in top_indices]
            self.summary[i] = reduced_text
            

"""
if __name__ == "__main__":
    
    df = get_raw_dataset(max_examples=2)
    docs = df["text"].tolist()
    #docs = ['The sun is shining airport',
    #     'The weather is sweet airport',
    #     'The airport is good',
    #     'The weather is shining']
    punctuation_set = list(string.punctuation)
    print({',' in punctuation_set})
    pass
    stopset = set(stopwords.words('english') + punctuation_set)
    

    count = CountVectorizer(tokenizer=word_tokenize, stop_words=list(stopset), analyzer="word")
    bag = count.fit_transform(docs)
    
    vocabulary_to_index = count.vocabulary_
    index_to_vocabulary = [None] * len(vocabulary_to_index)
    for keyword in vocabulary_to_index:
        index = vocabulary_to_index[keyword]
        index_to_vocabulary[index] = keyword
    print(f"\nvocabulary_to_index is: \n{vocabulary_to_index}")
    print(f"\nindex_to_vocabulary is: \n{index_to_vocabulary}")
    
    bag_array = bag.toarray()
    print(f"\nVector doc is type {type(bag_array)} and is: \n{bag_array}")

    np.set_printoptions(precision=2)
    tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=False)
    tfidf_matrix = tfidf.fit_transform(bag_array).toarray()
    print(f"\ntfidf is \n{tfidf_matrix}")

    # Now get the top k keywords from each doc
    print("\n")
    k=10
    for i in range(len(tfidf_matrix)):
        row = tfidf_matrix[i]
        top_indices = np.argpartition(row, -k)[-k:]
        reduced_text = [index_to_vocabulary[j] for j in top_indices]
        print(f"indices: {top_indices} -> {reduced_text}")
    
    N = 100
    poemsSummary = SummarizerPoems(max_docs=N, top_k=50)
    print(f"\nResults....\n")
    for i in range(5):
        document = poemsSummary.df.loc[i,["text"]]
        summary = poemsSummary.summary[i]
        print(f"Document:\n{document}")
        print(f"Summary:\n{summary}")
        print("_______________________________________")
        
