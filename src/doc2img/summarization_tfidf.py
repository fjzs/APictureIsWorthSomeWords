from dataloader import get_raw_dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import string
import pandas as pd


def preprocess_df(df: pd.DataFrame):
    """Removes punctuation and stop words from the df

    Args:
        df (pd.DataFrame): df to preprocess
    """
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    stop_words.add("’")
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    preprocessed_sentences = []
    for i, row in df.iterrows():
        sent = row["text"].lower()
        sent = sent.replace("\\n","")
        sent = sent.replace("\n","")
        sent_nopuncts = sent.translate(translator)
        words_list = sent_nopuncts.strip().split()
        filtered_words = [word for word in words_list if word not in stop_words and len(word) != 1]
        preprocessed_sentences.append(" ".join(filtered_words))
    
    df["text_preprocessed"] = preprocessed_sentences
    

class SummarizerPoems:
    
    def __init__(self, df: pd.DataFrame, top_k:int=77):
        """Trains the model with this df

        Args:
            df (pd.DataFrame): df to train the model on
            top_k (int, optional): Max tokens to retrieve (defaults to 77)
        """
        self.top_k = top_k
        self.df = df
        self.summary = [None] * len(self.df) # this is a list of tokens
        self.__summarize()

    def get_summary_of_doc(self, doc:str, topic:str) -> str:
        """Retrieves the summary of a doc already seen

        Args:
            doc (str): the raw doc
            topic (str): the original topic

        Returns:
            str: a summary
        """
        assert type(doc) == str
        assert type(topic) == str

        topic_rows = self.df[self.df["topic"] == topic]
        print(topic_rows)


    def __summarize(self):
        
        # Preprocess the docs
        preprocess_df(self.df)
        docs = self.df["text_preprocessed"].tolist()
        #punctuation_set = list(string.punctuation)
        #punctuation_set.append("’")
        #stopset = list(set(stopwords.words('english') + punctuation_set))
        #count = CountVectorizer(tokenizer=word_tokenize, stop_words=stopset, analyzer="word")
        count = CountVectorizer(tokenizer=word_tokenize, analyzer="word")
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
            self.summary[i] = " ".join(reduced_text)
            

if __name__ == "__main__":
    
    """
    df = get_raw_dataset(max_examples=30)
    docs = df["text"].tolist()
    punctuation_set = list(string.punctuation)
    punctuation_set.append("’")
    stopset = list(set(stopwords.words('english') + punctuation_set))
    count = CountVectorizer(tokenizer=word_tokenize, stop_words=stopset, analyzer="word")
    bag_array = count.fit_transform(docs).toarray() #shape is [D,V], D is number of docs, V is number of tokens
    print(f"\nVector doc is type {type(bag_array)} and is: \n{bag_array}")
    
    vocabulary_to_index = count.vocabulary_
    index_to_vocabulary = [None] * len(vocabulary_to_index)
    for keyword in vocabulary_to_index:
        index = vocabulary_to_index[keyword]
        index_to_vocabulary[index] = keyword
    print(f"\nvocabulary_to_index is: \n{vocabulary_to_index}")
    print(f"\nindex_to_vocabulary is: \n{index_to_vocabulary}")   
    

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
    """

    # Example with the pipeline
    df = get_raw_dataset(dataset_name="poems")
    print(df.head())
    poemsSummary = SummarizerPoems(df)
    print(f"\nResults....\n")
    print(poemsSummary.df.head())
    for i in range(1):
        document = poemsSummary.df.loc[i,["text"]]
        topic = poemsSummary.df.loc[i,["topic"]]
        summary = poemsSummary.summary[i]
        print(f"Document:\n{document}")
        print(f"Summary:\n{summary}")
        print("_______________________________________")
        print(poemsSummary.get_summary_of_doc(str(document), str(topic)))

