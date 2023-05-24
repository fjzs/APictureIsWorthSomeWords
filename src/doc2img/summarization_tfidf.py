from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import string
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_df(df: pd.DataFrame):
    """Removes punctuation and stop words from the df

    Args:
        df (pd.DataFrame): df to preprocess
    """
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    punctuation = string.punctuation + '’' + '“' + '”'
    translator = str.maketrans(punctuation, ' ' * len(punctuation))
    preprocessed_sentences = []
    for i, row in df.iterrows():
        sent = str(row["text"]).lower()
        sent = sent.replace("\\n"," ")
        sent = sent.replace("\n"," ")
        sent_nopuncts = sent.translate(translator)
        words_list = sent_nopuncts.strip().split()
        filtered_words = [word for word in words_list if word not in stop_words and len(word) != 1]
        preprocessed_sentences.append(" ".join(filtered_words))    
    df["text_preprocessed"] = preprocessed_sentences
    

class SummarizerTFIDF:
    
    def __init__(self, df: pd.DataFrame, top_k:int=77):
        """Trains the model with this df

        Args:
            df (pd.DataFrame): df to train the model on
            top_k (int, optional): Max tokens to retrieve (defaults to 77)
        """
        self.top_k = top_k
        self.df = df
        self.__summarize()

    def get_summary_of_index(self, index:int) -> str:
        """Gets the summary of a given index

        Args:
            index (int): 

        Returns:
            str: summary
        """
        assert index >= 0
        assert index < len(self.df)
        row =  self.df.iloc[index] # this is a series        
        summary = row["summary"]
        assert type(summary) == str
        return summary

    def get_summary_of_dftest(self, df_test: pd.DataFrame) -> pd.DataFrame:
        """Generates a 'summary' column in the test df

        Args:
            df_test (pd.DataFrame):

        Returns:
            pd.DataFrame:
        """
        assert type(df_test) == pd.DataFrame

        N = len(df_test)
        summaries = [None]*N
        for index, row in df_test.iterrows():
            text = row['text']
            topic = row['topic']
            summary = self.get_summary_of_doc(text, topic)
            summaries[index] = summary
        df_test["summary"] = summaries
        return df_test



    def get_summary_of_doc(self, doc:str, topic:str=None) -> str:
        """Retrieves the summary of a doc already seen

        Args:
            doc (str): the raw doc
            topic (str): the original topic (optional)

        Returns:
            str: a summary
        """
        assert type(doc) == str
        assert topic is not None
        assert type(topic) == str

        topic_rows = self.df[self.df["topic"] == topic]
        row = topic_rows[topic_rows["text"] == doc] # this is a df

        if row.empty:
            return "doc not found!"
        else:            
            summary = row.iloc[0]["summary"]
            assert type(summary) == str            
            return summary

    def __summarize(self):
        """
        Generates the column "summary" in the df_train passed (self.df)
        """
        
        # Preprocess the docs
        print("Preprocessing...")
        preprocess_df(self.df)
        docs = self.df["text_preprocessed"].tolist()
        count = CountVectorizer(tokenizer=word_tokenize, analyzer="word")
        bag_array = count.fit_transform(docs).toarray()
        
        # Assemble the dictionary in both directions
        print("Getting vocabulary...")
        vocabulary_to_index = count.vocabulary_
        index_to_vocabulary = [None] * len(vocabulary_to_index)
        for keyword in vocabulary_to_index:
            index = vocabulary_to_index[keyword]
            index_to_vocabulary[index] = keyword

        # Generate the tf-idf
        print("Computing TFIDF...")
        tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=False)
        tfidf_matrix = tfidf.fit_transform(bag_array).toarray()
        assert len(tfidf_matrix) == len(docs)

        # Now get the top k keywords from each doc
        print("Assembling summaries...")
        summaries = [None] * len(docs)
        for i in range(len(tfidf_matrix)):
            row = tfidf_matrix[i]
            top_indices = np.argpartition(row, -self.top_k)[-self.top_k:]
            reduced_text = [index_to_vocabulary[j] for j in top_indices]
            summaries[i] = " ".join(reduced_text)
        self.df["summary"] = summaries
            
"""
if __name__ == "__main__":    
    
    # Example with the pipeline
    from dataloader import get_raw_dataset    
    import yaml
    from summarization import get_summary    

    config_file = './src/config.yaml'
    with open(config_file) as cf_file:
        config = yaml.safe_load( cf_file.read())

    df_mini = get_raw_dataset("poems", "./mini_dataset/poems")
    df_full = get_raw_dataset("poems", "./../doc2img_data/poems")

    poemsSummary = SummarizerTFIDF(df_full)    
    #
    #for index, row in poemsSummary.df.iterrows():
    #    text = row['text']
    #    topic = row['topic']
    #    text_preprocessed = row['text_preprocessed']
    #    summary = poemsSummary.get_summary_of_index(index)        
    #    print(f"\nSummary:\n{summary}")
    
    df = get_summary(df_full, df_mini, config)
    df.to_csv("testDF.csv")
"""


