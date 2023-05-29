import os
import pandas as pd


def get_raw_dataset(dataset_type:str, dataset_path: str, max_examples:int = None) -> pd.DataFrame:
    """Retrieves the dataframe associated with a particular dataset name

    Args:
        dataset_type: 'poems' or 'nyt'
        dataset_path:
        max_examples:

    Raises:
        NotImplementedError: if the dataset_type is not recognized

    Returns:
        pd.DataFrame: DF form has columns=["text", "topic"]
    """

    # This is the dataframe we are going to fill
    # First column is the raw text, the other columns are metadata
    df = pd.DataFrame(columns=["text", "topic"])        

    if dataset_type == "poems":
        return get_dataset_poems(dataset_path, df, max_examples)
    elif dataset_type == "nyt":
        return get_dataset_nyt(dataset_path, df, max_examples)
    elif dataset_type=='stories':
        return get_dataset_stories(dataset_path, pd.DataFrame(columns=['text','topic']), max_examples)
    else:
        raise NotImplementedError(f"Dataset type {dataset_type} not implemented")


def read_text_file(filepath:str):
    """
    Reads a text file and returns the contents

    Args:
        filepath (str):

    Returns:
        success (bool)
        content (str)
    """
    success = True
    content = ""
    try:
         f = open(filepath, 'r', encoding="utf8")
         content = f.read()
    except OSError as e:
        print(f"\nUnable to open {filepath}: {e}")
        success = False
    
    return success, content


def get_dataset_nyt(path_to_dataset:str, df:pd.DataFrame, max_examples:int = 10):
    """
    Loads the NYT Articles Dataset

    Args:
        path_to_dataset (str): this is a .csv file
        df (pd.DataFrame): df to fill
        max_examples (int, optional): Defaults to 10.

    Returns:
        pd.DataFrame: df filled
    """
    if not path_to_dataset.endswith(".csv"):
        raise ValueError(f"Error, the path to nyt dataset should be a .csv file, it was: {path_to_dataset}")
    
    df_read = pd.read_csv(path_to_dataset)
    columns_to_fill = df.columns
    for c in columns_to_fill:
        df[c] = df_read[c]
    
    if max_examples is not None and (1 <= max_examples <= len(df)):
        return df.sample(n = max_examples)
    else:
        return df

def get_dataset_stories(path_to_dataset:str, df:pd.DataFrame, max_examples:int=10):
    texts = os.listdir(path_to_dataset)
    # df = pd.DataFrame(columns=['text',''])

    for text in texts:
        success, content = read_text_file(os.path.join(path_to_dataset, text))
        if success:
            new_row = {'text':content, 'topic':'no topic'}
            df.loc[len(df)] = new_row
    return df 


def get_dataset_poems(path_to_dataset:str, df:pd.DataFrame, max_examples:int = 10) -> pd.DataFrame:
    """
    Loads the Poems Dataset\\
    Source: https://www.kaggle.com/code/kerneler/starter-poems-dataset-nlp-653c215f-7/notebook

    Args:
        path_to_dataset (str)
        df (pd.DataFrame): df to fill
        max_examples (int, optional): Defaults to 10.

    Returns:
        pd.DataFrame: df filled
    """

    # Dataset folder structure:
    # topics/
    #   alone/
    #       doc1.txt
    #       doc2.txt
    #       ...    
    #   america/
    #   ...

    topics = os.listdir(os.path.join(path_to_dataset, "topics"))
    for topic in topics:
        
        topic_path = os.path.join(path_to_dataset, "topics", topic)

        #Avoiding error due to meta files such as .DS_Store
        if not os.path.isdir(topic_path): continue
        
        # Assemble the docs by type
        list_docs = os.listdir(topic_path)
        
        # Get the list of docs of this subfolder
        for doc in list_docs:
            doc_path = os.path.join(path_to_dataset, "topics", topic, doc)
            success_read, content = read_text_file(doc_path)
            
            # Add the file if it was successfully read
            if success_read:
                new_row = {"text": content, "topic": topic}
                df.loc[len(df)] = new_row
                if max_examples is not None and len(df) == max_examples:
                    return df
    
    return df

def __nyt_to_csv():
    # Code for creating the nyt df for the first time from the big text file
    # https://www.kaggle.com/code/aneridalwadi/3x-accelerated-spacy-pipelines
    URL = []
    content = []
    flag = False
    with open(".//..//doc2img_data//nytimes_news_articles.txt", "r", encoding="utf8") as file:
        for line in file:
            if(flag):
                if line.startswith("URL: "):
                    # When current article ends
                    # Append all content, set the flag
                    flag = False
                    content.append(get_content)
                    get_content = []
                else:
                    # Store article content 
                    if line.strip():
                        get_content.append(line)
                
            if line.startswith("URL: "):
                # We are here when we encounter a *new article*
                get_content = []
                flag = True
                URL.append(line.strip().replace("URL: ", ""))

    # Since the last URL doesn't have any content, we remove it
    del URL[-1]

    # Create the df and save it
    df = pd.DataFrame({'text': content, 'topic': ["no topic"]*len(content)})
    df['text']= df['text'].str.join(' ')
    #df = df.sample(n=30) activate this line to create a sample
    df.to_csv("nyt.csv")

"""
if __name__ == "__main__":
    # Testing poems
    path = './mini_dataset/poems'
    df = get_raw_dataset("poems", path, max_examples=None)
    print(f"\nlen of df: {len(df)}")
    print(df.head())
    
    # Testing nyt
    #df = get_raw_dataset("nyt")
    #print(df.shape)
    #__nyt_to_csv()
"""
    

    

