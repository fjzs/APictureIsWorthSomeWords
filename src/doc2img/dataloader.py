import os
import pandas as pd
import yaml


config_file = './config.yaml'

with open(config_file) as cf_file:
    config = yaml.safe_load( cf_file.read())

PATH_DATASET_POEMS = config['datasets']['poems']

def get_raw_dataset(dataset_name:str = "poems", max_examples:int = 10) -> pd.DataFrame:
    
    # This is the dataframe we are going to fill
    # First column is the raw text, the other columns are metadata
    df = pd.DataFrame(columns=["text", "topic"])

    if dataset_name == "poems":
        return get_dataset_poems(PATH_DATASET_POEMS, df, max_examples)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")


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
                if len(df) == max_examples:
                    return df
    
    return df


#if __name__ == "__main__":
#    df = get_raw_dataset(max_examples=10)
#    print(f"\nlen of df: {len(df)}")
#    print(df.head())