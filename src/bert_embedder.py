import os
import json
import math
import sys
import dask
from dask.distributed import Client
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import dask.multiprocessing
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer
from transformers import BertModel
from tqdm import tqdm

def load_csv(csv_file: str) -> pd.DataFrame:
    dtypes = {
        'id': str,
        'news': str
    }

    df = pd.read_csv(csv_file, dtype=dtypes)
    return df

def init_bert():
    ## Load pretrained model/tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=True)
    model = BertModel.from_pretrained('bert-base-multilingual-uncased', output_hidden_states=True)

    # Put the model in "evaluation" mode,meaning feed-forward operation.
    model.eval()

    return tokenizer, model

def get_vectors(row, tokenizer, model, file_name):
    encoding = tokenizer.encode(row['news'], add_special_tokens=True, max_length=512, truncation=True, padding="max_length")
    token_text = tokenizer.convert_ids_to_tokens(encoding)
    indexed_tokens = tokenizer.convert_tokens_to_ids(token_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])

    # Run the text through BERT, get the output and collect all of the hidden states produced from all 12 layers.
    with torch.no_grad():
        outputs = model(tokens_tensor)

        # Evaluating the model will return a different number of objects based on how it's  configured in the `from_pretrained` call earlier.
        # In this case, becase we set `output_hidden_states = True`, the third item will be the hidden states from all layers.
        # See the documentation for more details:https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]

        # initial embeddings can be taken from 0th layer of hidden states
        word_embed = hidden_states[0]
        save_as_npy(word_embed, file_name, row['id'])

        return word_embed

def save_as_npy(word_embed, file_name, id):
    output_path = './data/vectors/{file_name}/{id}.npy'.format(file_name=file_name, id=id)
    np.save(output_path, word_embed)

if __name__ == "__main__":
    print('Started')

    file_name = sys.argv[1]
    print('File name: ', file_name)
    
    csv_uri = f'./data/chunks/{file_name}.csv'
    df = load_csv(csv_uri)

    tokenizer, model = init_bert()

    parallel_tasks = []
    for idx, row in tqdm(df.iterrows()):
        task = dask.delayed(get_vectors)(row, tokenizer, model, file_name)
        parallel_tasks.append(task)

    print('Parallel Tasks: ', len(parallel_tasks))

    with ProgressBar():
        result = dask.compute(*parallel_tasks)
    
    print('Completed')  
