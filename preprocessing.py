import json
import os
import re

import pandas as pd
from tqdm import tqdm


def preprocess_text(text):
    #text = rm_stopwords_from_text(text)
    if isinstance(text, list):
        # If text is a list, join its elements into a single string
        text = ' '.join(text)
        return text

    # Then, perform lowercasing
    text = text.lower()

    text = re.sub(r'\d', '', text)

    text = text.replace('-', ' ')
    text = text.replace('–', ' ')
    text = re.sub(r'[()/\"“”;:$%&*@#!?.,]', '', text)
    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r'\s+([.,])', r'\1', text)

    text = re.sub(r'\.(\S)', r'. \1', text)

    text = text.strip()

    return text



def load_file_to_json_list(filename):
    data = []
    with open(filename, 'r') as f:
        json_list = list(f)
        for json_str in tqdm(json_list, desc=f'Loading data {filename}'):
            d = json.loads(json_str)
            data.append(d)
    return data


def paragraph_to_text(raw_paragraph_list):
    new_paragraph_list = []
    for i, paragraph in enumerate(raw_paragraph_list):
        paragraph_list = []
        for sentence in paragraph:
            sentence = ' '.join(sentence)
            paragraph_list.append(sentence)

        new_paragraph = ''.join(paragraph_list)
        new_paragraph_list.append(new_paragraph)

    paragraph_str = ''.join(new_paragraph_list)
    return paragraph_str


def summary_to_text(raw_summary_list):
    summary_list = []
    for i, summary in enumerate(raw_summary_list):
        summary_list.append(' '.join(summary))

    summary_str = ''.join(summary_list)
    return summary_str


def alter_json_data(json_list_data, filename=''):
    new_json_list = []
    for jsonl_data in tqdm(json_list_data, desc=f'Altering json data {filename}'):
        jsonl_data['paragraphs'] = paragraph_to_text(jsonl_data['paragraphs'])
        #jsonl_data['id'] = id_to_text(jsonl_data['id'])
        jsonl_data['summary'] = summary_to_text(jsonl_data['summary'])

        new_json_list.append(jsonl_data)

    return new_json_list


def filter_columns(dataframe):
    # Memilih kolom yang ingin dipertahankan
    selected_columns = ['id', 'paragraphs', 'summary']

    # Menghapus kolom yang tidak termasuk dalam selected_columns
    filtered_dataframe = dataframe[selected_columns]

    return filtered_dataframe


def create_dataset(jsonl):
    header = list(jsonl[0].keys())
    dataset_list = []
    for json_data in jsonl:
        row = []
        for h in header:
            row.append(json_data[h])
        dataset_list.append(row)

    return header, dataset_list


def create_dataset_from_files(file_list):
    df_header = None
    dataset_list = []
    for filename in file_list:
        json_l = load_file_to_json_list(filename)
        new_json_l = alter_json_data(json_l, filename)
        header, dataset_part = create_dataset(new_json_l)

        if not df_header: df_header = header
        dataset_list.extend(dataset_part)

    df_full = pd.DataFrame().from_records(dataset_list)
    df_full = df_full.rename(columns=dict(enumerate(header)))
    return df_full

def remove_indexes(summary_array):
    remove_indexes = []
    for i in range(len(summary_array)):
        count = 0
        for j in summary_array[i]:
            if j != 0:
                count += 1
        if count == 2:
            remove_indexes.append(i)
    return remove_indexes