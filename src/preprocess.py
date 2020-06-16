from utils.preprocessor import Preprocessor
import pandas as pd
import numpy as np
import sys
import json
import re
import os
from nltk.tokenize.util import is_cjk
import argparse


OUTPUT_PATH = "data/processed/"
FILE_EXTENSION = ".json"
WRITE_MODE = "w"
READ_MODE = "r"


UNDESIRED_WORDS_LIST = ["https", "http", "www"]


def get_filename(path):
    lowercase_path = path.lower()

    filename_with_extension  = lowercase_path.split("/")[-1]

    filename_without_extension = re.sub(FILE_EXTENSION, "", filename_with_extension)

    return filename_without_extension


def remove_bots_posts(df):
    bots = ["AutoModerator", "RemindMeBot", "WikiTextBot", "youtubefactsbot", "RedditNiobioBot", "NemLiNemLereiBot"]

    df_without_bot_posts = df[~df.author.isin(bots)]

    return df_without_bot_posts


def is_jp_word(word):
    return any([ is_cjk(char) for char in word ])


def has_undesired_word(text):
    return any(list(map(lambda word: (is_jp_word(word) or word in UNDESIRED_WORDS_LIST), text)))


def remove_undesired_words(df):
    return df[df['body'].map(lambda text: has_undesired_word(text)) == False]



parser = argparse.ArgumentParser(description='Preprocessor for dataset preprocessing: stopwords removal, lemmatization, duplicate records filtering and etcetera.')

parser.add_argument('--datasetFile', type=str, help='dataset file', required=True)
parser.add_argument('--stopwordsFile', type=str, help='additional stop-words file', required=False)
parser.add_argument('--field', type=str, help='field to be processed', required=True)
parser.add_argument('--lang', type=str, help='dataset language: string "en" or "pt"', required=True)
parser.add_argument('--lemmatize', type=bool, help='should lemmatize data?', required=True)
parser.add_argument('--desiredPos', nargs='+', help='part-of-speech categories to keep. These are simple Spacy POS categories', required=True)

args = parser.parse_args()

original_data_path = args.datasetFile
stopwords_file = args.stopwordsFile
field_of_interest = args.field
lang = args.lang
lemmatize_activated = args.lemmatize
posCategories = args.desiredPos

print("Path: ", original_data_path)
if stopwords_file != None: print("Additional stopwords file: ", stopwords_file)
print("Field: ", field_of_interest)
print("Language: ", lang)
print("Is to lemmatize data? ", lemmatize_activated)
print("POS categories to keep: ", posCategories)

data_string = json.load(open(original_data_path, READ_MODE))

original_data_frame = pd.DataFrame.from_dict(data_string)

print(original_data_frame.head())

print("Original data row count: ", len(original_data_frame))

df_without_duplicates = original_data_frame.drop_duplicates(subset=['body'], keep='first')

print("Row count after duplicates removal: ", len(df_without_duplicates))

df_deleted_posts_removed = df_without_duplicates[df_without_duplicates.body != "[deleted]"]

print("Row count after deleted posts removal: ", len(df_deleted_posts_removed))

df_empty_posts_removed = df_deleted_posts_removed[df_deleted_posts_removed.body != ""]

print("Row count after empty posts removal: ", len(df_empty_posts_removed))

df_without_bot_posts = remove_bots_posts(df_empty_posts_removed)

print("Row count after bots' posts removal: ", len(df_without_bot_posts))

df_without_undesired_words = remove_undesired_words(df_without_bot_posts)

print("Row count after undesired words removal: ", len(df_without_undesired_words))

output_filepath = OUTPUT_PATH + get_filename(original_data_path) + "[duplicates_bots_removed]" + FILE_EXTENSION

os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

json.dump(df_without_undesired_words.to_dict(orient='records'), open(output_filepath, WRITE_MODE))

print("Data without duplicates dumped to ", output_filepath)

data = np.array(df_without_undesired_words[field_of_interest], dtype = 'object')

processor = Preprocessor(posCategories, lang, lemmatize_activated)

processed_data = processor.preprocess(data, stopwords_file)

print("Size of data after preprocessing: ", len(processed_data))

df_after_preprocessing = df_without_undesired_words.assign(body=processed_data)

df_after_removal_of_undesired_words = remove_undesired_words(df_after_preprocessing)

print("Row count after undesired words removal: ", len(df_without_undesired_words))

df_after_preprocessing = df_after_removal_of_undesired_words[df_after_removal_of_undesired_words['body'].map(lambda field: len(field)) > 0]

print(f'Row count after removal of rows with empty "{field_of_interest}" fields: {len(df_after_preprocessing)}')

output_filepath = OUTPUT_PATH + get_filename(original_data_path) + "[processed]" + FILE_EXTENSION

os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

json.dump(df_after_preprocessing.to_dict(orient='records'), open(output_filepath, WRITE_MODE))

print("Data dumped to ", output_filepath)
