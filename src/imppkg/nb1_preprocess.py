# Path_to_SentimentArcs = "/gdrive/MyDrive/sentimentarcs_notebooks/" #@param ["/gdrive/MyDrive/sentiment_arcs/"] {allow-input: true}


# #@markdown Set this to the project root in your <b>GDrive folder</b>
# #@markdown <br> (e.g. /<wbr><b>gdrive/MyDrive/research/sentiment_arcs/</b>)

# #@markdown <hr>

# #@markdown **Which type of texts are you cleaning?** \

# Corpus_Genre = "novels" #@param ["novels", "social_media", "finance"]

# # Corpus_Type = "reference" #@param ["new", "reference"]
# Corpus_Type = "new" #@param ["new", "reference"]


# Corpus_Number = 5 #@param {type:"slider", min:0, max:10, step:1}


# #@markdown Put in the corresponding Subdirectory under **./text_raw**:
# #@markdown <li> All Texts as clean <b>plaintext *.txt</b> files 
# #@markdown <li> A <b>YAML Configuration File</b> describing each Texts

# #@markdown Please verify the required textfiles and YAML file exist in the correct subdirectories before continuing.

# print('Current Working Directory:')
# %cd $Path_to_SentimentArcs

# print('\n')

# if Corpus_Type == 'reference':
#   SUBDIR_TEXT_RAW = f'text_raw_{Corpus_Genre}_reference'
#   SUBDIR_TEXT_CLEAN = f'text_clean_{Corpus_Genre}_reference'
# else:
#   SUBDIR_TEXT_RAW = f'text_raw_{Corpus_Genre}_{Corpus_Type}_corpus{Corpus_Number}/'
#   SUBDIR_TEXT_CLEAN = f'text_clean_{Corpus_Genre}_{Corpus_Type}_corpus{Corpus_Number}/'

# PATH_TEXT_RAW = f'./text_raw/{SUBDIR_TEXT_RAW}'
# PATH_TEXT_CLEAN = f'./text_clean/{SUBDIR_TEXT_CLEAN}'

# # TODO: Clean up
# SUBDIR_TEXT_CLEAN = PATH_TEXT_CLEAN

# print(f'SUBDIR_TEXT_RAW:\n  [{SUBDIR_TEXT_RAW}]')
# print(f'PATH_TEXT_RAW:\n  [{PATH_TEXT_RAW}]')

# print(f'SUBDIR_TEXT_CLEAN:\n  [{SUBDIR_TEXT_CLEAN}]')
# print(f'PATH_TEXT_CLEAN:\n  [{PATH_TEXT_CLEAN}]')

# # %% [markdown]
# # # **[STEP 2] Automatic Configuration/Setup**

# # %%
# # Add PATH for ./utils subdirectory

# import sys
# import os

# !python --version

# print('\n')

# PATH_UTILS = f'{Path_to_SentimentArcs}utils'
# PATH_UTILS

# sys.path.append(PATH_UTILS)

# print('Contents of Subdirectory [./sentiment_arcs/utils/]\n')
# !ls $PATH_UTILS

# # More Specific than PATH for searching libraries
# # !echo $PYTHONPATH

# # %%
# # Review Global Variables and set the first few

# import global_vars as global_vars

# global_vars.SUBDIR_SENTIMENTARCS = Path_to_SentimentArcs
# global_vars.Corpus_Genre = Corpus_Genre
# global_vars.Corpus_Type = Corpus_Type
# global_vars.Corpus_Number = Corpus_Number

# global_vars.SUBDIR_TEXT_RAW = SUBDIR_TEXT_RAW
# global_vars.PATH_TEXT_RAW = PATH_TEXT_RAW

# dir(global_vars)

# # %% [markdown]
# # ## (each time) Custom Libraries & Define Globals

# # %%
# import os
# from pathlib import Path

# # %%
# # Initialize and clean for each iteration of notebook

# global_vars.corpus_texts_dt = {}
# global_vars.corpus_titles_dt = {}

# # %%
# # Import SentimentArcs Utilities to define Directory Structure
# #   based the Selected Corpus Genre, Type and Number

# !pwd 
# print('\n')

# # from utils import sa_config # .sentiment_arcs_utils
# from utils import sa_config

# print('Objects in sa_config()')
# print(dir(sa_config))
# print('\n')

# # Directory Structure for the Selected Corpus Genre, Type and Number
# sa_config.get_subdirs(Path_to_SentimentArcs, Corpus_Genre, Corpus_Type, Corpus_Number, 'none')


# # %%
# # TODO: fix
# # global_vars.SUBDIR_GRAPHS = './graphs/graphs_novels_new_corpus5/'

# # %%
# # Verify destination directories (create if they don't exist)
 
# print(f'SUBDIR_TEXT_RAW: {global_vars.SUBDIR_TEXT_RAW}')
# print('\n')
# print(f'SUBDIR_TEXT_CLEAN: {global_vars.SUBDIR_TEXT_CLEAN}')
# print('\n')
# print(f'SUBDIR_SENTIMENT_RAW: {global_vars.SUBDIR_SENTIMENT_RAW}')
# print('\n')
# print(f'SUBDIR_SENTIMENT_CLEAN: {global_vars.SUBDIR_SENTIMENT_CLEAN}')
# print('\n')
# print(f'SUBDIR_GRAPHS: {global_vars.SUBDIR_GRAPHS}')
# print('\n')

# # %%
# # Test if this is a new corpus by checking if destination subdirs exist,
# #   if they do not exist, automatically create all the folders text-raw/clean(2), sentiment-raw/clean(2), graphs(1)

# os.chdir(Path_to_SentimentArcs)

# if not os.path.exists(global_vars.SUBDIR_TEXT_RAW):
#     os.makedirs(global_vars.SUBDIR_TEXT_RAW)

# if not os.path.exists(global_vars.SUBDIR_TEXT_CLEAN):
#     os.makedirs(global_vars.SUBDIR_TEXT_CLEAN)

# if not os.path.exists(global_vars.SUBDIR_SENTIMENT_RAW):
#     os.makedirs(global_vars.SUBDIR_SENTIMENT_RAW)

# if not os.path.exists(global_vars.SUBDIR_SENTIMENT_CLEAN):
#     os.makedirs(global_vars.SUBDIR_SENTIMENT_CLEAN)

# if not os.path.exists(global_vars.SUBDIR_GRAPHS):
#     os.makedirs(global_vars.SUBDIR_GRAPHS)

# # %%
# # Check if required YAML config file exists in global_vars.SUBDIR_TEXT_RAW directory
# #   if not, then halt execution here with warning and instructions

# os.chdir(Path_to_SentimentArcs)

# yaml_sample_file_str = f'sample_text_raw_{Corpus_Genre}_config_info.yaml'
# if Corpus_Type == 'new':
#   yaml_file_config_str = f'text_raw_{Corpus_Genre}_new_corpus{Corpus_Number}_info.yaml'
#   # subdir_config = f'{global_vars.SUBDIR_TEXT_RAW}{file_config}'
# elif Corpus_Type == 'reference':
#   yaml_file_config_str = f'text_raw_{Corpus_Genre}_reference_info.yaml'
#   # subdir_config = f'{global_vars.SUBDIR_TEXT_RAW}{file_config}'
# else:
#   print(f'ERROR: Illegal value for Corpus_Type={Corpus_Type}')

# yaml_config_path_str = f'./{global_vars.SUBDIR_TEXT_RAW[2:]}{yaml_file_config_str}'
# print(f'Looking in: {yaml_config_path_str}\n')

# yaml_file_pth = Path(yaml_config_path_str)
# if yaml_file_pth.is_file():
#   print(f'Found YAML config file: {yaml_config_path_str}\n')
# else:
#   yaml_sample_str = f'sample_text_raw_{Corpus_Genre}_config_info.yaml'
#   src_yaml_path_str = f'./config/{yaml_sample_str}'
#   src_yaml_pth = Path(src_yaml_path_str)
#   dest_yaml_path_str = f'./{global_vars.SUBDIR_TEXT_RAW[2:]}{yaml_sample_str}'
#   dest_yaml_pth = Path(dest_yaml_path_str)
#   print(f'Copying sample from src: {src_yaml_path_str}')
#   print(f'        to destination: {dest_yaml_path_str}')
#   dest_yaml_pth.write_text(src_yaml_pth.read_text()) #for text files
#   print(f'ERROR: Missing required YAML config file at {yaml_file_pth}')
#   print(f'\n\nINSTRUCTIONS:\n\n1. Goto subdir:\n    {dest_yaml_path_str}\n\n2. Edit the sample yaml config file there:\n    {yaml_sample_file_str}\n\n3. Rename this sample yaml config file:\n    {yaml_file_config_str}\n\n4. Rerun this code cell and all successive ones')


# # %%
# # Call SentimentArcs Utility to define Global Variables

# sa_config.set_globals()

# # Verify sample global var set
# print(f'MIN_PARAG_LEN: {global_vars.MIN_PARAG_LEN}')
# print(f'STOPWORDS_ADD_EN: {global_vars.STOPWORDS_ADD_EN}')
# print(f'TEST_WORDS_LS: {global_vars.TEST_WORDS_LS}')
# print(f'SLANG_DT: {global_vars.SLANG_DT}')

# # %% [markdown]
# # ## Configure Jupyter Notebook

# # %%
# # Configure Jupyter

# # To reload modules under development

# # Option (a)
# %load_ext autoreload
# %autoreload 2
# # Option (b)
# # import importlib
# # importlib.reload(functions.readfunctions)


# # Ignore warnings
# import warnings
# warnings.filterwarnings('ignore')

# # Enable multiple outputs from one code cell
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

# from IPython.display import display
# from IPython.display import Image
# from ipywidgets import widgets, interactive

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# # %% [markdown]
# # ## (each time) Read YAML Configuration for Corpus and Models 

# # %%
# # from utils import sa_config # .sentiment_arcs_utils

# import yaml

# from utils import read_yaml

# print('Objects in read_yaml()')
# print(dir(read_yaml))
# print('\n')

# # Directory Structure for the Selected Corpus Genre, Type and Number
# read_yaml.read_corpus_yaml(Corpus_Genre, Corpus_Type, Corpus_Number)

# print('SentimentArcs Model Ensemble ------------------------------\n')
# model_titles_ls = global_vars.models_titles_dt.keys()
# print('\n'.join(model_titles_ls))


# print('\n\nCorpus Texts ------------------------------\n')
# corpus_titles_ls = list(global_vars.corpus_titles_dt.keys())
# print('\n'.join(corpus_titles_ls))


# print(f'\n\nThere are {len(model_titles_ls)} Models in the SentimentArcs Ensemble above.\n')
# print(f'\nThere are {len(corpus_titles_ls)} Texts in the Corpus above.\n')
# print('\n')


# # %% [markdown]
# # ## Install Libraries

# # %%
# # Library to Read R datafiles from within Python programs

# # !pip install pyreadr

# # %%
# # Powerful Industry-Grade NLP Library

# !pip install -U spacy

# # %%
# # NLP Library to Simply Cleaning Text

# !pip install texthero

# # %%
# # Advanced Sentence Boundry Detection Pythn Library
# #   for splitting raw text into grammatical sentences
# #   (can be difficult due to common motifs like Mr., ..., ?!?, etc)

# !pip install pysbd

# # %%
# # Python Library to expand contractions to aid in Sentiment Analysis
# #   (e.g. aren't -> are not, can't -> can not)

# !pip install contractions

# # %%
# # Library for dealing with Emoticons (punctuation) and Emojis (icons)

# !pip install emot

# # %% [markdown]
# # ## Load Libraries

# # %%
# # Core Python Libraries

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# %matplotlib inline

# import re
# import string
# from datetime import datetime
# import os
# import sys
# import glob
# import json
# from pathlib import Path
# from copy import deepcopy

# # %%
# # More advanced Sentence Tokenizier Object from PySBD
# from pysbd.utils import PySBDFactory

# # %%
# # Simplier Sentence Tokenizer Object from NLTK
# import nltk 
# from nltk.tokenize import sent_tokenize

# # Download required NLTK tokenizer data
# nltk.download('punkt')

# # %%
# # Instantiate and Import Text Cleaning Ojects into Global Variable space
# import texthero as hero
# from texthero import preprocessing

# # %%
# # Expand contractions (e.g. can't -> can not)
# import contractions

# # Translate emoticons :0 and emoji icons to text
# import emot 
# emot_obj = emot.core.emot() 

# from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO

# # Test
# text = "I love python ☮ 🙂 ❤ :-) :-( :-)))" 
# emot_obj.emoticons(text)

# # %%
# # Import spaCy, language model and setup minimal pipeline

# import spacy

# nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner'])
# # nlp.max_length = 1027203
# nlp.max_length = 2054406
# nlp.add_pipe(nlp.create_pipe('sentencizer')) # https://stackoverflow.com/questions/51372724/how-to-speed-up-spacy-lemmatization

# # Test some edge cases, try to find examples that break spaCy
# doc= nlp(u"Apples and oranges are similar. Boots and hippos aren't.")
# print('\n')
# print("Token Attributes: \n", "token.text, token.pos_, token.tag_, token.dep_, token.lemma_")
# for token in doc:
#     # Print the text and the predicted part-of-speech tag
#     print("{:<12}{:<12}{:<12}{:<12}{:<12}".format(token.text, token.pos_, token.tag_, token.dep_, token.lemma_))

# print('\nAnother Test:\n')
# doc = nlp(u"Apples and oranges are similar. Boots and hippos aren't.")

# for token in doc:
#     print("{:<12}{:<30}{:<12}".format(token.text, token.lemma, token.lemma_))

# # %% [markdown]
# # ## (each time to customize) Define/Customize Stopwords

# # %%
# # Customize Default SpaCy English Stopword List

# # Verify English Stopword List

# stopwords_spacy_en_ls = nlp.Defaults.stop_words

# ','.join([x for x in stopwords_spacy_en_ls])

# stopwords_en_ls = stopwords_spacy_en_ls

# print(f'\n\nThere are {len(stopwords_spacy_en_ls)} default English Stopwords from spaCy\n')

# # [CUSTOMIZE] Stopwords to ADD or DELETE from default spaCy English stopword list
# # Define words to keep by removing from stopwords list
# LOCAL_STOPWORDS_DEL_EN = set(global_vars.STOPWORDS_DEL_EN).union(set(['a','an','the','but','yet']))
# print(f'    Deleting these stopwords: {LOCAL_STOPWORDS_DEL_EN}')
# # Define words to remove by adding to stopword list
# LOCAL_STOPWORDS_ADD_EN = set(global_vars.STOPWORDS_ADD_EN).union(set(['a','an','the','but','yet']))
# print(f'    Adding these stopwords: {LOCAL_STOPWORDS_ADD_EN}\n')

# stopwords_en_ls = list(set(stopwords_spacy_en_ls).difference(set(LOCAL_STOPWORDS_DEL_EN)).union(set(LOCAL_STOPWORDS_ADD_EN)))
# print(f'Final Count: {len(stopwords_en_ls)} Stopwords')

# # %% [markdown]
# # ## **Utility Functions**

# # %% [markdown]
# # ### (each time) Generate Convenient Data Lists

# # %%
# # Derive List of Texts in Corpus a)keys and b)full author and titles

# print('Dictionary: corpus_titles_dt')
# global_vars.corpus_titles_dt
# print('\n')

# corpus_texts_ls = list(global_vars.corpus_titles_dt.keys())
# print(f'\nCorpus Texts:')
# for akey in corpus_texts_ls:
#   print(f'  {akey}')
# print('\n')

# print(f'\nNatural Corpus Titles:')
# corpus_titles_ls = [x[0] for x in list(global_vars.corpus_titles_dt.values())]
# for akey in corpus_titles_ls:
#   print(f'  {akey}')


# # %%
# # Get Model Families of Ensemble

# from utils.get_model_families import get_ensemble_model_famalies

# global_vars.models_ensemble_dt = get_ensemble_model_famalies(global_vars.models_titles_dt)

# print('\nTest: Lexicon Family of Models:')
# global_vars.models_ensemble_dt['lexicon']

# # %% [markdown]
# # ### Text Cleaning 

# # %%
# # [VERIFY]: Texthero preprocessing pipeline

# hero.preprocessing.get_default_pipeline()



# # Create Default and Custom Stemming TextHero pipeline

# # Create a custom cleaning pipeline
# def_pipeline = [preprocessing.fillna
#                 , preprocessing.lowercase
#                 , preprocessing.remove_digits
#                 , preprocessing.remove_punctuation
#                 , preprocessing.remove_diacritics
#                 # , preprocessing.remove_stopwords
#                 , preprocessing.remove_whitespace]

# # Create a custom cleaning pipeline
# stem_pipeline = [preprocessing.fillna
#                 , preprocessing.lowercase
#                 , preprocessing.remove_digits
#                 , preprocessing.remove_punctuation
#                 , preprocessing.remove_diacritics
#                 , preprocessing.remove_stopwords
#                 , preprocessing.remove_whitespace
#                 , preprocessing.stem]
                   
# # Test: pass the custom_pipeline to the pipeline argument
# # df['clean_title'] = hero.clean(df['title'], pipeline = custom_pipeline)df.head()

# # %%
# # Test Text Cleaning Functions

# # Verify in SentimentArcs Root Directory
# os.chdir(Path_to_SentimentArcs)

# %run -i './utils/text_cleaners.py'

# test_suite_ls = ['text2lemmas',
#                  'text_str2sents',
#                  'textfile2df',
#                  'emojis2text',
#                  'all_emos2text',
#                  'expand_slang',
#                  'clean_text',
#                  'lemma_pipe'
#                  ]

# # test_suite_ls = []

# # Test: text2lemmas()
# if 'text2lemmas' in test_suite_ls:
#   text2lemmas('I am going to start studying more often and working harder.', lowercase=True, remove_stopwords=False)
#   print('\n')

# # Test: text_str2sents()
# if 'text_str2sents' in test_suite_ls:
#   text_str2sents('Hello. You are a great dude! WTF?\n\n You are a goat. What is a goat?!? A big lazy GOAT... No way-', pysbd_only=False) # !?! Dr. and Mrs. Elipses...', pysbd_only=True)
#   print('\n')

# # Test: textfile2df()
# if 'textfile2df' in test_suite_ls:
#   # ???
#   print('\n')

# # Test: emojis2text()
# if 'emojis2text' in test_suite_ls:
#   test_str = "Hilarious 😂. The feeling of making a sale 😎, The feeling of actually ;) fulfilling orders 😒"
#   test_str = emojis2text(test_str)
#   print(f'test_str: [{test_str}]')
#   print('\n')

# # Test: all_emos2text()
# if 'all_emos2text' in test_suite_ls:
#   test_str = "Hilarious 😂. The feeling :o of making a sale 😎, The feeling :( of actually ;) fulfilling orders 😒"
#   all_emos2text(test_str)
#   print('\n')

# # Test: expand_slang():
# if 'expand_slang' in test_suite_ls:
#   expand_slang('idk LOL you suck!')
#   print('\n')

# # Test: clean_text()
# if 'clean_text' in test_suite_ls:
#   test_df = pd.DataFrame({'text_dirty':['The RAin in SPain','WTF?!?! Do you KnoW...']})
#   clean_text(test_df, 'text_dirty', text_type='formal')
#   print('\n')

# # Test: lemma_pipe()
# if 'lemma_pipe' in test_suite_ls:
#   print('\nTest #1:\n')
#   test_ls = ['I am running late for a meetings with all the many people.',
#             'What time is it when you fall down running away from a growing problem?',
#             "You've got to be kidding me - you're joking right?"]
#   lemma_pipe(test_ls)
#   print('\nTest #2:\n')
#   texts = pd.Series(["I won't go and you can't make me.", "Billy is running really quickly and with great haste.", "Eating freshly caught seafood."])
#   for doc in nlp.pipe(texts):
#     print([tok.lemma_ for tok in doc])
#   print('\nTest #3:\n')
#   lemma_pipe(texts)


# # %%
# # Test Text Cleaning Functions

# %run -i './utils/text_cleaners.py'
# # from utils.text_cleaners import text2lemmas, text_str2sents, emojis2text, expand_slang, clean_text, lemma_pipe

# test_suite_ls = ['text2lemmas',
#                  'text_str2sents',
#                  'textfile2df',
#                  'emojis2text',
#                  'all_emos2text',
#                  'expand_slang',
#                  'clean_text',
#                  'lemma_pipe'
#                  ]

# # Comment out this line to active tests above
# # test_suite_ls = []


# # Test: text2lemmas()
# if 'text2lemmas' in test_suite_ls:
#   text2lemmas('I am going to start studying more often and working harder.', lowercase=True, remove_stopwords=False)
#   print('\n')

# # Test: text_str2sents()
# if 'text_str2sents' in test_suite_ls:
#   text_str2sents('Hello. You are a great dude! WTF?\n\n You are a goat. What is a goat?!? A big lazy GOAT... No way-', pysbd_only=False) # !?! Dr. and Mrs. Elipses...', pysbd_only=True)
#   print('\n')

# # Test: textfile2df()
# if 'textfile2df' in test_suite_ls:
#   # ???
#   print('\n')

# # Test: emojis2text()
# if 'emojis2text' in test_suite_ls:
#   test_str = "Hilarious 😂. The feeling of making a sale 😎, The feeling of actually ;) fulfilling orders 😒"
#   test_str = emojis2text(test_str)
#   print(f'test_str: [{test_str}]')
#   print('\n')

# # Test: all_emos2text()
# if 'all_emos2text' in test_suite_ls:
#   test_str = "Hilarious 😂. The feeling :o of making a sale 😎, The feeling :( of actually ;) fulfilling orders 😒"
#   all_emos2text(test_str)
#   print('\n')

# # Test: expand_slang():
# if 'expand_slang' in test_suite_ls:
#   expand_slang('idk LOL you suck!')
#   print('\n')

# # Test: clean_text()
# if 'clean_text' in test_suite_ls:
#   test_df = pd.DataFrame({'text_dirty':['The RAin in SPain','WTF?!?! Do you KnoW...']})
#   clean_text(test_df, 'text_dirty', text_type='formal')
#   print('\n')
# """
# # Test: lemma_pipe()
# if 'lemma_pipe' in test_suite_ls:
#   print('\nTest #1:\n')
#   test_ls = ['I am running late for a meetings with all the many people.',
#             'What time is it when you fall down running away from a growing problem?',
#             "You've got to be kidding me - you're joking right?"]
#   lemma_pipe(test_ls)
#   print('\nTest #2:\n')
#   texts = pd.Series(["I won't go and you can't make me.", "Billy is running really quickly and with great haste.", "Eating freshly caught seafood."])
#   for doc in nlp.pipe(texts):
#     print([tok.lemma_ for tok in doc])
#   print('\nTest #3:\n')
#   lemma_pipe(texts)
# """

# # %% [markdown]
# # ### File Functions

# # %%
# # Verify in SentimentArcs Root Directory
# os.chdir(Path_to_SentimentArcs)

# %run -i './utils/file_utils.py'
# # from utils.file_utils import *

# # %run -i './utils/file_utils.py'

# # TODO: Not used? Delete?
# # get_fullpath(text_title_str, ftype='data_clean', fig_no='', first_note = '',last_note='', plot_ext='png', no_date=False)

# # %% [markdown]
# # # **[STEP 3] Read in Corpus and Clean**

# # %% [markdown]
# # ## Create List of Raw Textfiles

# # %%
# global_vars.SUBDIR_SENTIMENTARCS

# # %%
# # TODO: Temp fix until print(f'Original: {SUBDIR_TEXT_RAW}\n')
# path_text_raw = './' + '/'.join(global_vars.SUBDIR_TEXT_RAW.split('/')[1:-1])
# print(f'path_text_raw: {path_text_raw}\n')
# # SUBDIR_TEXT_RAW = path_text_raw + '/'
# print(f'Full Path to Corpus text_raw: {global_vars.SUBDIR_SENTIMENTARCS}{global_vars.SUBDIR_TEXT_RAW[2:]}')

# # %%
# %whos list

# # %%
# corpus_texts_ls

# # %%
# """
# # DELETE: Already created lists of corpus texts/titles in Utility Functions Section above

# # Get a list of all the Textfile filename roots in Subdir text_raw

# # Verify in SentimentArcs Root Directory
# os.chdir(Path_to_SentimentArcs)

# corpus_titles_ls = list(global_vars.corpus_titles_dt.keys())

# print(f'Corpus_Genre: {global_vars.Corpus_Genre}')
# print(f'Corpus_Type: {global_vars.Corpus_Type}\n')

# # Build path to Corpus Subdir
# # TODO: Temp fix until print(f'Original: {SUBDIR_TEXT_RAW}\n')
# # path_text_raw = './' + '/'.join(SUBDIR_TEXT_RAW.split('/')[1:-1]) + '/' + SUBDIR_TEXT_RAW
# # path_text_raw = './text_raw' + global_vars.SUBDIR_TEXT_RAW
# path_text_raw = global_vars.SUBDIR_TEXT_RAW
# print(f'Corpus Subdir: {path_text_raw}')

# # Create a List (preprocessed_ls) of all preprocessed text files
# try:
#   # texts_raw_ls = glob.glob(f'{SUBDIR_TEXT_RAW}*.txt')
#   texts_raw_root_ls = glob.glob(f'{path_text_raw}/*.txt')
#   texts_raw_root_ls = [x.split('/')[-1] for x in texts_raw_root_ls]
#   texts_raw_root_ls = [x.split('.')[0] for x in texts_raw_root_ls]
# except IndexError:
#   raise RuntimeError('No *.txt files found')

# print(f'\ntexts_raw_root_ls:\n  {texts_raw_root_ls}\n')

# text_ct = 0
# for afile_root in texts_raw_root_ls:
#   # file_root = file_fullpath.split('/')[-1].split('.')[0]
#   text_ct += 1
#   print(f'{afile_root}: ') # {corpus_titles_dt[afile_root]}')

# print(f'\nThere are {text_ct} Texts defined in SentmentArcs [corpus_dt] and found in the subdir: [SUBDIR_TEXT_RAW]')

# """;

# # %%
# glob.glob(f'{path_text_raw}/*.txt')

# # %% [markdown]
# # ## Read and Segment into Sentences

# # %%
# corpus_texts_ls

# # %%
# %%time
# %%capture

# # Read all Corpus Textfiles and Segment each into Sentences

# # NOTE:   3m30s Entire Corpus of 25 
# #         7m30s Ref Corpus 32 Novels
# #         7m24s Ref Corpus 32 Novels
# #         1m00s New Corpus 2 Novels

# #        13m55s 17:00 @20220405 Finance FedBoard Gov Speeches 32M + EU Cent Bank SPeeches 38M
# #        16m59s 19:49 @20220405 Finance FedBoard Gov Speeches 32M + EU Cent Bank SPeeches 38M


# #           23s 22:21 @20220405 New Corpus1 2 Novels

# # Read all novel files into a Dictionary of DataFrames
# #   Dict.keys() are novel names
# #   Dict.values() are DataFrames with one row per Sentence

# # Continue here ONLY if last cell completed WITHOUT ERROR

# # anovel_df = pd.DataFrame()

# for i, file_root in enumerate(corpus_texts_ls):
#   file_fullpath = f'{global_vars.SUBDIR_TEXT_RAW}{file_root}.txt'
#   print(f'Processing Novel #{i}: {file_fullpath}') # {file_root}')
#   # fullpath_str = novels_subdir + asubdir + '/' + asubdir + '.txt'
#   # print(f"  Size: {os.path.getsize(file_fullpath)}")

#   global_vars.corpus_texts_dt[file_root] = textfile2df(file_fullpath)
  
# # corpus_dt.keys()

# # Verify First Text is Segmented into text_raw Sentences
# print('\n\n')

# # global_vars.corpus_texts_dt[corpus_titles_ls[0]].head()
# text_no = 0
# print(f'Verify sample segmented Text: \n    {corpus_texts_ls[text_no]}\n')
# global_vars.corpus_texts_dt[corpus_texts_ls[text_no]].head()


# # %% [markdown]
# # ## Clean Sentences

# # %%
# global_vars.corpus_texts_dt.keys()

# # %%
# %%time

# # NOTE: (no stem) 4m09s (24 Novels)
# #       (w/ stem) 4m24s (24 Novels)


# #         4m10s 17:00 @20220405 Finance FedBoard Gov Speeches 32M + EU Cent Bank SPeeches 38M
# #         4m56s 19:49 @20220405 Finance FedBoard Gov Speeches 32M + EU Cent Bank SPeeches 38M

# #           23s 22:21 @20220405 New Corpus1 2 Novels


# i = 0

# for key_novel, atext_df in global_vars.corpus_texts_dt.items():

#   print(f'Processing Novel #{i}: {key_novel}...')

#   atext_df['text_clean'] = clean_text(atext_df, 'text_raw', text_type='formal')
#   atext_df['text_clean'] = lemma_pipe(atext_df['text_clean'])
#   atext_df['text_clean'] = atext_df['text_clean'].astype('string')

#   # TODO: Fill in all blank 'text_clean' rows with filler semaphore
#   atext_df.text_clean = atext_df.text_clean.fillna('empty_placeholder')

#   atext_df.head(2)

#   print(f'  shape: {atext_df.shape}')

#   i += 1

# # %%
# # Verify the first Text in Corpus is cleaned

# text_no = 0
# global_vars.corpus_texts_dt[corpus_texts_ls[text_no]].head(10)
# global_vars.corpus_texts_dt[corpus_texts_ls[text_no]].tail(10)
# global_vars.corpus_texts_dt[corpus_texts_ls[text_no]].info()

# # %% [markdown]
# # ## Save Cleaned Corpus

# # %%
# # Verify in SentimentArcs Root Directory
# os.chdir(Path_to_SentimentArcs)

# print('Currently in SentimentArcs root directory:')
# !pwd

# # Verify Subdir to save Cleaned Texts and Texts into..

# print(f'\nSaving Clean Texts to Subdir: {SUBDIR_TEXT_CLEAN}')
# print(f'\nSaving these Texts:\n  {global_vars.corpus_texts_dt.keys()}')

# # %%
# # Save the cleaned Textfiles

# i = 0
# for key_novel, anovel_df in global_vars.corpus_texts_dt.items():
#   anovel_fname = f'{key_novel}.csv'

#   anovel_fullpath = f'{SUBDIR_TEXT_CLEAN}{anovel_fname}'
#   print(f'Saving Novel #{i}: {key_novel}\n  to {anovel_fullpath}')
#   global_vars.corpus_texts_dt[key_novel].to_csv(anovel_fullpath)
#   i += 1

# # %%
# # Verify files were written

# !ls -altr $PATH_TEXT_CLEAN

# # %% [markdown]
# # # **[END OF NOTEBOOK]**


