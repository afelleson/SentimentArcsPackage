# # %% [markdown]
# # # **Compute Sentiment Using 4 SyuzhetR and 7 SentimentR Models**
# # 
# # * https://www.youtube.com/watch?v=U3ByGh8RmSc
# # 
# # * https://github.com/ttimbers/intro-to-reticulate
# # 
# # [Use R on Google Colab!](https://colab.research.google.com/notebook#create=true&language=r)

# # %% [markdown]
# # # **[STEP 1] Manual Configuration/Setup**
# # 
# # 

# # %% [markdown]
# # ## (Popups) Connect Google gDrive

# # %%
# # [INPUT REQUIRED]: Authorize access to Google gDrive

# # Connect this Notebook to your permanent Google Drive
# #   so all generated output is saved to permanent storage there

# try:
#   from google.colab import drive
#   IN_COLAB=True
# except:
#   IN_COLAB=False

# if IN_COLAB:
#   print("Attempting to attach your Google gDrive to this Colab Jupyter Notebook")
#   drive.mount('/gdrive', force_remount=True)
# else:
#   print("Your Google gDrive is attached to this Colab Jupyter Notebook")

# # %% [markdown]
# # ## (3 Inputs) Define Directory Tree

# # %%
# # [CUSTOMIZE]: Change the text after the Unix '%cd ' command below (change directory)
# #              to math the full path to your gDrive subdirectory which should be the 
# #              root directory cloned from the SentimentArcs github repo.

# # NOTE: Make sure this subdirectory already exists and there are 
# #       no typos, spaces or illegals characters (e.g. periods) in the full path after %cd

# # NOTE: In Python all strings must begin with an upper or lowercase letter, and only
# #         letter, number and underscores ('_') characters should appear afterwards.
# #         Make sure your full path after %cd obeys this constraint or errors may appear.

# # #@markdown **Instructions**

# # #@markdown Set Directory and Corpus names:
# # #@markdown <li> Set <b>Path_to_SentimentArcs</b> to the project root in your **GDrive folder**
# # #@markdown <li> Set <b>Corpus_Genre</b> = [novels, finance, social_media]
# # #@markdown <li> <b>Corpus_Type</b> = [reference_corpus, new_corpus]
# # #@markdown <li> <b>Corpus_Number</b> = [1-20] (id nunmber if a new_corpus)

# #@markdown <hr>

# # Step #1: Get full path to SentimentArcs subdir on gDrive
# # =======
# #@markdown **Accept default path on gDrive or Enter new one:**

# Path_to_SentimentArcs = "/gdrive/MyDrive/sentimentarcs_notebooks/" #@param ["/gdrive/MyDrive/sentiment_arcs/"] {allow-input: true}


# #@markdown Set this to the project root in your <b>GDrive folder</b>
# #@markdown <br> (e.g. /<wbr><b>gdrive/MyDrive/research/sentiment_arcs/</b>)

# #@markdown <hr>

# #@markdown **Which type of texts are you cleaning?** \

# Corpus_Genre = "novels" #@param ["novels", "social_media", "finance"]

# # Corpus_Type = "reference" #@param ["new", "reference"]
# Corpus_Type = "new" #@param ["new", "reference"]


# Corpus_Number = 5 #@param {type:"slider", min:1, max:10, step:1}


# #@markdown Put in the corresponding Subdirectory under **./text_raw**:
# #@markdown <li> All Texts as clean <b>plaintext *.txt</b> files 
# #@markdown <li> A <b>YAML Configuration File</b> describing each Texts

# #@markdown Please verify the required textfiles and YAML file exist in the correct subdirectories before continuing.

# print('Current Working Directory:')
# %cd $Path_to_SentimentArcs

# print('\n')

# if Corpus_Type == 'reference':
#   SUBDIR_SENTIMENT_RAW = f'sentiment_raw_{Corpus_Genre}_reference'
#   SUBDIR_TEXT_CLEAN = f'text_clean_{Corpus_Genre}_reference'
# else:
#   SUBDIR_SENTIMENT_RAW = f'sentiment_raw_{Corpus_Genre}_{Corpus_Type}_corpus{Corpus_Number}/'
#   SUBDIR_TEXT_CLEAN = f'text_clean_{Corpus_Genre}_{Corpus_Type}_corpus{Corpus_Number}/'

# # PATH_SENTIMENT_RAW = f'./sentiment_raw/{SUBDIR_TEXT_RAW}'
# # PATH_TEXT_CLEAN = f'./text_clean/{SUBDIR_TEXT_CLEAN}'
# PATH_SENTIMENT_RAW = f'./sentiment_raw/{SUBDIR_SENTIMENT_RAW}'
# PATH_TEXT_CLEAN = f'./text_clean/{SUBDIR_TEXT_CLEAN}'

# # TODO: Clean up
# # SUBDIR_TEXT_CLEAN = PATH_TEXT_CLEAN

# print(f'PATH_SENTIMENT_RAW:\n  [{PATH_SENTIMENT_RAW}]')
# print(f'SUBDIR_SENTIMENT_RAW:\n  [{SUBDIR_SENTIMENT_RAW}]')

# print('\n')

# print(f'PATH_TEXT_CLEAN:\n  [{PATH_TEXT_CLEAN}]')
# print(f'SUBDIR_TEXT_CLEAN:\n  [{SUBDIR_TEXT_CLEAN}]')

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

# global_vars.SUBDIR_SENTIMENT_RAW = SUBDIR_SENTIMENT_RAW
# global_vars.PATH_SENTIMENT_RAW = PATH_SENTIMENT_RAW

# global_vars.SUBDIR_TEXT_CLEAN = SUBDIR_TEXT_CLEAN
# global_vars.PATH_TEXT_CLEAN = PATH_TEXT_CLEAN

# dir(global_vars)

# # %% [markdown]
# # ## (each time) Custom Libraries & Define Globals

# # %%
# # Initialize and clean for each iteration of notebook

# # dir(global_vars)

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

# global_vars.corpus_titles_dt

# # %% [markdown]
# # ## Install Libraries: R

# # %%
# # !pip install rpy2

# # %%
# # !pip install -U rpy2

# # %%
# # Load Jupyter rpy2 Extension  
# #   enables the %%R magic commands

# %load_ext rpy2.ipython

# # %reload_ext rpy2.ipython

# # %%
# %%time 
# %%capture 
# %%R

# # Install Syuzhet.R, Sentiment.R and Utility Libraries

# # NOTE: 1m12s 
# #       1m05s

# #       1m13s 00:47 @20220406Wed Novels Corpus1 2 Novels

# install.packages(c('syuzhet', 'sentimentr', 'tidyverse', 'lexicon'))

# library(syuzhet)
# library(sentimentr)
# library(tidyverse)
# library(lexicon)

# # %%
# # Load Python libraries to exchange data with R Program Space and read R Datafiles

# import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr

# # %%
# %%R

# # Verify R in Kernel Version
# # R.version.string

# # Verfiy R Kernel Environment
# # Sys.getenv

# # Verify R Kernel Session Info
# sessionInfo()

# # %% [markdown]
# # ## Install Libraries: Python

# # %%
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# %matplotlib inline

# from glob import glob
# import copy
# import json # Installed above in YAML Configuration Section

# # %% [markdown]
# # ## Setup Matplotlib Style
# # 
# # * https://matplotlib.org/stable/tutorials/introductory/customizing.html

# # %%
# # Configure Matplotlib

# # View available styles
# # plt.style.available

# # Verify in SentimentArcs Root Directory
# os.chdir(Path_to_SentimentArcs)

# %run -i './utils/config_matplotlib.py'

# config_matplotlib()

# print('Matplotlib Configuration ------------------------------')
# print('\n  (Uncomment to view)')
# # plt.rcParams.keys()
# print('\n  Edit ./utils/config_matplotlib.py to change')

# # %% [markdown]
# # ## Setup Seaborn Style

# # %%
# # Configure Seaborn

# # Verify in SentimentArcs Root Directory
# os.chdir(Path_to_SentimentArcs)

# %run -i './utils/config_seaborn.py'

# config_seaborn()

# print('Seaborn Configuration ------------------------------\n')
# # print('\n  Update ./utils/config_seaborn.py to display seaborn settings')

# """
# # Seaborn: Set Context
# # sns.set_context("notebook")

# # Seaborn: Set Theme (Scale of Font)
# sns.set_theme('paper')  # paper, notebook, talk, poster

# # Seaborn: Set Style
# # sns.set_style('ticks') # darkgrid, whitegrid, dark, white, and ticks
# plt.style.use('seaborn-whitegrid')

# # sns.set_palette('tab10')
# # sns.color_palette()

# # sns.set_palette('tab10')
# # sns.color_palette()
# """;

# # %%
# """
# # Configure Seaborn

# # Verify in SentimentArcs Root Directory
# os.chdir(Path_to_SentimentArcs)

# %run -i './utils/config_seaborn.py'

# config_seaborn()

# print('Seaborn Configuration ------------------------------\n')
# # print('\n  Update ./utils/config_seaborn.py to display seaborn settings')
# # View previous seaborn configuration
# print('\n Old Seaborn Configurtion Settings:\n')
# sns.axes_style()
# print('\n\n')

# # Update and View new seaborn configuration
# print('\n New Seaborn Configurtion Settings:\n')
# # sns.set_style('white')
# sns.set_context('paper')
# sns.set_style('white')
# sns.set_palette('tab10')

# # Change defaults
# # sns.set(style='white', context='talk', palette='tab10')
# """;

# # %% [markdown]
# # ## Python Utility Functions

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
# # # **[STEP 3] Read all Preprocessed Novels**

# # %%
# !ls {Path_to_SentimentArcs}{PATH_TEXT_CLEAN}

# # %%
# # Verify cwd and subdir of Cleaned Corpus Texts

# print('Current Working Directory:')
# !pwd

# print(f'\nSubdir with all Cleaned Texts of Corpus:\n  {SUBDIR_TEXT_CLEAN}')

# print(f'\n\nFilenames of Cleaned Texts:\n')
# !ls -1 {Path_to_SentimentArcs}{PATH_TEXT_CLEAN}

# # %%
# # Create a List (preprocessed_ls) of all preprocessed text files

# # Verify in SentimentArcs Root Directory
# os.chdir(Path_to_SentimentArcs)

# try:
#     preprocessed_ls = glob(f'{PATH_TEXT_CLEAN}/*.csv')
#     preprocessed_ls = [x.split('/')[-1] for x in preprocessed_ls]
#     preprocessed_ls = [x.split('.')[0] for x in preprocessed_ls]
# except IndexError:
#     raise RuntimeError('No csv file found')

# print('\n'.join(preprocessed_ls))
# print('\n')
# print(f'Found {len(preprocessed_ls)} Preprocessed files in {SUBDIR_TEXT_CLEAN}')

# # %%
# # Read all preprocessed text files into master DataFrame (corpus_dt)

# corpus_texts_dt = {}

# for i,atext in enumerate(preprocessed_ls):
#   print(f'Processing #{i}: {atext}...')
#   afile_fullpath = f'{PATH_TEXT_CLEAN}/{atext}.csv'
#   print(f'               {afile_fullpath}')
#   atext_df = pd.read_csv(afile_fullpath, index_col=[0])
#   corpus_texts_dt[atext] = atext_df

# # %%
# # Verify the Text read into master Dictionary of DataFrames

# corpus_texts_dt.keys()
# print('\n')
# print(f'There were {len(corpus_texts_dt)} preprocessed Text read into the Dict corpus_texts_dt')

# # %%
# # Check if there are any Null strings in the text_clean columns

# for i, atext in enumerate(list(corpus_texts_dt.keys())):
#   print(f'\nText #{i}: {atext}')
#   nan_ct = corpus_texts_dt[atext].text_clean.isna().sum()
#   if nan_ct > 0:
#     print(f'      {nan_ct} Null strings in the text_clean column')

# # %%
# corpus_texts_dt.keys()

# # %%
# # Fill in all the Null value of text_clean with placeholder 'empty_string'

# for i, atext in enumerate(list(corpus_texts_dt.keys())):
#   # print(f'Novel #{i}: {atext}')
#   # Fill all text_clean == Null with 'empty_string' so sentimentr::sentiment doesn't break
#   corpus_texts_dt[atext][corpus_texts_dt[atext].text_clean.isna()] = 'empty_string'

# # %%
# corpus_texts_ls

# # %%
# # Verify DataFrame of first Text in Corpus Dictionary

# corpus_texts_dt[corpus_texts_ls[0]].head()

# # %% [markdown]
# # # **[STEP 4] Get Sentiments with SyuzhetR (4 Models)**

# # %% [markdown]
# # ## Compute New SyuzhetR Values

# # %%
# corpus_texts_ls

# # %%
# for i, acorpus in enumerate(corpus_texts_ls):
#   corpus_texts_dt[acorpus].head()

# # %%
# # Verify text_clean of sample text

# corpus_texts_dt[corpus_texts_ls[0]]['text_clean'].to_list()[:10]

# # %%
# %%time

# # Compute Sentiments from all 4 Syuzhet Models applied to all 32 Novels (4 x 32 = 128 runs)

# # NOTE:  9m45s 23:30 on 20220114 Colab Pro (33 Novels)
# #       28:32s 21:06 on 20220226 Colab Pro (33 Novels)
# #        3m20s 19:11 on 20220217 Colab Pro (2 Novels)
# #        3m05s 19:17 on 20220217 Colab Pro (2 Novels)

# #        2m21s 00:57 @20220406Wed Colab Pro (2 Novels)

# #        1h29m 09:24 @20220406Wed Colab Pro (2 Financial Ref: Speeches FedGov & EU CenBank)

# #        3m05s 21:13 on 20220415 Colab Pro (3 Novels, 628k, 662k, 897k)
# #        3m05s 21:48 on 20220415 Colab Pro (3 Novels, 628k, 662k, 897k)
# #        6m03s 07:39 on 20220416 Colab Pro (3 Novels, 628k, 662k, 897k)

# #        1m31s 14:23 on 20220419 Colab Pro (1 Novels, 502k)

# # base = importr('base')
# syuzhet = importr('syuzhet')

# # corpus_syuzhetr_dt = {}

# # base.rank(0, na_last = True)
# texts_titles_ls = list(corpus_texts_dt.keys())
# texts_titles_ls.sort()
# for i, anovel in enumerate(texts_titles_ls):
#   print(f'Processing Novel #{i}: {anovel}...')
#   corpus_texts_dt[anovel]['syuzhetr_syuzhet'] = syuzhet.get_sentiment(corpus_texts_dt[anovel]['text_clean'].to_list(), method='syuzhet')
#   corpus_texts_dt[anovel]['syuzhetr_bing'] = syuzhet.get_sentiment(corpus_texts_dt[anovel]['text_clean'].to_list(), method='bing')
#   corpus_texts_dt[anovel]['syuzhetr_afinn'] = syuzhet.get_sentiment(corpus_texts_dt[anovel]['text_clean'].to_list(), method='afinn')
#   corpus_texts_dt[anovel]['syuzhetr_nrc'] = syuzhet.get_sentiment(corpus_texts_dt[anovel]['text_clean'].to_list(), method='nrc')

# # %%
# corpus_texts_dt[texts_titles_ls[0]].head()

# # %%
# # Verify First Text in Corpus has New SyuzhetR Columns with Plausible Values

# # corpus_texts_dt[next(iter(corpus_texts_dt))].head()

# corpus_texts_dt[texts_titles_ls[0]].head()
# corpus_texts_dt[texts_titles_ls[0]].info()

# # corpus_texts_dt[texts_titles_ls[1]].head()
# # corpus_texts_dt[texts_titles_ls[1]].info()

# # %% [markdown]
# # ## Checkpoint: Save SyuzhetR Values

# # %%
# %whos str

# # %%
# # Verify in SentimentArcs Root Directory
# #   and destination Subdir for Raw Sentiment Values

# !pwd
# print('\n')

# print(f'PATH_SENTIMENT_RAW: {PATH_SENTIMENT_RAW}\n\n')

# print('Existing Sentiment Datafiles in Destination Subdir:\n')

# !ls $PATH_SENTIMENT_RAW

# # %%
# # Verify Saving Corpus

# print(f'Saving Text_Type: {Corpus_Genre}')
# print(f'     Corpus_Type: {Corpus_Type}')

# print(f'\nThese Text Titles:\n')
# corpus_texts_dt.keys()

# print(f'\nTo This Subdirectory:\n')
# PATH_SENTIMENT_RAW

# # %%
# # Reorder and filter out cols/models to save

# syuzhetr_only_dt = {}
# cols_syuzhetr_ls = []

# for i, atext in enumerate(corpus_texts_ls):
#   print(f'\n\nText #{i}: {atext}')
#   # print(f'      {corpus_texts_dt[atext].info()}')
#   cols_syuzhetr_ls = [x for x in corpus_texts_dt[atext].columns if 'syuzhetr' in x]
#   cols_syuzhetr_ls = ['text_raw', 'text_clean'] + cols_syuzhetr_ls
#   # print(f'      {cols_syuzhetr_ls}')
#   syuzhetr_only_dt[atext] = corpus_texts_dt[atext][cols_syuzhetr_ls]
#   syuzhetr_only_dt[atext].columns

# # %%
# Corpus_Type

# # %%
# # Save sentiment values to subdir_sentiments

# if Corpus_Type == 'new':
#   save_filename = f'sentiment_raw_{Corpus_Genre}_{Corpus_Type}_corpus{Corpus_Number}_all_4syuzhetr.json'
# elif Corpus_Type == 'reference':
#   save_filename = f'sentiment_raw_{Corpus_Genre}_{Corpus_Type}_all_4syuzhetr.json'
# else:
#   print(f'ERROR: Illegal Corpus_Type: {Corpus_Type}')

# write_dict_dfs(syuzhetr_only_dt, out_file=save_filename, out_dir=f'{PATH_SENTIMENT_RAW}/')

# # %%
# # Verify json file created

# !ls -altr $PATH_SENTIMENT_RAW

# # %%
# corpus_texts_dt[corpus_texts_ls[0]].head()

# # %% [markdown]
# # ## Plot SyuzhetR 4 Models

# # %%
# #@markdown Select option to save plots:
# Save_Raw_Plots = True #@param {type:"boolean"}

# Save_Smooth_Plots = True #@param {type:"boolean"}
# Resolution_in_dpi = "300" #@param ["100", "300"]



# # %%
# # Get Col Names for all 4 SyuzhetR Models

# cols_all_ls = corpus_texts_dt[texts_titles_ls[0]].columns

# cols_syuzhetr_ls = [x for x in cols_all_ls if 'syuzhetr_' in x]
# cols_syuzhetr_ls

# # %%
# corpus_texts_dt[corpus_texts_ls[0]].iloc[0]

# # %%
# global_vars.corpus_titles_dt.keys()

# # %%
# # Save sentiment values to subdir_sentiments

# if Corpus_Type == 'new':
#   SUBDIR_GRAPHS = f'graphs_{Corpus_Genre}_{Corpus_Type}_corpus{Corpus_Number}/'
#   # save_filename = f'sentiment_raw_{Corpus_Genre}_{Corpus_Type}_corpus{Corpus_Number}_all_4syuzhetr.json'
# elif Corpus_Type == 'reference':
#   SUBDIR_GRAPHS = f'graphs_{Corpus_Genre}_{Corpus_Type}/'
#   # save_filename = f'sentiment_raw_{Corpus_Genre}_{Corpus_Type}_all_4syuzhetr.json'
# else:
#   print(f'ERROR: Illegal Corpus_Type: {Corpus_Type}')

# SUBDIR_GRAPHS = f'{global_vars.SUBDIR_GRAPHS}{SUBDIR_GRAPHS}'
# print(f'Saving to SUBDIR_GRAPHS: {SUBDIR_GRAPHS}')
# # print(f'save_filename: {save_filename}')
# # write_dict_dfs(syuzhetr_only_dt, out_file=save_filename, out_dir=f'{PATH_SENTIMENT_RAW}')

# # %%
# global_vars.SUBDIR_GRAPHS

# # %%
# !pwd

# # %%
# !ls ./graphs/graphs_novels_new_corpus5/

# # %%
# # Verify 4 SyuzhetR Models with Plots

# for i, anovel in enumerate(list(corpus_texts_dt.keys())):

#   print(f'Novel #{i}: {global_vars.corpus_titles_dt[anovel][0]}')

#   # Raw Sentiments 
#   fig = corpus_texts_dt[anovel][cols_syuzhetr_ls].plot(title=f'{global_vars.corpus_titles_dt[anovel][0]}\n SyuzhetR 4 Models: Raw Sentiments', alpha=0.3)
#   # plt.show();

#   if Save_Raw_Plots:
#     # save_filename = f'{global_vars.SUBDIR_GRAPHS}plot_{anovel}_syuzhetr_raw_dpi{Resolution_in_dpi}.png'
#     save_filename = f'{Path_to_SentimentArcs}{global_vars.SUBDIR_GRAPHS[2:]}plot_{anovel}_syuzhetr_raw_dpi{Resolution_in_dpi}.png'
#     print(f'\n\nSaving to: {save_filename}')
#     plt.savefig(save_filename, dpi=int(Resolution_in_dpi))

  
#   # Smoothed Sentiments (SMA 10%)
#   # novel_sample = 'cdickens_achristmascarol'
#   win_10per = int(corpus_texts_dt[anovel].shape[0] * 0.1)
#   _ = corpus_texts_dt[anovel][cols_syuzhetr_ls].rolling(win_10per, center=True, min_periods=0).mean().plot(title=f'{global_vars.corpus_titles_dt[anovel][0]}\n SyuzhetR 4 Models: Smoothed Sentiments (SMA 10%)', alpha=0.3)
#   # plt.show();

#   if Save_Smooth_Plots:
#     # save_filename = f'{global_vars.SUBDIR_GRAPHS}plot_{anovel}_syuzhetr_smooth10sma_dpi{Resolution_in_dpi}.png'
#     save_filename = f'{global_vars.SUBDIR_GRAPHS[2:]}plot_{anovel}_syuzhetr_smooth10sma_dpi{Resolution_in_dpi}.png'
#     print(f'\n\nSaving to: {save_filename}')
#     plt.savefig(save_filename, dpi=int(Resolution_in_dpi))


# # %% [markdown]
# # # **[STEP 5] Get Sentiments with SentimentR (8 Models)**

# # %% [markdown]
# # ## Compute New SentimentR Values
# # 
# # Call function in external get_sentimentr.R from within Python Loop
# # 
# # * https://medium.com/analytics-vidhya/calling-r-from-python-magic-of-rpy2-d8cbbf991571
# # 
# # * https://rpy2.github.io/doc/v3.0.x/html/generated_rst/pandas.html

# # %%
# %%file get_sentimentr.R

# library(sentimentr)
# library(lexicon)

# get_sentimentr_values <- function(s_v) {
  
#   print('Processing sentimentr_jockersrinker')
#   sentimentr_jockersrinker <- sentiment(s_v, polarity_dt=lexicon::hash_sentiment_jockers_rinker, 
#                                         hypen="", amplifier.weight=0.8, n.before=5, n.after=2,
#                                         adversative.weight=0.25, neutral.nonverb.like = FALSE, missing_value = 0)

#   print('Processing sentimentr_jockers')
#   sentimentr_jockers <- sentiment(s_v, polarity_dt=lexicon::hash_sentiment_jockers, 
#                                         hypen="", amplifier.weight=0.8, n.before=5, n.after=2,
#                                         adversative.weight=0.25, neutral.nonverb.like = FALSE, missing_value = 0)

#   print('Processing sentimentr_huliu')
#   sentimentr_huliu <- sentiment(s_v, polarity_dt=lexicon::hash_sentiment_huliu, 
#                                         hypen="", amplifier.weight=0.8, n.before=5, n.after=2,
#                                         adversative.weight=0.25, neutral.nonverb.like = FALSE, missing_value = 0)

#   print('Processing sentimentr_nrc')
#   sentimentr_nrc <- sentiment(s_v, polarity_dt=lexicon::hash_sentiment_nrc, 
#                                         hypen="", amplifier.weight=0.8, n.before=5, n.after=2,
#                                         adversative.weight=0.25, neutral.nonverb.like = FALSE, missing_value = 0)

#   print('Processing sentimentr_senticnet')
#   sentimentr_senticnet <- sentiment(s_v, polarity_dt=lexicon::hash_sentiment_senticnet, 
#                                         hypen="", amplifier.weight=0.8, n.before=5, n.after=2,
#                                         adversative.weight=0.25, neutral.nonverb.like = FALSE, missing_value = 0)

#   print('Processing sentimentr_sentiword')
#   sentimentr_sentiword <- sentiment(s_v, polarity_dt=lexicon::hash_sentiment_sentiword, 
#                                         hypen="", amplifier.weight=0.8, n.before=5, n.after=2,
#                                         adversative.weight=0.25, neutral.nonverb.like = FALSE, missing_value = 0)

#   print('Processing sentimentr_loughran_mcdonald')
#   sentimentr_loughran_mcdonald <- sentiment(s_v, polarity_dt=lexicon::hash_sentiment_loughran_mcdonald, 
#                                         hypen="", amplifier.weight=0.8, n.before=5, n.after=2,
#                                         adversative.weight=0.25, neutral.nonverb.like = FALSE, missing_value = 0)

#   print('Processing sentimentr_socal_google')
#   sentimentr_socal_google <- sentiment(s_v, polarity_dt=lexicon::hash_sentiment_socal_google, 
#                                         hypen="", amplifier.weight=0.8, n.before=5, n.after=2,
#                                         adversative.weight=0.25, neutral.nonverb.like = FALSE, missing_value = 0)

#   anovel_sentimentr_df <- data.frame('text_clean' = s_v,
#                                 'sentimentr_jockersrinker' = sentimentr_jockersrinker$sentiment,
#                                 'sentimentr_jockers' = sentimentr_jockers$sentiment,
#                                 'sentimentr_huliu' = sentimentr_huliu$sentiment,
#                                 'sentimentr_nrc' = sentimentr_nrc$sentiment,
#                                 'sentimentr_senticnet' = sentimentr_senticnet$sentiment,
#                                 'sentimentr_sentiword' = sentimentr_sentiword$sentiment,
#                                 'sentimentr_loughran_mcdonald' = sentimentr_loughran_mcdonald$sentiment,
#                                 'sentimentr_socal_google' = sentimentr_socal_google$sentiment
#                                 )
#   return(anovel_sentimentr_df)

# }

# # %%
# # Verify the *.R file above was written correctly

# !cat get_sentimentr.R

# # %%
# # Setup python robject with external library::function()
# # https://rpy2.github.io/doc/v3.0.x/html/generated_rst/pandas.html

# # import rpy2.robjects as robjects

# # Defining the R script and loading the instance in Python
# # from rpy2.robjects import pandas2ri 
# r = robjects.r

# # Loading the function we have defined in R.
# r['source']('get_sentimentr.R')

# # Reading and processing data
# get_sentimentr_function_r = robjects.globalenv['get_sentimentr_values']

# # %%
# corpus_texts_dt.keys()

# # %%
# # Test

# # Convert Python List of Strings to a R vector of characters
# # test_ls = corpus_texts_dt[next(iter(corpus_texts_dt))]['text_clean'].to_list()
# test_ls = corpus_texts_dt[corpus_texts_ls[0]]['text_clean'].to_list()
# s_v = robjects.StrVector(test_ls)
# type(s_v)

# get_sentimentr_function_r(s_v)

# # %%
# text_clean_ct = corpus_texts_dt[corpus_texts_ls[0]].text_clean.isna().sum()
# text_clean_ct
# # len(text_clean_ls.isnull())

# # %% [markdown]
# # **[RE-EXECUTE] May have to re-execute following code cell several times**

# # %%
# %whos dict

# # %%
# corpus_texts_dt.keys()

# # %%
# %%time

# # NOTE:  8m19s 13 Novels 
# #       16m39s 19 Novels
# #      -----------------
# #       24m58s 32 Novels
# #        5m00s  @19:44 on 20220227 Colab Pro (2 Novels)

# #        3m18s 21:24 on 20220415 Colab Pro (3 Novels, 628k, 662k, 897k)
# #        3m09s 21:45 on 20220415 Colab Pro (3 Novels, 628k, 662k, 897k)

# #        3m17s 08:17 on 20220416 Colab Pro (3 Novels, 628k, 662k, 897k)


# # Call external get_sentimentr::get_sentimentr_values with Python loop over all novels

# # novels_sentimentr_dt = {}

# # Reset corpus_texts_dt
# # corpus_texts_dt = {}

# # TODO: Norm var name to atext_df <- anovel_df
# anovel_df = pd.DataFrame()

# novels_titles_ls = list(corpus_texts_dt.keys())
# novels_titles_ls.sort()
# # for i, anovel in enumerate(novels_titles_ls[:19]):
# for i, anovel in enumerate(novels_titles_ls):  
#   print(f'\nProcessing Novel #{i}: {anovel}')
  
#   # Delete contents of anovel_df DataFrame
#   anovel_df = anovel_df[0:0]

#   print(f'     {corpus_texts_dt[anovel].shape}')
#   # Get text_clean as list of strings
#   text_clean_ls = corpus_texts_dt[anovel]['text_clean'].to_list()

#   # Convert Python List of Strings to a R vector of characters
#   # https://rpy2.github.io/doc/v3.0.x/html/generated_rst/pandas.html
#   s_v = robjects.StrVector(text_clean_ls)
#   anovel_df_r = get_sentimentr_function_r(s_v)

#   # Convert rpy2.robjects.vectors.DataFrame to pandas.core.frame.DataFrame
#   # https://stackoverflow.com/questions/20630121/pandas-how-to-convert-r-dataframe-back-to-pandas 
#   print(f'type(anovel_df_r): {type(anovel_df_r)}')
#   anovel_df = pd.DataFrame.from_dict({ key : np.asarray(anovel_df_r.rx2(key)) for key in anovel_df_r.names })
#   print(f'type(anovel_df): {type(anovel_df)}')

#   # Save Results
#   # novels_dt[anovel] = anovel_df.copy(deep=True)

#   # This works for Novels New Corpus Texts
#   corpus_texts_dt[anovel]['sentimentr_jockersrinker'] = anovel_df['sentimentr_jockersrinker']
#   corpus_texts_dt[anovel]['sentimentr_jockers'] = anovel_df['sentimentr_jockers']
#   corpus_texts_dt[anovel]['sentimentr_huliu'] = anovel_df['sentimentr_huliu']
#   corpus_texts_dt[anovel]['sentimentr_nrc'] = anovel_df['sentimentr_nrc']
#   corpus_texts_dt[anovel]['sentimentr_senticnet'] = anovel_df['sentimentr_senticnet']
#   corpus_texts_dt[anovel]['sentimentr_sentiword'] = anovel_df['sentimentr_sentiword']
#   corpus_texts_dt[anovel]['sentimentr_loughran_mcdonald'] = anovel_df['sentimentr_loughran_mcdonald']
#   corpus_texts_dt[anovel]['sentimentr_socal_google'] = anovel_df['sentimentr_socal_google'] 


# """
#   # This works for Novels Reference Corpus Texts
#   corpus_texts_dt[anovel]['sentimentr_jockersrinker'] = anovel_df[anovel]['sentimentr_jockersrinker']
#   corpus_texts_dt[anovel]['sentimentr_jockers'] = anovel_df[anovel]['sentimentr_jockers']
#   corpus_texts_dt[anovel]['sentimentr_huliu'] = anovel_df[anovel]['sentimentr_huliu']
#   corpus_texts_dt[anovel]['sentimentr_nrc'] = anovel_df[anovel]['sentimentr_nrc']
#   corpus_texts_dt[anovel]['sentimentr_senticnet'] = anovel_df[anovel]['sentimentr_senticnet']
#   corpus_texts_dt[anovel]['sentimentr_sentiword'] = anovel_df[anovel]['sentimentr_sentiword']
#   corpus_texts_dt[anovel]['sentimentr_loughran_mcdonald'] = anovel_df[anovel]['sentimentr_loughran_mcdonald']
#   corpus_texts_dt[anovel]['sentimentr_socal_google'] = anovel_df[anovel]['sentimentr_socal_google'] 
# """ 

# # %%
# anovel_df.head()

# # %%
# corpus_texts_dt[texts_titles_ls[0]].head()

# # %%
# cols_sentimentr_ls = [x for x in corpus_texts_dt[corpus_texts_ls[0]].columns if 'sentimentr_' in x]
# cols_sentimentr_ls

# # %%
# # Verify DataFrame shape of first Text in Corpus

# corpus_texts_dt[corpus_texts_ls[0]].shape

# # %% [markdown]
# # ## Checkpoint: Save SentimentR Values

# # %%
# # Verify in SentimentArcs Root Directory
# #   and destination Subdir for Raw Sentiment Values

# !pwd
# print('\n')

# print(f'PATH_SENTIMENT_RAW: {PATH_SENTIMENT_RAW}\n\n')

# print('Existing Sentiment Datafiles in Destination Subdir:\n')

# !ls $PATH_SENTIMENT_RAW

# # %%
# # Verify Saving Corpus

# print(f'Saving Text_Type: {Corpus_Genre}')
# print(f'     Corpus_Type: {Corpus_Type}')

# print(f'\nThese Text Titles:\n')
# corpus_texts_dt.keys()

# # %%
# # Reorder and filter out cols/models to save

# sentimentr_only_dt = {}
# cols_sentimentr_ls = []

# for i, atext in enumerate(corpus_texts_ls):
#   print(f'\n\nModel #{i}: {atext}')
#   # print(f'      {corpus_texts_dt[atext].info()}')
#   cols_sentimentr_ls = [x for x in corpus_texts_dt[atext].columns if 'sentimentr' in x]
#   cols_sentimentr_ls = ['text_raw', 'text_clean'] + cols_sentimentr_ls
#   # print(f'      {cols_syuzhetr_ls}')
#   sentimentr_only_dt[atext] = corpus_texts_dt[atext][cols_sentimentr_ls]
#   sentimentr_only_dt[atext].columns

# # %%
# # Save sentiment values to subdir_sentiments

# if Corpus_Type == 'new':
#   save_filename = f'sentiment_raw_{Corpus_Genre}_{Corpus_Type}_corpus{Corpus_Number}_all_8sentimentr.json'
# elif Corpus_Type == 'reference':
#   save_filename = f'sentiment_raw_{Corpus_Genre}_{Corpus_Type}_all_8sentimentr.json'
# else:
#   print(f'ERROR: Illegal Corpus_Type: {Corpus_Type}')

# write_dict_dfs(sentimentr_only_dt, out_file=save_filename, out_dir=f'{PATH_SENTIMENT_RAW}/')

# # %%
# # Save sentiment values to subdir_sentiments
# """
# save_filename = f'all_{Corpus_Genre}_{Corpus_Type}_8sentimentr.json'

# write_dict_dfs(sentimentr_only_dt, out_file=save_filename, out_dir=f'{PATH_SENTIMENT_RAW}')
# """;

# # %%
# # Verify Dictionary was saved correctly by reading back the *.json datafile

# test_dt = read_dict_dfs(in_file=save_filename, in_dir=f'{PATH_SENTIMENT_RAW}/')
# test_dt.keys()

# # %% [markdown]
# # ## Plot SentimentR 8 Models

# # %%
# #@markdown Select option to save plots:
# Save_Raw_Plots = True #@param {type:"boolean"}

# Save_Smooth_Plots = True #@param {type:"boolean"}
# Resolution_in_dpi = "300" #@param ["100", "300"]



# # %%
# # Get Col Names for all SentimentR Models
# cols_all_ls = corpus_texts_dt[corpus_texts_ls[0]].columns
# cols_sentimentr_ls = [x for x in cols_all_ls if 'sentimentr_' in x]
# cols_sentimentr_ls

# # %%
# corpus_texts_dt[corpus_texts_ls[0]]

# # %%
# # Save sentiment values to subdir_sentiments

# if Corpus_Type == 'new':
#   SUBDIR_GRAPHS = f'graphs_{Corpus_Genre}_{Corpus_Type}_corpus{Corpus_Number}/'
#   # save_filename = f'sentiment_raw_{Corpus_Genre}_{Corpus_Type}_corpus{Corpus_Number}_all_4syuzhetr.json'
# elif Corpus_Type == 'reference':
#   SUBDIR_GRAPHS = f'graphs_{Corpus_Genre}_{Corpus_Type}/'
#   # save_filename = f'sentiment_raw_{Corpus_Genre}_{Corpus_Type}_all_4syuzhetr.json'
# else:
#   print(f'ERROR: Illegal Corpus_Type: {Corpus_Type}')

# SUBDIR_GRAPHS = f'{global_vars.SUBDIR_GRAPHS}{SUBDIR_GRAPHS}'
# print(f'Saving to SUBDIR_GRAPHS: {SUBDIR_GRAPHS}')
# # print(f'save_filename: {save_filename}')
# # write_dict_dfs(syuzhetr_only_dt, out_file=save_filename, out_dir=f'{PATH_SENTIMENT_RAW}')

# # %%
# # Verify 8 SentimentR Models with Plots

# for i, anovel in enumerate(list(corpus_texts_dt.keys())):

#   print(f'Novel #{i}: {global_vars.corpus_titles_dt[anovel][0]}')

#   # Raw Sentiments 
#   fig = corpus_texts_dt[anovel][cols_sentimentr_ls].plot(title=f'{global_vars.corpus_titles_dt[anovel][0]}\n SentimentR 8 Models: Raw Sentiments', alpha=0.3)
#   # plt.show();

#   if Save_Raw_Plots:
#     # save_filename = f'{global_vars.SUBDIR_GRAPHS}plot_sentimentr_raw_{anovel}_dpi{Resolution_in_dpi}.png'
#     save_filename = f'{Path_to_SentimentArcs}{global_vars.SUBDIR_GRAPHS[2:]}plot_{anovel}_sentimentr_raw_dpi{Resolution_in_dpi}.png'
#     print(f'\n\nSaving to: {save_filename}')
#     plt.savefig(save_filename, dpi=int(Resolution_in_dpi))

  
#   # Smoothed Sentiments (SMA 10%)
#   # novel_sample = 'cdickens_achristmascarol'
#   win_10per = int(corpus_texts_dt[anovel].shape[0] * 0.1)
#   corpus_texts_dt[anovel][cols_sentimentr_ls].rolling(win_10per, center=True, min_periods=0).mean().plot(title=f'{global_vars.corpus_titles_dt[anovel][0]}\n SentimentR 7 Models: Smoothed Sentiments (SMA 10%)', alpha=0.3)
#   # plt.show();

#   if Save_Smooth_Plots:
#     # save_filename = f'{global_vars.SUBDIR_GRAPHS}plot_sentimentr_smooth10sma_{anovel}_dpi{Resolution_in_dpi}.png'
#     save_filename = f'{Path_to_SentimentArcs}{global_vars.SUBDIR_GRAPHS[2:]}plot_{anovel}_sentimentr_smooth_dpi{Resolution_in_dpi}.png'
#     print(f'\n\nSaving to: {save_filename}')
#     plt.savefig(save_filename, dpi=int(Resolution_in_dpi))


# # %%
# corpus_texts_ls

# # %%
# text_indx = 0

# corpus_texts_dt[corpus_texts_ls[text_indx]].head()

# # %%
# # Retrieve a range of Lines from the Text

# sentence_start = 50
# sentence_end = 60

# ' '.join(list(corpus_texts_dt[corpus_texts_ls[text_indx]].iloc[sentence_start:sentence_end]['text_raw']))

# # %% [markdown]
# # # **END OF NOTEBOOK**

# # %% [markdown]
# # ---


