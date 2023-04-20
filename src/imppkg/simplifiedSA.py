# Standard library imports
# import string
import re
# import datetime
# import os
import configparser

import numpy as np
import modin.pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

# from cleantext import clean
# import contractions

# from sklearn.preprocessing import MinMaxScaler, StandardScaler

# # for gutenberg import
import requests

# for text cleaning
import nltk # TODO: verify okay to keep this dependency. Dependencies: 12 (only 1 restricted by version); Dependent packages: 1.78K.
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# # for peak detection
# from scipy.signal import find_peaks
# import warnings # don't know what this is for
# warnings.filterwarnings('ignore') # don't know what this is for

# Read the toml file that contains settings that advanced users can edit (via ???)
config = configparser.ConfigParser()
config.read('SA_settings.toml')

# Setup matplotlib
plt.rcParams["figure.figsize"] = (20,10) 

# Global Vars — if these are only used once in the final code, move each to the function where it's used
# novel_filename_str = ''
# novel_title_str = ''
# novel_raw_str = ''
# novel_clean_str = ''
# novel_lines_ls = []
# novel_sentences_ls = []
# novel_paragraphs_ls = []
TEXT_ENCODING = config.get('imports', 'text_encoding')
PARA_SEP = config.get('imports', 'paragraph_separation')

# Main (Modin — uses multiple cores for operations on pandas dfs) DataFrame for Novel Sentiments
sentiment_df = pd.DataFrame


# Custom Exceptions
class InputFormatException(Exception):
    pass

## COMMON FUNCTIONS ##

def test_func():
    print("test_func() ran")


def save_text2txt_and_download(text_obj, file_suffix='_save.txt'):
  '''
  INPUT: text object and suffix to add to output text filename
  OUTPUT: Write text object to text file (both temp VM and download)
  '''
  # "text object" = string or list of strings

  if type(text_obj) == str:
    print('STEP 1. Processing String Object\n')
    str_obj = text_obj
  elif type(text_obj) == list:
    if (len(text_obj) > 0):
      if type(text_obj[0]) == str:
        print('STEP 1. Processing List of Strings Object\n')
        str_obj = "\n".join(text_obj)
      else:
        print('ERROR: Object is not a List of Strings [save_text2txt_and_download()]')
        return -1
    else:
      print('ERROR: Object is an empty List [save_text2txt_and_download()]')
      return -1
  else:
    print('ERROR: Object Type is neither String nor List [save_text2txt_and_download()]')
    return -1

  datetime_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
  # out_filename = novel_name_str.split('.')[0] + '_' + datetime_str + file_suffix
  out_filename = novel_name_str.split('.')[0] + file_suffix

  # Write file to temporary VM filesystem
  print(f'STEP 2. Saving textfile to temporary VM file: {out_filename}\n')
  with open(out_filename, "w") as fp:
    fp.write(str_obj)

  # Download permanent copy of file
  print(f'STEP 3. Downloading permanent copy of textfile: {out_filename}\n')
  files.download(out_filename)


def save_df2csv_and_download(df_obj, novel_name_str, file_suffix='_save.csv', nodate=True):
  '''
  INPUT: DataFrame object and suffix to add to output csv filename
  OUTPUT: Write DataFrame object to csv file (both temp VM and download)
  '''

  if isinstance(df_obj, pd.DataFrame):
    datetime_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if nodate:
      out_filename = novel_name_str.split('.')[0] + file_suffix
    else:
      out_filename = novel_name_str.split('.')[0] + '_' + datetime_str + file_suffix
    # print(f'STEP 1. Saving DataFrame: {df_obj.__name__} to temporary VM file: {out_filename}\n') # Also, isinstance(obj, pd.DataFrame)
    print(f'STEP 1. Saving DataFrame to temporary VM file: {out_filename}\n')
    df_obj.to_csv(out_filename, index=False) 
  else:
    print(f'ERROR: Object is not a DataFrame [save_df2csv_and_download()]')
    return -1

  # Download permanent copy of file
  print(f'STEP 2. Downloading permanent copy of csvfile: {out_filename}\n')
  files.download(out_filename)


def expand_contractions(input_str):
  '''
  INPUT: long string
  OUTPUT: long string with expanded contractions
  '''

  output_str = contractions.fix(input_str)

  return output_str


## IPYNB SECTIONS AS FUNCTIONS ##

def uploadText(uploaded : str, novel_title : str):
    '''
    Parameter(s):
    uploaded: dictionary with filename as key and properly formatted text body as value (should have just one key-value pair)
    novel_title_str: [Title] by [Author] (string)
    '''
    # TODO: change type of 'uploaded' to be whatever Dev wants
    # Possible addition: Ability to process multiple input files

    novel_filename_str = list(uploaded.keys())[0]
    filename_ext_str = novel_filename_str.split('.')[-1]
    if filename_ext_str == 'txt':
        # Extract from Dict and decode binary into char string
        novel_raw_str = uploaded[novel_filename_str].decode(TEXT_ENCODING)
    else:
        raise InputFormatException("Must provide path to a plain text file (*.txt)")

    print( f'Novel Filename:\n\n  {novel_filename_str}\n\n\n' +
            f'Novel Title: {novel_title}\n' +
            f'  Char Len: {len(novel_raw_str)}\n' +
            '====================================\n\n' +
            f'Beginning:\n\n {novel_raw_str[:500]}\n\n\n' +
            '\n------------------------------------\n' +
            f'Ending:\n\n {novel_raw_str[-500:]}\n\n\n')

    return novel_raw_str # return as single-item dict with novel_title as key instead? or a custom "SAtext" object with data members title, body, segmented_body, clean_body?

def peepUpload(novel_raw_str): # would make more sense as a method imo. could take in an SAtext object and be a method, or take in a dict to be able to print the file name
    # Return string showing beginning and end of text for user verification
    # f'Novel Filename:\n\n  {novel_filename_str}\n\n\n' +
        #     f'Novel Title: {novel_title}\n' +
    stringToPeep =     (f'  Char Len: {len(novel_raw_str)}\n' +
            '====================================\n\n' +
            f'Beginning:\n\n {novel_raw_str[:500]}\n\n\n' +
            '\n------------------------------------\n' +
            f'Ending:\n\n {novel_raw_str[-500:]}\n\n\n')
    print(stringToPeep)
    return(stringToPeep)


def gutenbergImport(Novel_Title : str, Gutenberg_URL : str, 
                    sentence_first_str = None, sentence_last_str = None):
    #@title Enter the URL of your novel at ***gutenberg.net.au***
    #@markdown Paste the URL to the ***HTML version*** (not plain text).

    # Novel_Title = 'Frankenstein by Mary Shelley'  #@param {type: "string"}

    # Gutenberg_URL = 'https://gutenberg.net.au/ebooks/z00006.html'  #@param {type: "string"}
    # the first sentence in the body of your novel: sentence_first_str
    # the last sentence in the body of your novel: sentence_last_str


    # Get raw HTML of novel from Gutenberg.net.au
    response=requests.get(Gutenberg_URL)  # TODO: Pass the URL to the .get() method of the requests object
    html = response.text

    # Use HTML <p> tags to extract text into list of paragraphs
    paragraphs = re.findall(r'<p>(.*?)</p>', html, flags=re.DOTALL)

    if (len(paragraphs) < 3):
        InputFormatException("Fewer than three paragraphs detected")

    # TODO: figure out what this is doing and why (seems like it's just undoing what we did, plus the \r\n replacement)
    # Concatenate all paragraphs into a single novel string
    # For every paragraph, replace each hardcoded \r\n with a single space
    paragraphs_flat = [re.sub(r'\r\n', ' ', paragraph) for paragraph in paragraphs]
    # Concatenate all paragraphs into single strings separated by two \n
    novel_raw_str = '\n\n'.join(paragraphs_flat)
    
    if (sentence_first_str is not None & sentence_last_str is not None): # using optional function args
        # Remove header
        novel_raw_str = ' '.join(novel_raw_str.partition(sentence_first_str)[1:])
        # Remove footer
        novel_raw_str = ' '.join(novel_raw_str.partition(sentence_last_str)[:2])

    # Print string for user verification
    # TODO: talk to Dev about the best way to return this. 
    #   novel_raw_str could alter an argument instead of being a return value?
    #   or i can write a separate function verifyImport(novel_raw_str) that returns the stuff printed below
    print('\nSTART OF NOVEL: -----\n'+
          novel_raw_str[:1000] + '\n\n'+
          '\nEND OF NOVEL: -----\n\n'+
          novel_raw_str[-1000:])
    
    return(novel_raw_str)


def segmentText(novel_raw_str :  str):
    # Segment by sentence
    novel_sentences_ls = sent_tokenize(novel_raw_str) # using nltk.tokenize

    # Most of the rest of this function (not the delete empty sentences part) is just returning things for user verification
    sentence_count = len(novel_sentences_ls)
    num_senteces_to_show = 5

    verificationString = f'\nFirst {num_senteces_to_show} Sentences: -----\n\n'
    for i, asent in enumerate(novel_sentences_ls[:num_senteces_to_show]):
        verificationString += f'Sentences #{i}: {asent}\n'

    print(f'\nLast {num_senteces_to_show} Sentences: -----\n')
    for i, asent in enumerate(novel_sentences_ls[-num_senteces_to_show:]):
        verificationString += f'Sentences #{sentence_count - (num_senteces_to_show - i)}: {asent}\n'

    verificationString += f'\n\nThere are {sentence_count} Sentences in the novel\n'

    # Delete the empty Sentences and those without any alphabetic characters
    novel_sentences_ls = [x.strip() for x in novel_sentences_ls if len(x.strip()) > 0]
    novel_sentences_ls = [x.strip() for x in novel_sentences_ls if re.search('[a-zA-Z]', x)]
    
    num_sentences_removed = sentence_count - len(novel_sentences_ls)
    if (num_sentences_removed!=0):
        verificationString += f'\n\n{num_sentences_removed} empty and/or non-alphabetic sentences removed\n'
    # Q: How does sentence number & returning sentences around crux points still match up after doing this? Or do we not care exactly where the crux is in the original text?

    # Plot distribution of sentence lengths
    # _ = plt.hist([len(x) for x in novel_sentences_ls], bins=100)

    print(verificationString) # same deal as before: have a separate verification function that returns this? return this in a list along with the actual return value? just print it?
    return novel_sentences_ls


def clean_str(dirty_str): # to be called within clean_text
  # TODO: all of this
  '''
  INPUT: a raw string
  OUTPUT: a clean string
  '''

  contraction_expanded_str = contractions.fix(dirty_str)

  clean_str = clean(contraction_expanded_str,4,
      fix_unicode=True,               # fix various unicode errors
      to_ascii=True,                  # transliterate to closest ASCII representation
      lower=True,                     # lowercase text
      no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
      no_urls=False,                  # replace all URLs with a special token
      no_emails=False,                # replace all email addresses with a special token
      no_phone_numbers=False,         # replace all phone numbers with a special token
      no_numbers=False,               # replace all numbers with a special token
      no_digits=False,                # replace all digits with a special token
      no_currency_symbols=False,      # replace all currency symbols with a special token
      no_punct=False,                 # remove punctuations
      # replace_with_punct="",          # instead of removing punctuations you may replace them
      # replace_with_url="<URL>",
      # replace_with_email="<EMAIL>",
      # replace_with_phone_number="<PHONE>",
      # replace_with_number="<NUMBER>",
      # replace_with_digit="0",
      # replace_with_currency_symbol="<CUR>",
      lang="en"                       # set to 'de' for German special handling
  )

  # Replace all new lines/returns with single whitespace
  clean_str = ' '.join(clean_str.split())
  # clean_str = clean_str.replace('\n\r', ' ')
  # clean_str = clean_str.replace('\n', ' ')
  # clean_str = clean_str.replace('\r', ' ')
  # clean_str = ' '.join(clean_str.split())
  return clean_str 


def clean_text(): # TODO: change name of function
    # Create sentiment_df to hold text sentences and corresponding sentiment values

    sentiment_df = pd.DataFrame({'text_raw': novel_sentences_ls})
    sentiment_df['text_raw'] = sentiment_df['text_raw'].astype('string')
    sentiment_df['text_raw'] = sentiment_df['text_raw'].str.strip()

    # clean the 'text_raw' column and create the 'text_clean' column
    # novel_df['text_clean'] = hero.clean(novel_df['text_raw'])
    sentiment_df['text_clean'] = sentiment_df['text_raw'].apply(lambda x: clean_str(x)) # call clean_str()
    sentiment_df['text_clean'] = sentiment_df['text_clean'].astype('string')
    sentiment_df['text_clean'] = sentiment_df['text_clean'].str.strip()
    sentiment_df['text_raw_len'] = sentiment_df['text_raw'].apply(lambda x: len(x))
    sentiment_df['text_clean_len'] = sentiment_df['text_clean'].apply(lambda x: len(x))

    sentiment_df.head()
    sentiment_df.info()

    # Check for any empty text_clean strings
    sentiment_df[sentiment_df['text_clean_len'] == 0]['text_clean']

    # Drop Sentence if Raw length < 1 (Double check)
    sentiment_df = sentiment_df[sentiment_df['text_raw_len'] > 0]
    sentiment_df.shape

    # Fill any empty text_clean with a neutral word
    neutral_word = 'NEUTRALWORD'
    sentiment_df[sentiment_df['text_clean_len'] == 0]['text_clean'] = neutral_word
    sentiment_df[sentiment_df['text_clean_len'] == 0]['text_clean_len'] = 11
    sentiment_df['text_clean_len'].sort_values(ascending=True) # , key=lambda x: len(x), inplace=True)
    # sentiment_df.text_clean.fillna(value='', inplace=True)

    # Add Line Numbers
    sentence_no_ls = list(range(sentiment_df.shape[0]))
    sentiment_df.insert(0, 'line_no', sentence_no_ls)

    # View the shortest lines by text_raw_len
    sentiment_df.sort_values(by=['text_raw_len']).head(20)  

    # Save segmented and cleaned text to file
    novel_camel_str = ''.join([re.sub(r'[^\w\s]','',x).capitalize() for x in novel_title_str.split()])
    save_df2csv_and_download(sentiment_df, novel_camel_str, '_cleaned.csv', nodate=True)


def vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_vader_ls = [sid_obj.polarity_scores(asentence)['compound'] for asentence in sentiment_df['text_clean'].to_list()]
    
    # Create new VADER DataFrame to save results
    vader_df = sentiment_df[['line_no', 'text_raw', 'text_clean']].copy(deep=True)
    vader_df['sentiment'] = pd.Series(sentiment_vader_ls) 
    vader_df.head()

    win_per = 0.1
    win_size = int(win_per * vader_df.shape[0])
    _ = vader_df['sentiment'].rolling(win_size, center=True).mean().plot(grid=True)

    # Save Model Sentiment Time Series
    novel_camel_str = ''.join([re.sub(r'[^\w\s]','',x).capitalize() for x in novel_title_str.split()])
    save_df2csv_and_download(vader_df, novel_camel_str, '_vader.csv', nodate=True)

def textblob():
    from textblob import TextBlob
    testimonial = TextBlob("Textblob is amazingly simple to use. What great fun!")
    print(testimonial.sentiment.polarity)
    sentiment_textblob_ls = [TextBlob(asentence).sentiment.polarity for asentence in sentiment_df['text_clean'].to_list()]
    # sentiment_df['textblob'] = sentiment_df['text_clean'].apply(lambda x : TextBlob(x).sentiment.polarity)
    # Create new TextBlob DataFrame to save results
    textblob_df = sentiment_df[['line_no', 'text_raw', 'text_clean']].copy(deep=True)
    textblob_df['sentiment'] = pd.Series(sentiment_textblob_ls) 
    textblob_df.head()

    win_per = 0.1
    win_size = int(win_per * textblob_df.shape[0])
    _ = textblob_df['sentiment'].rolling(win_size, center=True).mean().plot(grid=True)

    # Save Model Sentiment Time Series
    novel_camel_str = ''.join([re.sub(r'[^\w\s]','',x).capitalize() for x in novel_title_str.split()])
    save_df2csv_and_download(textblob_df, novel_camel_str, '_textblob.csv', nodate=True)


def preptransformer():
    from transformers import pipeline
    from transformers import AutoTokenizer, AutoModelWithLMHead  # T5Base 50k
    from transformers import AutoModelForSequenceClassification, Trainer
    from transformers import AutoModelForSeq2SeqLM, AutoModelWithLMHead
    from transformers import BertTokenizer, BertForSequenceClassification
    # import sentencepiece

    # Create class for data preparation
    class SimpleDataset:
        def __init__(self, tokenized_texts):
            self.tokenized_texts = tokenized_texts
        
        def __len__(self):
            return len(self.tokenized_texts["input_ids"])
        
        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.tokenized_texts.items()}


def distilbert():
    # Load tokenizer and model, create trainer

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    trainer = Trainer(model=model)

    # (there was a test here i don't think was necessary to run, so it's just in the test file)

    # Compute Sentiment Time Series
    line_ls = sentiment_df['text_clean'].to_list()

    # Tokenize texts and create prediction data set
    tokenized_texts = tokenizer(line_ls,truncation=True,padding=True)
    pred_dataset = SimpleDataset(tokenized_texts)

    # Run predictions
    predictions = trainer.predict(pred_dataset)

    # Transform predictions to labels
    sentiment_ls = predictions.predictions.argmax(-1)
    labels_ls = pd.Series(sentiment_ls).map(model.config.id2label)
    scores_ls = (np.exp(predictions[0])/np.exp(predictions[0]).sum(-1,keepdims=True)).max(1)

    # Create DataFrame with texts, predictions, labels, and scores
    line_no_ls = list(range(len(sentiment_ls)))
    distilbert_df = pd.DataFrame(list(zip(line_no_ls, line_ls,sentiment_ls,labels_ls,scores_ls)), columns=['line_no','line','sentiment','label','score'])
    distilbert_df.head()

    # Ensure balance of sentiments
    # distilbert_df['distilbert'].unique()
    _ = distilbert_df['label'].hist()

    # Plot
    win_per = 0.1
    win_size = int(win_per * distilbert_df.shape[0])
    _ = distilbert_df['sentiment'].rolling(win_size, center=True).mean().plot(grid=True)

    # Save segmented and cleaned text to file
    novel_camel_str = ''.join([re.sub(r'[^\w\s]','',x).capitalize() for x in novel_title_str.split()])
    save_df2csv_and_download(distilbert_df, novel_camel_str, '_distilbert.csv', nodate=True)


def combineModels():
    # Merge all dataframes
    sentiment_all_df = sentiment_df[['line_no','text_raw','text_clean']].copy(deep=True)

    print('Try to merge VADER...')
    try:
        sentiment_all_df['vader'] = vader_df['sentiment']
        print('  Success\n')
    except:
        print('  FAILED\n')

    print('Try to merge TextBlob...')
    try:
        sentiment_all_df['textblob'] = textblob_df['sentiment']
        print('  Success\n')
    except:
        print('  FAILED\n')

    print('Try to merge DistilBERT...')
    try:
        sentiment_all_df['distilbert'] = distilbert_df['sentiment']
        print('  Success\n')
    except:
        print('  FAILED\n')

    print('Try to merge NLPTown...')
    try:
        sentiment_all_df['nlptown'] = nlptown_df['sentiment']
        print('  Success\n')
    except:
        print('  FAILED\n')

    print('Try to merge RoBERTa15lg...')
    try:
        sentiment_all_df['roberta15lg'] = roberta15lg_df['sentiment']
        print('  Success\n')
    except:
        print('  FAILED\n')

    # Verify
    sentiment_all_df.head()
    sentiment_all_df.info()

    # Save Sentiment Timeseries to Datafile
    novel_camel_str = ''.join([re.sub(r'[^\w\s]','',x).capitalize() for x in novel_title_str.split()])
    save_df2csv_and_download(sentiment_all_df, novel_camel_str, '_merged.csv', nodate=True)


def visualize():
    ##@title Enter the Sliding Window width as Percent of Novel length (default 10%, larger=smoother)
    Window_Percent = 10 #@param {type:"slider", min:1, max:20, step:1}
    win_per = Window_Percent
    win_size = int(win_per/100 * sentiment_all_df.shape[0])


    # Plot Raw Timeseries
    ax = sentiment_all_df[model_samelen_ls].rolling(win_size, center=True).mean().plot(grid=True, lw=3, colormap='Dark2')
    ax.title.set_text(f'Sentiment Analysis \n {novel_title_str} \n Raw Sentiment Timeseries')
    plt.show()


    # Plot Standard Scalar Normalized
    # Compute the mean of each raw Sentiment Timeseries and adjust to [-1.0, 1.0] Range
    model_samelen_adj_mean_dt = {}

    for amodel in model_samelen_ls:
        amodel_min = sentiment_all_df[amodel].min()
        amodel_max = sentiment_all_df[amodel].max()
        amodel_range = amodel_max - amodel_min
        amodel_raw_mean = sentiment_all_df[amodel].mean()
    
    if amodel_range > 2.0:
        model_samelen_adj_mean_dt[amodel] = (amodel_raw_mean + amodel_min)/(amodel_max - amodel_min)*2 + -1.0
    elif amodel_range < 1.1:
        model_samelen_adj_mean_dt[amodel] = (amodel_raw_mean + amodel_min)/(amodel_max - amodel_min)*2 + -1.0
    else:
        model_samelen_adj_mean_dt[amodel] = amodel_raw_mean
    
    print(f'Model: {amodel}\n  Raw Mean: {amodel_raw_mean}\n  Adj Mean: {model_samelen_adj_mean_dt[amodel]}\n  Min: {amodel_min}\n  Max: {amodel_max}\n  Range: {amodel_range}\n')

    # Normalize Timeseries with StandardScaler (u=0, sd=+/- 1)

    # sentiment_all_norm_df = pd.DataFrame()
    sentiment_all_norm_df = sentiment_all_df[['line_no','text_raw','text_clean']].copy(deep=True)
    sentiment_all_norm_df[model_samelen_ls] = StandardScaler().fit_transform(sentiment_all_df[model_samelen_ls])
    sentiment_all_norm_df.head()

    # Plot Normalized Timeseries to same mean

    ax = sentiment_all_norm_df[model_samelen_ls].rolling(win_size, center=True).mean().plot(grid=True, colormap='Dark2', lw=3)
    ax.title.set_text(f'Sentiment Analysis \n {novel_title_str} \n Normalization: Standard Scaler')

    plt.show()

    # Save segmented and cleaned text to file
    novel_camel_str = ''.join([re.sub(r'[^\w\s]','',x).capitalize() for x in novel_title_str.split()])
    save_df2csv_and_download(sentiment_all_norm_df, novel_camel_str, '_allstdscaler.csv', nodate=True)


     # Plot StandardScaler + Original Mean
    # Plot Normalized Timeseries to adjusted original mean
    sentiment_all_adjnorm_df = sentiment_all_df[['line_no','text_raw','text_clean']].copy(deep=True)
    for amodel in model_samelen_ls:
        sentiment_all_adjnorm_df[amodel] = sentiment_all_norm_df[amodel] + model_samelen_adj_mean_dt[amodel]

    ax = sentiment_all_norm_df[model_samelen_ls].rolling(win_size, center=True).mean().plot(grid=True, alpha=0.3, colormap='Dark2')
    ax.hlines(y=0.0, xmin=0, xmax=sentiment_all_norm_df.shape[0], linewidth=3, linestyles='--', color='r', alpha=0.5)
    _ = sentiment_all_adjnorm_df[model_samelen_ls].rolling(win_size, center=True).mean().plot(ax=ax, grid=True, colormap='Dark2', lw=3)
    ax.title.set_text(f'Sentiment Analysis \n {novel_title_str} \n Normalization: Standard Scaler + True Mean Adjustment')
    plt.show();

    sentiment_all_adjnorm_df.head()

    # Save segmented and cleaned text to file
    novel_camel_str = ''.join([re.sub(r'[^\w\s]','',x).capitalize() for x in novel_title_str.split()])
    save_df2csv_and_download(sentiment_all_adjnorm_df, novel_camel_str, '_alladjstdscaler.csv', nodate=True)

    #@title Create SMA using Standard Scaler Normalization:
    Add_Original_Mean = False #@param {type:"boolean"}  (user input)

    # Create Simple Moving Average DataFrame _sma_df from Window Percentage
    col_nonmodels_ls = ['line_no','text_raw','text_clean']
    col_models_ls = list(set(sentiment_all_df.columns.to_list()) - set(col_nonmodels_ls))
    sentiment_all_sma_df = sentiment_all_df[col_nonmodels_ls].copy(deep=True)

    Add_Original_Mean = True

    if Add_Original_Mean:
        for amodel in col_models_ls:
            sentiment_all_sma_df[amodel] = sentiment_all_adjnorm_df[amodel].rolling(win_size, center=True).mean()
    else:
        for amodel in col_models_ls:
            sentiment_all_sma_df[amodel] = sentiment_all_norm_df[amodel].rolling(win_size, center=True).mean()

    # Verify
    _ = sentiment_all_sma_df[col_models_ls].plot(grid=True)

def peakDetection():
    # Which cols hold Model Sentiment Timeseries
    col_nonmodels_ls = ['line_no','text_raw','text_clean']
    col_models_ls = list(set(sentiment_all_sma_df.columns.to_list()) - set(col_nonmodels_ls))
    col_models_ls

    #@title Which Lexicon?
    nt_Model = "textblob" #@param ['vader', 'textblob', 'distilbert', 'nlptown', 'roberta15lg']



    #@title Tune the main Hyperparameter for each of the 4 Peak Detection Algorithms:
    Distance_Min = 360 #@param {type:"slider", min:100, max:1000, step:10}
    Prominence_Min = 0.05 #@param {type:"slider", min:0.01, max:0.1, step:0.01}
    Width_Min = 25 #@param {type:"slider", min:25, max:500, step:5}
    Threshold_Min = 0.007 #@param {type:"slider", min:0.001, max:0.01, step:0.001}

    plt.rcParams['figure.figsize'] = [30, 20]

    # model_name = f'{Sentiment_Model.lower()}_sma{int(win_per)}'
    model_name = Sentiment_Model.lower().strip()

    x = sentiment_all_sma_df[model_name]

    # Peak Algo #1 (by Distance)
    distance_min = Distance_Min # 750

    # Peak Algo #2 (by Prominence)
    prominence_min = Prominence_Min # 0.01

    # Peak Algo #3 (by Width)
    width_min = Width_Min # 175

    # Peak Algo #4 (by Threshold)
    threshold_min = Threshold_Min # 0.001


    peaks, _ = find_peaks(x, distance=distance_min)
    peaks2, _ = find_peaks(x, prominence=prominence_min)      # BEST!
    peaks3, _ = find_peaks(x, width=width_min)
    peaks4, _ = find_peaks(x, threshold=threshold_min)     # Required vertical distance to its direct neighbouring samples, pretty useless


    x_inv = pd.Series([-x for x in sentiment_all_sma_df[model_name].to_list()])

    valleys, _ = find_peaks(x_inv, distance=distance_min)
    valleys2, _ = find_peaks(x_inv, prominence=prominence_min)      # BEST!
    valleys3, _ = find_peaks(x_inv, width=width_min)
    valleys4, _ = find_peaks(x_inv, threshold=threshold_min)     # Required vertical distance to its direct neighbouring samples, pretty useless

    _ = plt.subplot(2, 2, 1)
    _ = plt.grid(True, alpha=0.3)
    _ = plt.plot(x)
    _ = plt.title(f'Distance Peak Detection ({len(peaks)+len(valleys)} Cruxes) \n {len(peaks)} Peaks & {len(valleys)} Valleys')
    _ = plt.plot(peaks, x[peaks], "^g", markersize=7)
    _ = plt.plot(valleys, x[valleys], "vr", markersize=7)
    for x_val in peaks:
        _ = plt.text(x_val, x[x_val], f'-----{x_val}', ha='center', va='bottom', rotation=90, size='large', color='black', weight='semibold')
    for x_val in valleys:
        _ = plt.text(x_val, x[x_val], f'-----{x_val}', ha='center', va='top', rotation=270, size='large', color='black', weight='semibold')

    _ = plt.subplot(2, 2, 2)
    _ = plt.grid(True, alpha=0.3)
    _ = plt.plot(x)
    _ = plt.title(f'Prominence Peak Detection ({len(peaks2)+len(valleys2)} Cruxes) \n {len(peaks2)} Peaks & {len(valleys2)} Valleys')
    _ = plt.plot(peaks2, x[peaks2], "^g", markersize=7)
    _ = plt.plot(valleys2, x[valleys2], "vr", markersize=7)
    for x_val in peaks2:
        _ = plt.text(x_val, x[x_val], f'-----{x_val}', ha='center', va='bottom', rotation=90, size='large', color='black', weight='semibold')
    for x_val in valleys2:
        _ = plt.text(x_val, x[x_val], f'-----{x_val}', ha='center', va='top', rotation=270, size='large', color='black', weight='semibold')

    _ = plt.subplot(2, 2, 3)
    _ = plt.grid(True, alpha=0.3)
    _ = plt.plot(x)
    _ = plt.title(f'Width Peak Detection ({len(peaks3)+len(valleys3)} Cruxes) \n {len(peaks3)} Peaks & {len(valleys3)} Valleys')
    _ = plt.plot(valleys3, x[valleys3], "vr", markersize=7)
    _ = plt.plot(peaks3, x[peaks3], "^g", markersize=7)
    for x_val in peaks3:
        _ = plt.text(x_val, x[x_val], f'-----{x_val}', ha='center', va='bottom', rotation=90, size='large', color='black', weight='semibold')
    for x_val in valleys3:
        _ = plt.text(x_val, x[x_val], f'-----{x_val}', ha='center', va='top', rotation=270, size='large', color='black', weight='semibold')

    _ = plt.subplot(2, 2, 4)
    _ = plt.grid(True, alpha=0.3)
    _ = plt.plot(x)
    _ = plt.title(f'Threshold Peak Detection ({len(peaks4)+len(valleys4)} Cruxes) \n {len(peaks4)} Peaks & {len(valleys4)} Valleys')
    _ = plt.plot(valleys4, x[valleys4], "vr", markersize=7)
    _ = plt.plot(valleys4, x[valleys4], "^g", markersize=7)
    for x_val in peaks4:
        _ = plt.text(x_val, x[x_val], f'-----{x_val}', ha='center', va='bottom', rotation=90, size='large', color='black', weight='semibold')
    for x_val in valleys4:
        _ = plt.text(x_val, x[x_val], f'-----{x_val}', ha='center', va='top', rotation=270, size='large', color='black', weight='semibold')

    _ = plt.suptitle(f'{novel_title_str}\n Peak Detection of Sentiment Analysis (SMA {win_per}%)', fontsize=20)
    _ = plt.grid(True, alpha=0.3)

    _ = plt.show();

    plt.rcParams['figure.figsize'] = [12,8]


    #@title Select a Peak Detection Algorithms to View in Detail (usually Distance or Width is best):
    plt.rcParams['figure.figsize'] = [20, 10]
    Peak_Algorithm = "Width" #@param ["Distance", "Prominence", "Width", "Threshold"]
    if Peak_Algorithm == 'Distance':
        peaks = peaks
        valleys = valleys
    elif Peak_Algorithm == 'Prominence':
        peaks = peaks2
        valleys = valleys2  
    elif Peak_Algorithm == 'Width':
        peaks = peaks3
        valleys = valleys3
    else:
        # Assume Peak_Algorithm == 'Threshold'
        peaks = peaks4
        valleys = valleys4

    # model_name = f'{Sentiment_Model.lower()}_sma10'

    # x = novel_clean_df[model_name]

    # peaks2, _ = find_peaks(x, prominence=peak_prominence)  

    # x_inv = pd.Series([-x for x in novel_clean_df[model_name].to_list()])
    # valleys2, _ = find_peaks(x_inv, prominence=peak_prominence)     

    _ = plt.plot(x)
    _ = plt.plot(peaks, x[peaks], "^g", markersize=15, label='peak sentence#')
    _ = plt.plot(valleys, x[valleys], "vr", markersize=15, label='valley sentence#')
    for x_val in peaks:
        _ = plt.text(x_val, x[x_val], f'    {x_val}', horizontalalignment='left', size='medium', color='black', weight='semibold')
    for x_val in valleys:
        _ = plt.text(x_val, x[x_val], f'    {x_val}', horizontalalignment='left', size='medium', color='black', weight='semibold')
    _ = plt.title(f'{Novel_Title}\n {Peak_Algorithm} Peak Detection \n Sentiment Analysis (SMA {win_per}%)', fontsize=16)
    _ = plt.ylabel('Sentiment')
    _ = plt.xlabel('Sentence No.')
    _ = plt.legend(loc='best')
    _ = plt.grid(True, alpha=0.3)

    filename_plot = f"cruxes_plot_{Novel_Title.replace(' ', '_')}.png"
    _ = plt.savefig(filename_plot, dpi=300)
    _ = plt.show();

    print(f'\n\n     >>>>> SAVED PLOT TO FILE: [{filename_plot}] <<<<<')

    # Download Crux Point Plot file 'crux_plot.png' to your laptop
    files.download(filename_plot)


def crux_extraction(sentencesAroundCrux):
    # preconditions: sentencesAroundCrux is an integer, probably between 1 and 20

    # May have to rerun!!
    # Print Context around each Sentiment Peak
    halfwin = int(sentencesAroundCrux/2)
    crux_sents_ls = []
    newline = '\n'

    print('==================================================')
    print('============     Peak Crux Points   ==============')
    print('==================================================\n\n')

    # for i, apeak in enumerate(peaks2):
    for i, peak in enumerate(peaks):
        crux_sents_ls = []
        for sent_idx in range(peak-halfwin,peak+halfwin+1):
            sent_cur = sentiment_df.iloc[sent_idx].text_raw
            if sent_idx == peak:
                sent_str = sent_cur.upper()
            else:
                sent_str = sent_cur
            crux_sents_ls.append(sent_str)
    
    # context_ls = sentiment_df.iloc[apeak-halfwin:apeak+halfwin].text_raw
    print(f"Peak #{i} at Sentence #{peak}:\n\n{newline.join(crux_sents_ls)}\n\n\n")

    print('==================================================')
    print('===========     Crux Valley Points    ============')
    print('==================================================\n\n')

    # for i, avalley in enumerate(valleys2):
    for i, avalley in enumerate(valleys):
        crux_sents_ls = []
        for sent_idx in range(avalley-halfwin,avalley+halfwin+1):
            sent_cur = sentiment_df.iloc[sent_idx].text_raw
            if sent_idx == avalley:
                sent_str = sent_cur.upper()
            else:
                sent_str = sent_cur
            crux_sents_ls.append(sent_str)

    # context_ls = novel_df.iloc[avalley-halfwin:avalley+halfwin].text_raw
    print(f"Valley #{i} at Sentence #{avalley}:\n\n{newline.join(crux_sents_ls)}\n\n\n")

    filename_cruxes = f"cruxes_context_{Novel_Title.replace(' ', '_')}.txt" 

    with open(filename_cruxes, 'w') as f:
        f.write(str(cap))


    # Download Crux Point Report file 'cruxes.txt' to your laptop
    files.download(filename_cruxes)



