"""simplifiedSA.py

[TODO: description]

Based on an ipynb by Jon Chun, Feb. 2023 TODO: link if on github
Version 1: Alex Felleson, May 2023
"""

# Standard library imports
import re
import datetime
import configparser
import os
# import string # defines constants such as ascii_letters (currently not used)

import numpy as np
# import modin.pandas as pd # Modin — uses multiple cores for operations on pandas dfs. I think this uses an old version of pandas, forcing a reinstall of pandas :( Also removes from functionality from df methods, including sort_values(). So no longer using this, for now.
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

# For gutenberg import (currently not in frontend UI)
import requests

# For segmenting by sentence
THIS_SOURCE_FILE_PATH = os.path.abspath(__file__)
THIS_SOURCE_FILE_DIRECTORY = os.path.dirname(THIS_SOURCE_FILE_PATH)
nltk_download_dir = os.path.join(THIS_SOURCE_FILE_DIRECTORY, 'my_nltk_dir')
import nltk
nltk.download('punkt', download_dir=nltk_download_dir)
nltk.data.load(os.path.join(nltk_download_dir, 'tokenizers','punkt','english.pickle'))
    # Note: To support other langauges, add more nltk.data.load() commands like the one above; just change 'english' to the new language name
from nltk.tokenize import sent_tokenize

# For text cleaning
from cleantext import clean # note: remove dependency (clean-text library) if possible
import contractions # remove dependency?

# For timeseries normalizations
from sklearn.preprocessing import MinMaxScaler, StandardScaler # TODO: figure out if necessary or if you can do this manually

# For peak detection
from scipy.signal import find_peaks


# Read the toml file that contains settings that advanced users can edit (via ??? TODO?)
config = configparser.ConfigParser()
def get_filepath_in_src_dir(filename: str) -> str:
    result = os.path.join(THIS_SOURCE_FILE_DIRECTORY, filename)
    if not os.path.exists(result):
        raise Exception("Failed to create filepath in current directory")
    else:
        return result

config.read(get_filepath_in_src_dir('SA_settings.toml'))

# Set up matplotlib
plt.rcParams["figure.figsize"] = (20,10)
plt.rcParams["font.size"] = 22
# TODO: add more setup commands from old util file?

# Global variables
TEXT_ENCODING = config.get('imports', 'text_encoding') # TODO: Use the chardet library to detect the file's character encoding instead
PARA_SEP = config.get('imports', 'paragraph_separation')
CURRENT_DIR = os.getcwd()
# TODO: consider making TITLE a global variable. I don't like that because the user has to set it. Other options are passing it to every function and making a SAobject that has title as a member datum.
ALL_MODELS_LIST = ['vader', 'textblob', 'distilbert'] #TODO: add nlptown and roberta15lg
# Custom Exceptions
class InputFormatException(Exception):
    pass

## COMMON FUNCTIONS ##


# Code to import a csv as a pd.df that can be passed to the model functions: df = pd.read_csv('saved_file.csv')

# def import_df(filepath: str) -> pd.DataFrame: 
#     # This function should exist (rather than having the user import their df themself) only if we're using modin.pandas as pd instead of using pandas
#     file_extension = filepath.split('.')[-1]
#     if file_extension=="csv": # Technically, pd.read_csv() should also work with .txt files
#         return pd.read_csv(filepath)
#     else:
#         raise InputFormatException("Can only import a dataframe from a .csv")
#     # Could have requirements for filename formatting and get title from filename here, and be passing the text around in this whole program as a dict with title: content (str or df) or turn it into an object with member data title, etc as stated in another comment (go look at that one)

def uniquify(path: str) -> str:
    """Generate a unique filename for a file to be saved.
    
    Append (1), (2), etc to a file name if the file already exists.

    Args:
        path (str): complete path to the file, including the extension

    Returns:
        str: edited complete path to the file, including the extension
    """
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path

def download_df(df_obj: pd.DataFrame, title: str, save_filepath=CURRENT_DIR, filename_suffix='_save', nodate=True):
    '''
    INPUT: DataFrame object and suffix to add to output csv filename
    OUTPUT: Write DataFrame object to csv file (both temp VM and download)
    '''
    camel_title = ''.join([re.sub(r'[^\w\s]','',x).capitalize() for x in title.split()])
    if isinstance(df_obj, pd.DataFrame):
        if nodate:
            out_filename = camel_title.split('.')[0] + filename_suffix + ".csv"
        else:
            datetime_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            out_filename = camel_title.split('.')[0] + '_' + datetime_str + filename_suffix + ".csv"

        completepath = os.path.join(save_filepath,out_filename)
        df_obj.to_csv(uniquify(completepath), index=False)  # TODO: test
    else:
        print(f'ERROR: Object is not a DataFrame [download_df2csv_and_download()]')
        return -1


## IPYNB SECTIONS AS FUNCTIONS ##

def upload_text(filepath: str) -> str:

    filename_ext_str = filepath.split('.')[-1]
    if filename_ext_str == 'txt':
        # raw_text_str = uploaded[novel_filename_str].decode(TEXT_ENCODING)
        with open(filepath,'r') as f:
            raw_text_str = f.read()
            # TODO: figure out if decoding (using TEXT_ENCODING) would be useful here
            return raw_text_str
    else:
        raise InputFormatException("Must provide path to a plain text file (*.txt)")

     # return as single-item dict with title as key instead? or a custom "SAtext" object with data members title, body, segmented_body, clean_body?

def preview(something) -> str: # would make more sense as a method imo. could take in an SAtext object and be a method, or take in a dict to be able to print the file name
    # Return string showing beginning and end of text for user verification, or show some clean text lines from a df
    # input can be a string or a df
    if type(something) == str:
        if len(something)<1000:
            stringToPeep =     (f'  Length of text (in characters): {len(something)}\n' +
                                f'Entire text: \n' +
                                something)
        stringToPeep =     (f'  Length of text (in characters): {len(something)}\n' +
                '====================================\n\n' +
                f'Beginning of text:\n\n {something[:500]}\n\n\n' +
                '\n------------------------------------\n' +
                f'End of text:\n\n {something[-500:]}\n\n\n')
    elif type(something) == pd.DataFrame:
        if len(something)<20: # TODO: test this comparison value is exactly correct
            stringToPeep = ("Short dataframe. Here's the whole thing: \n" +
                            '\n'.join(something['cleaned_text'][0:9].astype(str)))
        stringToPeep = "First 10 sentences:\n\n"
        stringToPeep += '\n'.join(something['cleaned_text'][0:9].astype(str))
        stringToPeep += "\n\n ... \n\n"
        stringToPeep += "Last 10 sentences:\n\n"
        stringToPeep += '\n'.join(something['cleaned_text'][-11:-1].astype(str))
    else:
        TypeError("You may only preview a string or dataframe with a cleaned_text column")
        stringToPeep = ""
        
    stringToPeep += "\n\n"
    print(stringToPeep) # TODO: remove all print statements (at very end of package development)
    return stringToPeep

# TODO: clean up & test or remove
def gutenberg_import(title: str, Gutenberg_URL: str, 
                    sentence_first_str = None, sentence_last_str = None) -> str:
    #@title Enter the URL of your novel at ***gutenberg.net.au***
    #@markdown Paste the URL to the ***HTML version*** (not plain text).

    # title = 'Frankenstein by Mary Shelley'  #@param {type: "string"}

    # Gutenberg_URL = 'https://gutenberg.net.au/ebooks/z00006.html'  #@param {type: "string"}
    # the first sentence in the body of your novel: sentence_first_str
    # the last sentence in the body of your novel: sentence_last_str


    # Get raw HTML of novel from Gutenberg.net.au
    response=requests.get(Gutenberg_URL)  # TODO: Pass the URL to the .get() method of the requests object
    html = response.text

    # Use HTML <p> tags to extract text into list of paragraphs
    paragraphs = re.findall(r'<p>(.*?)</p>', html, flags=re.DOTALL)

    if (len(paragraphs) < 3):
        raise InputFormatException("Fewer than three paragraphs detected")

    # TODO: figure out what this is doing and why (seems like it's just undoing what we did, plus the \r\n replacement)
    # Concatenate all paragraphs into a single novel string
    # For every paragraph, replace each hardcoded \r\n with a single space
    paragraphs_flat = [re.sub(r'\r\n', ' ', paragraph) for paragraph in paragraphs]
    # Concatenate all paragraphs into single strings separated by two \n
    raw_text_str = '\n\n'.join(paragraphs_flat)
    
    if (sentence_first_str is not None  and  sentence_last_str is not None): # using optional function args
        # Remove header
        raw_text_str = ' '.join(raw_text_str.partition(sentence_first_str)[1:])
        # Remove footer
        raw_text_str = ' '.join(raw_text_str.partition(sentence_last_str)[:2])
    
    return(raw_text_str)


def segment_sentences(raw_text_str:  str) -> list: # TODO: don't print/have a verification string if there aren't parameters to adjust here
    # Segment by sentence
    sentences_list = sent_tokenize(raw_text_str) # using nltk.tokenize

    # Most of the rest of this function (not the delete empty sentences part) is just returning things for user verification
    sentence_count = len(sentences_list)
    num_senteces_to_show = 5

    verificationString = f'\n----- First {num_senteces_to_show} Sentences: -----\n\n'
    for i, asent in enumerate(sentences_list[:num_senteces_to_show]):
        verificationString += f'Sentences #{i}: {asent}\n'

    print(f'\n----- Last {num_senteces_to_show} Sentences: -----\n')
    for i, asent in enumerate(sentences_list[-num_senteces_to_show:]):
        verificationString += f'Sentences #{sentence_count - (num_senteces_to_show - i)}: {asent}\n'

    verificationString += f'\n\nThere are {sentence_count} Sentences in the text\n'

    # Delete the empty Sentences and those without any alphabetic characters
    sentences_list = [x.strip() for x in sentences_list if len(x.strip()) > 0]
    sentences_list = [x.strip() for x in sentences_list if re.search('[a-zA-Z]', x)]
    
    num_sentences_removed = sentence_count - len(sentences_list)
    if (num_sentences_removed!=0):
        verificationString += f'\n\n{num_sentences_removed} empty and/or non-alphabetic sentences removed\n'
    # Q: How does sentence number & returning sentences around crux points still match up after doing this? Or do we not care exactly where the crux is in the original text? A: The "raw text" column in sentiment_df is the segmentedBySentences result

    # Plot distribution of sentence lengths
    # _ = plt.hist([len(x) for x in sentences_list], bins=100)

    print(verificationString) # TODO: same deal as before: have a separate verification function that returns this? return this in a list along with the actual return value? just print it?
    
    return sentences_list


def clean_string(dirty_str: str) -> str: # to be called within create_df_with_text (formerly known as clean_text)
    #TODO: add options, and add more functions in here that take care of stuff clean-text doesn't, like emoticons
    '''
    INPUT: a raw string
    OUTPUT: a clean string
    '''

    contraction_expanded_str = contractions.fix(dirty_str)

    clean_str = clean(contraction_expanded_str, # TODO: detemine if we want to keep this dependency (clean-text). Chun says no. Find alternative?
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
        no_punct=False,                 # remove punctuation
        # replace_with_punct="",          # instead of removing punctuations, you may replace them
        # replace_with_url="<URL>",
        # replace_with_email="<EMAIL>",
        # replace_with_phone_number="<PHONE>",
        # replace_with_number="<NUMBER>",
        # replace_with_digit="0",
        # replace_with_currency_symbol="<CUR>",
        lang="en"                       # set to 'de' for German special handling
    )

    # Replace all new lines/returns with single whitespace
    # clean_str = clean_str.replace('\n\r', ' ') # I think these are commented out bc clean() with no_line_breaks=True already does those
    # clean_str = clean_str.replace('\n', ' ')
    # clean_str = clean_str.replace('\r', ' ')
    clean_str = ' '.join(clean_str.split()) # remove leading, trailing, and repeated spaces

    return clean_str 


def create_clean_df(sentences_list: list, title: str, save = False, save_filepath = CURRENT_DIR) -> pd.DataFrame:
    
    # Create sentiment_df to hold text sentences and corresponding sentiment values
    sentiment_df = pd.DataFrame
    sentiment_df = pd.DataFrame({'text_raw': sentences_list})
    sentiment_df['text_raw'] = sentiment_df['text_raw'].astype('string')
    sentiment_df['text_raw'] = sentiment_df['text_raw'].str.strip()

    # clean the 'text_raw' column and create the 'cleaned_text' column
    # novel_df['cleaned_text'] = hero.clean(novel_df['text_raw'])
    sentiment_df['cleaned_text'] = sentiment_df['text_raw'].apply(lambda x: clean_string(x)) # call clean_str()
    sentiment_df['cleaned_text'] = sentiment_df['cleaned_text'].astype('string')
    sentiment_df['cleaned_text'] = sentiment_df['cleaned_text'].str.strip() # strips leading and trailing whitespaces & newlines
    sentiment_df['text_raw_len'] = sentiment_df['text_raw'].apply(lambda x: len(x))
    sentiment_df['cleaned_text_len'] = sentiment_df['cleaned_text'].apply(lambda x: len(x))

    print(sentiment_df.head())
    print(sentiment_df.info())

    # Drop Sentence if Raw length < 1 (Double check)
    sentiment_df = sentiment_df.loc[sentiment_df['text_raw_len'] > 0]
    sentiment_df.shape

    # Fill any empty cleaned_text with a neutral word
    neutral_word = 'NEUTRALWORD'
    sentiment_df[sentiment_df['cleaned_text_len'] == 0]['cleaned_text'] = neutral_word
    sentiment_df[sentiment_df['cleaned_text_len'] == 0]['cleaned_text_len'] = 11
    sentiment_df['cleaned_text_len'].sort_values(ascending=True) # , key=lambda x: len(x), inplace=True)
    # sentiment_df.cleaned_text.fillna(value='', inplace=True)

    # Add Line Numbers
    sentence_nums = list(range(sentiment_df.shape[0]))
    sentiment_df.insert(0, 'sentence_num', sentence_nums)

    # View the shortest lines by text_raw_len
    # print("shortest lines by text_raw_len: \n" + sentiment_df.sort_values(by='text_raw_len').head(20)) # only works if you're using pandas instead of modin.pandas

    if save:
        download_df(sentiment_df, title, save_filepath=save_filepath, filename_suffix='_cleaned')
    
    return sentiment_df


def preprocess_text(raw_text_str: str, title: str, save = False, save_filepath = CURRENT_DIR)  -> pd.DataFrame:
    sentences_list = segment_sentences(raw_text_str)
    return create_clean_df(sentences_list, title, save, save_filepath)


def vader(sentiment_df: pd.DataFrame, title: str, save_filepath = CURRENT_DIR) ->  pd.DataFrame:
    print("vader")
    """ TODO

    Args:
        sentiment_df (pd.DataFrame):  TODO
        title (str): title of the text
        plot (str, optional): "display", "save", "both", or "none". Defaults to "none".
        save_filepath (str): TODO

    Returns:
        pd.DataFrame:  TODO
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_vader_ls = [sid_obj.polarity_scores(asentence)['compound'] for asentence in sentiment_df['cleaned_text'].to_list()]
    
    # Create new VADER DataFrame to save results
    vader_df = sentiment_df[['sentence_num', 'text_raw', 'cleaned_text']].copy(deep=True)
    vader_df['sentiment'] = pd.Series(sentiment_vader_ls) 
        
    return vader_df
    # TODO: consider just appending these results to sentiment_df, and if someone wants the vader data only, they can subset that df. This woudl eliminate the need for combine_all_results or whatever in the main pipeline


def textblob(sentiment_df: pd.DataFrame, title: str) -> pd.DataFrame:
    print("textblob")
    from textblob import TextBlob
    sentiment_textblob_ls = [TextBlob(asentence).polarity for asentence in sentiment_df['cleaned_text'].to_list()]
    # sentiment_df['sentiment'] = sentiment_df['cleaned_text'].apply(lambda x : TextBlob(x).sentiment.polarity) # add textblob column to sentiment_df
    
    # Create new TextBlob DataFrame to save results
    textblob_df = sentiment_df[['sentence_num', 'text_raw', 'cleaned_text']].copy(deep=True)
    textblob_df['sentiment'] = pd.Series(sentiment_textblob_ls) 
    
    return textblob_df


# Create class for data preparation for transformer models
class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts
    
    def __len__(self):
        return len(self.tokenized_texts["input_ids"])
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}
        
        
def distilbert(sentiment_df: pd.DataFrame, title: str) -> pd.DataFrame:
    print("distilbert")
    # Some of these might be needed in other transformer models to be added later (TODO)
    from transformers import AutoTokenizer #, AutoModelWithLMHead  # T5Base 50k
    from transformers import AutoModelForSequenceClassification, Trainer
    # from transformers import pipeline
    # from transformers import AutoModelForSeq2SeqLM, AutoModelWithLMHead
    # from transformers import BertTokenizer, BertForSequenceClassification
    # import sentencepiece
    
    # Load tokenizer and model, create trainer
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    trainer = Trainer(model=model)

    # (there was a test here i don't think was necessary to run, so it's just in the test file)

    # Compute Sentiment Time Series
    line_ls = sentiment_df['cleaned_text'].to_list()

    # Tokenize texts and create prediction data set
    tokenized_texts = tokenizer(line_ls,truncation=True,padding=True)
    pred_dataset = SimpleDataset(tokenized_texts)

    # Run predictions
    prediction_results = trainer.predict(pred_dataset)

    # Transform predictions to labels
    sentiment_ls = np.argmax(prediction_results.predictions, axis=-1) # used to be prediction_results.predictions.argmax(-1)
    labels_ls = pd.Series(sentiment_ls).map(model.config.id2label)
    scores_ls = (np.exp(prediction_results[0])/np.exp(prediction_results[0]).sum(-1,keepdims=True)).max(1)

    # Create DataFrame with texts, predictions, labels, and scores
    sentence_num_ls = list(range(len(sentiment_ls)))
    distilbert_df = pd.DataFrame(list(zip(sentence_num_ls, line_ls,sentiment_ls,labels_ls,scores_ls)), columns=['sentence_num','line','sentiment','label','score'])

    # TODO: decide where to put this
    # Ensure balance of sentiments
    # distilbert_df['distilbert'].unique()
    # _ = distilbert_df['label'].hist()

    return distilbert_df


def combine_model_results(sentiment_df: pd.DataFrame, title, **kwargs) -> pd.DataFrame:
    print("combine_model_results")
    # TODO: make these named params instead of freeform? as a check.
    '''
    Optional named args: vader = vader_df, textblob = textblob_df, 
                         distilbert = distilbert_df, nlptown = nlptown_df, 
                         roberta15lg = roberta15lg_df
    TODO: make sure this is working
    '''
    # Merge all dataframes into a new dataframe
    all_sentiments_df = sentiment_df[['sentence_num','text_raw','cleaned_text']].copy(deep=True)
    for key, value in kwargs.items():
        try:
            all_sentiments_df[key] = value['sentiment']
        except:
            print(f'Warning: failed to append {key} sentiments\n')
    
    return all_sentiments_df

def compute_sentiments(sentiment_df: pd.DataFrame, title: str, models = ALL_MODELS_LIST) -> pd.DataFrame:
    print("compute_sentiments")
    all_sentiments_df = sentiment_df[['sentence_num','text_raw','cleaned_text']].copy(deep=True)
    if "vader" in models:
        all_sentiments_df['vader'] = vader(sentiment_df,title)['sentiment']
    if "textblob" in models:
        all_sentiments_df['textblob'] = textblob(sentiment_df,title)['sentiment']
    if "distilbert" in models:
        all_sentiments_df['distilbert'] = distilbert(sentiment_df,title)['sentiment']
    for user_model in models:
        if user_model not in ALL_MODELS_LIST:
            print(f"Warning: {user_model} model not found in list of accepted models. Check your spelling.")
    return all_sentiments_df

# This function works on a df containing multiple models, and it creates a new df with the same column names but new sentiment values.
# TODO: Also create functions that allow the user to input a df with only one model's sentiment values and append adjusted & normalized sentiments as new columns on the same df, in case they want to compare different adjustments & smoothing methods for the same model.
def plot_sentiments(all_sentiments_df: pd.DataFrame, 
                        title: str, 
                        models = ALL_MODELS_LIST,
                        adjustments="normalizedAdjMean", # TODO: add a 'rescale' option, where all points are rescaled from their model's original scale to -1 to 1
                        smoothing="sma",
                        plot = "save",
                        save_filepath=CURRENT_DIR, 
                        window_pct = 10,
                        ) -> pd.DataFrame:
    """Saves a .png plot of raw or adjusted sentiments from the selected models.

    Saves a .png plot of raw, normed, or normed & adjusted sentiments 
    from the selected models to the specified directory. Sentiment curves
    are smoothed using the specified method before plotting. The function
    also returns the points from the plot in the form of a [TODO].
    
    @param models: must contain models with the same timesereies length 
    (sentimentR cannot be included without adjustments — used to be model_samelen_ls)

    Args:
        all_sentiments_df (pd.DataFrame): Dataframe containing sentiment 
            values in columns named after the models in `models`
        title (str): Title of text
        adjustments (str): "none" (plot raw sentiments), "normalizedZeroMean" 
            (normalize to mean=0, sd=1), "normalizedAdjMean" (normalize and add
            the scaled mean that woudld be computed by adjusting the 
            original scores so their range is exactly[TODO: change depending on chun's answer] -1 to 1).
        smoothing (str): "sma" (simple moving average, aka sliding 
            window with window size determined by window_pct), "lowess"
            (LOWESS smoothing using parameter = [TODO])
        plot (str): "display", "save", "both", or "none"
        save_filepath (str): path (ending in '/') to the directory
            the resulting plot png should be stored in.
            Defaults to the current working directory.
        window_pct (int): percentage of total text length to use as the
            window size for SMA smoothing
        models (list[str]): list of the lowercase names of the 
            models to plot. These models' timeseries/results must have the same 
            length. (SentimentR cannot be included without adjustments.)

    Returns:
        TODO

    """
    print("plot_sentiments")
    
    if window_pct > 20 or window_pct < 1:
        print("Warning: window percentage outside expected range")
    window_size = int(window_pct / 100 * all_sentiments_df.shape[0])

    camel_title = ''.join([re.sub(r'[^\w\s]','',x).capitalize() for x in title.split()])
    
    if adjustments == "raw":
        # Plot Raw Timeseries
        raw_rolling_mean = all_sentiments_df[['sentence_num','text_raw','cleaned_text']].copy(deep=True)
        raw_rolling_mean[models] = all_sentiments_df[models].rolling(window_size, center=True).mean() #Q: won't this have NA vals for the first few?
        plt.figure().clear()
        ax = raw_rolling_mean[models].plot(grid=True, lw=3)
        ax.set_title(f'{title} Sentiment Analysis \n Raw Sentiment Timeseries')
        plt.xlabel('Sentence Number')
        plt.ylabel('Sentiment')
        if plot == "save" or plot == "both":
            completepath = os.path.join(save_filepath,f"{camel_title}_rawSentiments.png")
            plt.savefig(uniquify(completepath))
        if plot == "display" or plot == "both":
            plt.show()
        
        return raw_rolling_mean

    else:
        # Compute the mean of each raw Sentiment Timeseries and adjust to [-1.0, 1.0] Range
        models_adj_mean_dt = {}
        for model in models:
            model_min = all_sentiments_df[model].min()
            model_max = all_sentiments_df[model].max()
            model_range = model_max - model_min
            model_raw_mean = all_sentiments_df[model].mean()
            # Rescaling formula: Rescaled Value = (Original Value - Original Minimum) / (Original Maximum - Original Minimum) * (New Maximum - New Minimum) + New Minimum
                # TODO: rescale based on each model's potential range instead?
            if model_range > 2.0:
                models_adj_mean_dt[model] = 2*(model_raw_mean + model_min)/model_range - 1.0
            elif model_range < 1.1: #Q: why not <= 1.0?
                models_adj_mean_dt[model] = 2*(model_raw_mean + model_min)/model_range - 1.0
                models_adj_mean_dt[model] = 2 * (model_raw_mean - model_min) / model_range - 1.0
            else:
                models_adj_mean_dt[model] = model_raw_mean
            print(f'Model: {model}\n  Raw Mean: {model_raw_mean}\n  Adj Mean: {models_adj_mean_dt[model]}\n  Min: {model_min}\n  Max: {model_max}\n  Range: {model_range}\n')

        # Normalize Timeseries with StandardScaler (u=0, sd=+/- 1)
        all_sentiments_norm_df = all_sentiments_df[['sentence_num','text_raw','cleaned_text']].copy(deep=True)
        all_sentiments_norm_df[models] = StandardScaler().fit_transform(all_sentiments_df[models])

        if adjustments == "normalizedZeroMean":
            # Plot Normalized Timeseries to same mean (Q: Is this mean 0? If not, change filename below.)
            norm_rolling_mean = all_sentiments_norm_df[['sentence_num','text_raw','cleaned_text']].copy(deep=True)
            norm_rolling_mean[models] = all_sentiments_norm_df[models].rolling(window_size, center=True).mean()
            plt.figure().clear()
            ax = norm_rolling_mean[models].plot(grid=True, lw=3)
            ax.set_title(f'{title} Sentiment Analysis \n Normalization: Standard Scaler')
            plt.xlabel('Sentence Number')
            plt.ylabel('Sentiment')
            if plot == "save" or plot == "both":
                completepath = os.path.join(save_filepath,f"{camel_title}_normalizedZeroMeanSentiments.png")
                plt.savefig(uniquify(completepath))
            if plot == "display" or plot == "both":
                plt.show()
            

            return norm_rolling_mean

        else: # adjustments == "normalizedAdjMean"
            # Plot StandardScaler + Original Mean
            # Plot Normalized Timeseries to their adjusted/rescaled original means
            all_sentiments_adjnorm_df = all_sentiments_df[['sentence_num','text_raw','cleaned_text']].copy(deep=True)
            for amodel in models:
                all_sentiments_adjnorm_df[amodel] = all_sentiments_norm_df[amodel] + models_adj_mean_dt[amodel]

            norm_adj_rolling_mean = all_sentiments_adjnorm_df[['sentence_num','text_raw','cleaned_text']].copy(deep=True)
            norm_adj_rolling_mean[models] = all_sentiments_adjnorm_df[models].rolling(window_size, center=True).mean()
            plt.figure().clear()
            ax = norm_adj_rolling_mean[models].plot(grid=True, lw=3)
            ax.set_title(f'{title} Sentiment Analysis \n Normalization: Standard Scaler + Scaled Mean Adjustment')
            plt.xlabel('Sentence Number')
            plt.ylabel('Sentiment')
            if plot == "save" or plot == "both":
                completepath = os.path.join(save_filepath,f"{camel_title}_normalizedAdjustedMeanSentiments.png")
                plt.savefig(uniquify(completepath))
            if plot == "display" or plot == "both":
                plt.show()

            return norm_adj_rolling_mean
        
    #TODO: add lowess option
    # from statsmodels.nonparametric.smoothers_lowess import lowess
    # y=current_sentiment_arc_df[selected_model.value].values
    # x=np.arange(current_sentiment_arc_df.shape[0]) # i think this is just sentence num?
    # lowess(y, x, frac=1/30)[:,1].tolist()

def find_cruxes(smoothed_sentiments_df: pd.DataFrame, 
                title: str,
                model: str,
                algo = "width",
                plot = "save",
                save_filepath = CURRENT_DIR,
                distance_min = 360,
                prominence_min = 0.05,
                width_min = 25
                ) -> tuple[list[int],list[float],list[int],list[float]]:
    """[summary] TODO
    
    Uses find_peaks() from scipy.signal (using the parameter specified
    by 'algo') to identify peaks and troughs in one model's sentiment
    plot. Returns 

    Args:
        smoothed_sentiments_df (pd.DataFrame): TODO
        title (str): title of text
        model (str): 'vader', 'textblob', or 'distilbert'
        algo (str): "distance", "prominence", or "width". Defaults to 
            "width".
        plot (str): "display", "save", "both" or "none". Defaults to "save".
        save_filepath (str): path (ending in '/') to the directory
            the plot should be stored in as a .png. Defaults to the 
            current working directory.
        distance_min (int, only needed if algo="distance"): 
            Required minimum number of sentences (>= 1) between 
            neighboring peaks. The smallest peaks are removed  until 
            the condition is fulfilled for all remaining peaks. Defaults 
            to 360. TODO: add accepted ranges for all these params and add checks in the code (auto correct to max or min)
        prominence_min (float, only needed if algo="prominence"): TODO
        width_min (int, only needed if algo="width"): TODO
        
    Returns:
        A tuple containing four lists: peak x-values, peak y-values,
            trough x-values, and trough y-values
    """
    
    model_name = model.lower().strip() # (Just in case the accepted values for the model param 
                                       # get messed up in future edits)

    x = smoothed_sentiments_df[model_name]
    x_inverted = pd.Series([-x for x in smoothed_sentiments_df[model_name].to_list()])

    if algo == 'Distance':
        peaks, _ = find_peaks(x, distance=distance_min)
        valleys, _ = find_peaks(x_inverted, distance=distance_min)
    elif algo == 'Prominence':
        peaks, _ = find_peaks(x, prominence=prominence_min)
        valleys, _ = find_peaks(x_inverted, prominence=prominence_min)
    else: # algo == 'Width'
        peaks, _ = find_peaks(x, width=width_min)
        valleys, _ = find_peaks(x_inverted, width=width_min)

    if plot != "none":
        plt.figure().clear()
        plt.plot(smoothed_sentiments_df[model_name])
        _ = plt.plot(peaks, x[peaks], "^g", markersize=15, label='peak sentence#') # green triangles
        _ = plt.plot(valleys, x[valleys], "vr", markersize=15, label='valley sentence#') #red triangles
        yadjust = (smoothed_sentiments_df[model_name].max(skipna=True) - smoothed_sentiments_df[model_name].min(skipna=True)) / 40
        print(yadjust)
        for x_val in peaks: # x_val = index of a peak in x
            _ = plt.text(x_val, x[x_val]+yadjust, f'{x_val}', horizontalalignment='center', color='black', weight='semibold', fontsize=16)
        for x_val in valleys:
            _ = plt.text(x_val, x[x_val]-2*yadjust, f'{x_val}', horizontalalignment='center', color='black', weight='semibold', fontsize=16)
        _ = plt.title(f'{title} \n {algo}-based peak detection ({len(peaks)+len(valleys)} cruxes) \n {len(peaks)} Peaks & {len(valleys)} Valleys')
        plt.xlabel('Sentence Number')
        plt.ylabel('Sentiment')
        _ = plt.legend(loc='best')
        _ = plt.grid(True, alpha=0.3)

    if plot == "save" or plot == "both":
        completepath = os.path.join(save_filepath,f"{title} cruxes ({algo} algorithm).png")
        plt.savefig(uniquify(completepath))
    if plot == "display" or plot == "both":
        plt.show()
    
    return peaks, list(x[peaks]), valleys, list(x[valleys])
    # TODO: would be nice to have a way for the user to click on a plot where they think the peaks 
    # and valleys should be, and then easily convert those chosen points to the right format to 
    # highlight them on the plot and pass to the context function

def crux_context(sentiment_df: pd.DataFrame, 
                 peaks: list, 
                 valleys: list, 
                 n=10) -> tuple[list,str]:
    """Return sentences around each sentiment crux

    Args:
        sentiment_df (pd.DataFrame): dataframe with original raw text
        peaks (list): indices (sentence nums) of peaks
        valleys (list): indices (sentence nums) of troughs
        n (int, optional): Number of sentences around 
            each crux point to display. Defaults to 10.
            
    Returns:
        crux_context_list (list of tuples): each tuple contains the 
            string "peak" or "valley", the location of the crux point
            (int sentence number), and a list of the sentences around it
            (list[str])
        crux_context_str (str): a string displaying the n sentences around 
            each peak or valley, labeled with the same information 
            contained in crux_context_list
    """
    
    halfwindow = int(n/2) + 1 # round up
    newline = '\n'
    crux_context_list = []

    crux_context_str = '=================================================='
    crux_context_str += '============     Peak Crux Points   =============='
    crux_context_str += '==================================================\n\n'

    for i, peak in enumerate(peaks): # Iterate through all peaks
        peaks_list = sentiment_df.iloc[peak-halfwindow:peak+halfwindow].text_raw
        crux_context_str += f"Peak #{i} at Sentence #{peak}:\n\n{newline.join(peaks_list)}\n\n\n"
        crux_context_list.append(("peak",peak,peaks_list))

    crux_context_str += '=================================================='
    crux_context_str += '===========     Crux Valley Points    ============'
    crux_context_str += '==================================================\n\n'

    for i, valley in enumerate(valleys): # Iterate through all valleys
        crux_valleys_list = sentiment_df.iloc[valley-halfwindow-1:valley+halfwindow].text_raw
        crux_context_str += f"Valley #{i} at Sentence #{valley}:\n\n{newline.join(crux_valleys_list)}\n\n\n"
        crux_context_list.append(("valley",valley,crux_valleys_list))

    return crux_context_list, crux_context_str

