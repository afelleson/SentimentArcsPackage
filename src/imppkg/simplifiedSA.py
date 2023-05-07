"""simplifiedSA.py

[TODO: description]

Based on a .ipynb by Jon Chun, Feb. 2023 TODO: link if on github
Version 1: Alex Felleson, May 2023
"""

# Standard library imports
import re
import datetime
import configparser
import os # used to get path to current file for SA_settings.toml location
# import string # defines constants such as ascii_letters (currently not used)

import numpy as np
# import modin.pandas as pd # Modin — uses multiple cores for operations on pandas dfs. I think this uses an old version of pandas, forcing a reinstall of pandas :( Also removes from functionality from df methods, including sort_values(). So no longer using this, for now.
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

# For gutenberg import (currently not in frontend UI)
import requests

# For segmenting by sentence
THIS_PACKAGE_FILE_PATH = os.path.abspath(__file__)
PACKAGE_SRC_DIRECTORY = os.path.dirname(THIS_PACKAGE_FILE_PATH)
nltk_download_dir = os.path.join(PACKAGE_SRC_DIRECTORY, 'my_nltk_dir')
import nltk
nltk.download('punkt', download_dir=nltk_download_dir)
nltk.data.load(os.path.join(nltk_download_dir, 'tokenizers/punkt/english.pickle'))
    # Note: To support other langauges, add more nltk.data.load() commands like the one above; just change 'english' to the new language name
from nltk.tokenize import sent_tokenize

# For text cleaning
from cleantext import clean # note: remove dependency (clean-text library) if possible
import contractions # remove dependency?

# For timeseries normalizations
from sklearn.preprocessing import MinMaxScaler, StandardScaler # TODO: figure out if necessary or if you can do this manually

# For peak detection
from scipy.signal import find_peaks


# Read the toml file that contains settings that advanced users can edit (via ???)
config = configparser.ConfigParser()
def get_filepath_in_current_package_dir(filename: str) -> str:
    result = os.path.join(PACKAGE_SRC_DIRECTORY, filename)
    if result == None:
        raise Exception("Failed to create filepath in current directory")
    else:
        return result

config.read(get_filepath_in_current_package_dir('SA_settings.toml'))

# Set up matplotlib
plt.rcParams["figure.figsize"] = (20,10)
# TODO: add more setup commands from old util file?

# Global variables
TEXT_ENCODING = config.get('imports', 'text_encoding') # TODO: Use the chardet library to detect the file's character encoding instead
PARA_SEP = config.get('imports', 'paragraph_separation')
CURRENT_DIR = os.getcwd()
# TODO: consider making TITLE a global variable. I don't like that because the user has to set it. Other options are passing it to every function and making a SAobject that has title as a member datum.

# Custom Exceptions
class InputFormatException(Exception):
    pass

## COMMON FUNCTIONS ##

def test_func():
    print("test_func() ran")

# Code to import a csv as a pd.df that can be passed to the model functions: df = pd.read_csv('saved_file.csv')

# def import_df(filepath: str) -> pd.DataFrame: 
#     # This function should exist (rather than having the user import their df themself) only if we're using modin.pandas as pd instead of using pandas
#     file_extension = filepath.split('.')[-1]
#     if file_extension=="csv": # Technically, pd.read_csv() should also work with .txt files
#         return pd.read_csv(filepath)
#     else:
#         raise InputFormatException("Can only import a dataframe from a .csv")
#     # Could have requirements for filename formatting and get title from filename here, and be passing the text around in this whole program as a dict with title: content (str or df) or turn it into an object with member data title, etc as stated in another comment (go look at that one)


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
        # print(f'STEP 1. Saving DataFrame: {df_obj.__name__} to temporary VM file: {out_filename}\n') # Also, isinstance(obj, pd.DataFrame)
        print(f'STEP 1. Saving DataFrame to temporary VM file: {out_filename}\n')
        df_obj.to_csv(f"{save_filepath}{out_filename}", index=False) 
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
        
    print(stringToPeep) # TODO: remove all print statements (at very end of package development)
    return stringToPeep

# TODO: clean up & test or remove
def gutenberg_import(Novel_Title: str, Gutenberg_URL: str, 
                    sentence_first_str = None, sentence_last_str = None) -> str:
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
    sentences_ls = sent_tokenize(raw_text_str) # using nltk.tokenize

    # Most of the rest of this function (not the delete empty sentences part) is just returning things for user verification
    sentence_count = len(sentences_ls)
    num_senteces_to_show = 5

    verificationString = f'\n----- First {num_senteces_to_show} Sentences: -----\n\n'
    for i, asent in enumerate(sentences_ls[:num_senteces_to_show]):
        verificationString += f'Sentences #{i}: {asent}\n'

    print(f'\n----- Last {num_senteces_to_show} Sentences: -----\n')
    for i, asent in enumerate(sentences_ls[-num_senteces_to_show:]):
        verificationString += f'Sentences #{sentence_count - (num_senteces_to_show - i)}: {asent}\n'

    verificationString += f'\n\nThere are {sentence_count} Sentences in the text\n'

    # Delete the empty Sentences and those without any alphabetic characters
    sentences_ls = [x.strip() for x in sentences_ls if len(x.strip()) > 0]
    sentences_ls = [x.strip() for x in sentences_ls if re.search('[a-zA-Z]', x)]
    
    num_sentences_removed = sentence_count - len(sentences_ls)
    if (num_sentences_removed!=0):
        verificationString += f'\n\n{num_sentences_removed} empty and/or non-alphabetic sentences removed\n'
    # Q: How does sentence number & returning sentences around crux points still match up after doing this? Or do we not care exactly where the crux is in the original text? A: The "raw text" column in sentiment_df is the segmentedBySentences result

    # Plot distribution of sentence lengths
    # _ = plt.hist([len(x) for x in sentences_ls], bins=100)

    print(verificationString) # TODO: same deal as before: have a separate verification function that returns this? return this in a list along with the actual return value? just print it?
    
    return sentences_ls


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


def create_clean_df(sentences_ls: list, novel_title: str, save = False, save_filepath = CURRENT_DIR) -> pd.DataFrame:
    
    # Create sentiment_df to hold text sentences and corresponding sentiment values
    sentiment_df = pd.DataFrame
    sentiment_df = pd.DataFrame({'text_raw': sentences_ls})
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
    sentence_no_ls = list(range(sentiment_df.shape[0]))
    sentiment_df.insert(0, 'sentence_num', sentence_no_ls)

    # View the shortest lines by text_raw_len
    # print("shortest lines by text_raw_len: \n" + sentiment_df.sort_values(by='text_raw_len').head(20)) # only works if you're using pandas instead of modin.pandas

    if save:
        download_df(sentiment_df, novel_title, save_filepath=save_filepath, filename_suffix='_cleaned')
    
    return sentiment_df


def vader(sentiment_df: pd.DataFrame, novel_title: str) ->  pd.DataFrame:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_vader_ls = [sid_obj.polarity_scores(asentence)['compound'] for asentence in sentiment_df['cleaned_text'].to_list()]
    
    # Create new VADER DataFrame to save results
    vader_df = sentiment_df[['sentence_num', 'text_raw', 'cleaned_text']].copy(deep=True)
    vader_df['sentiment'] = pd.Series(sentiment_vader_ls) 
    vader_df.head()

    win_per = 0.1
    win_size = int(win_per * vader_df.shape[0])
    _ = vader_df['sentiment'].rolling(win_size, center=True).mean().plot(grid=True)

    # # Save Model Sentiment Time Series to file
    # novel_camel_str = ''.join([re.sub(r'[^\w\s]','',x).capitalize() for x in novel_title.split()])
    # download_df(vader_df, novel_camel_str, "vader")
    # note: just run the download_df() function with the returned value if you want to do this
    
    return vader_df
    # TODO: consider just appending these results to sentiment_df, and if someone wants the vader data only, they can subset that df. This woudl eliminate the need for combine_all_results or whatever in the main pipeline


def textblob(sentiment_df: pd.DataFrame, novel_title: str) -> pd.DataFrame:
    from textblob import TextBlob
    sentiment_textblob_ls = [TextBlob(asentence).polarity for asentence in sentiment_df['cleaned_text'].to_list()]
    # sentiment_df['textblob'] = sentiment_df['cleaned_text'].apply(lambda x : TextBlob(x).sentiment.polarity) # add textblob column to sentiment_df
    
    # Create new TextBlob DataFrame to save results
    textblob_df = sentiment_df[['sentence_num', 'text_raw', 'cleaned_text']].copy(deep=True)
    textblob_df['sentiment'] = pd.Series(sentiment_textblob_ls) 
    textblob_df.head()

    # Plot results
    window_pct = 0.1
    window_size = int(window_pct * textblob_df.shape[0])
    _ = textblob_df['sentiment'].rolling(window_size, center=True).mean().plot(grid=True)

    # # Save Model Sentiment Time Series to file
    # novel_camel_str = ''.join([re.sub(r'[^\w\s]','',x).capitalize() for x in novel_title.split()])
    # download_df(textblob_df, novel_camel_str, "textblob")
    #  # note: just run the download_df() function with the returned value if you want to do this
    
    return textblob_df


# Create class for data preparation for transformer models
class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts
    
    def __len__(self):
        return len(self.tokenized_texts["input_ids"])
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}
        
        
def distilbert(sentiment_df: pd.DataFrame, novel_title: str) -> pd.DataFrame:
    # Some of these might be needed in other transformer models to be added later (TODO)
    from transformers import pipeline
    from transformers import AutoTokenizer, AutoModelWithLMHead  # T5Base 50k
    from transformers import AutoModelForSequenceClassification, Trainer
    # from transformers import AutoModelForSeq2SeqLM, AutoModelWithLMHead
    from transformers import BertTokenizer, BertForSequenceClassification
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
    distilbert_df.head()

    # TODO: decide where to put this
    # Ensure balance of sentiments
    # distilbert_df['distilbert'].unique()
    _ = distilbert_df['label'].hist()

    # Q: is this different from the visualize function below?
    # TODO: decide if this is needed anywhere
    # Plot
    win_per = 0.1
    win_size = int(win_per * distilbert_df.shape[0])
    _ = distilbert_df['sentiment'].rolling(win_size, center=True).mean().plot(grid=True)

    # # Save results to file
    # novel_camel_str = ''.join([re.sub(r'[^\w\s]','',x).capitalize() for x in novel_title.split()])
    # download_df(distilbert_df, novel_camel_str, "distilbert")
    
    return distilbert_df


def combine_model_results(sentiment_df: pd.DataFrame, novel_title, **kwargs) -> pd.DataFrame:
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
            print(f'Success in appending {key} sentiments\n')
        except:
            print(f'Failed in appending {key} sentiments\n')

    # # Save Sentiment Timeseries to Datafile
    # download_df(all_sentiments_df, novel_title, "merged")
    
    return all_sentiments_df


# This function works on a df containing multiple models, and it creates a new df with the same column names but new sentiment values.
# TODO: Also create functions that allow the user to input a df with only one model's sentiment values and append adjusted & normalized sentiments as new columns on the same df, in case they want to compare different adjustments & smoothing methods for the same model.
def plot_sentiments(all_sentiments_df: pd.DataFrame, 
                        title: str, 
                        adjustments="normalizedAdjMean", # TODO: add a 'rescale' option, where all points are rescaled from their model's original scale to -1 to 1
                        smoothing="sma",
                        save_filepath=CURRENT_DIR, 
                        window_pct = 10,
                        models = ['vader', 'textblob', 'distilbert']) -> pd.DataFrame:
    """Saves a .png plot of raw or adjusted sentiments from the selected models.

    Saves a .png plot of raw, normed, or normed & adjusted sentiments 
    from the selected models to the specified directory. Sentiment curves
    are smoothed using the specified method before plotting. The function
    also returns the points from the plot in the form of a [TODO].
    
    @param models: must contain models with the same timesereies length 
    (sentimentR cannot be included without adjustments — used to be model_samelen_ls)

    Args:
        all_sentiments_df (pd.DataFrame): Dataframe containing sentiment 
            values in columns named after the models in @'models'
        title (str): Title of text
        adjustments (str): "none" (plot raw sentiments), "normalizedZeroMean" 
            (normalize to mean=0, sd=1), "normalizedAdjMean" (normalize and add
            the scaled mean that woudld be computed by adjusting the 
            original scores so their range is exactly[TODO: change depending on chun's answer] -1 to 1).
        smoothing (str): "sma" (simple moving average, aka sliding 
            window with window size determined by window_pct), "lowess"
            (LOWESS smoothing using parameter = [TODO])
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
    if window_pct>20 | window_pct<1:
        print("Warning: window percent outside expected range")
    window_size = int(window_pct/100 * all_sentiments_df.shape[0])

    if adjustments=="raw":
        # Plot Raw Timeseries    
        raw_rolling_mean = all_sentiments_df[models].rolling(window_size, center=True).mean() #Q: won't this have NA vals for the first few
        ax = raw_rolling_mean.plot(grid=True, lw=3)
        ax.title.set_text(f'Sentiment Analysis \n {title} \n Raw Sentiment Timeseries')
        plt.savefig(f"{save_filepath}{title}_raw_sentiments_plot.png")
        plt.show()
        
        return raw_rolling_mean

    else:
        # Compute the mean of each raw Sentiment Timeseries and adjust to [-1.0, 1.0] Range
        models_adj_mean_dt = {}
        if len(models)>1:
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
        all_sentiments_norm_df.head()

        if adjustments=="normalizedZeroMean":
            # Plot Normalized Timeseries to same mean (Q: Is this mean 0? If not, change filename below.)
            norm_rolling_mean = all_sentiments_norm_df[models].rolling(window_size, center=True).mean()
            ax = norm_rolling_mean.plot(grid=True, lw=3)
            ax.title.set_text(f'Sentiment Analysis \n {title} \n Normalization: Standard Scaler')
            # plt.show()
            plt.savefig(f"{save_filepath}{title}_normalized_0mean_sentiments_plot.png")

            return norm_rolling_mean

        else: # adjustments=="normalizedAdjMean"
            # Plot StandardScaler + Original Mean
            # Plot Normalized Timeseries to their adjusted/rescaled original means
            all_sentiments_adjnorm_df = all_sentiments_df[['sentence_num','text_raw','cleaned_text']].copy(deep=True)
            for amodel in models:
                all_sentiments_adjnorm_df[amodel] = all_sentiments_norm_df[amodel] + models_adj_mean_dt[amodel]

            norm_adj_rolling_mean = all_sentiments_adjnorm_df[models].rolling(window_size, center=True).mean()
            ax = norm_adj_rolling_mean.plot(grid=True, lw=3)
            ax.title.set_text(f'Sentiment Analysis \n {title} \n Normalization: Standard Scaler + True Mean Adjustment')
            plt.savefig(f"{save_filepath}{title}_normalized_adjusted_mean_sentiments_plot.png")
            plt.show()

            return norm_adj_rolling_mean
        
    #TODO: add lowess option
    # from statsmodels.nonparametric.smoothers_lowess import lowess
    # y=current_sentiment_arc_df[selected_model.value].values
    # x=np.arange(current_sentiment_arc_df.shape[0]) # i think this is just sentence num?
    # lowess(y, x, frac=1/30)[:,1].tolist()

def peakDetection(smoothed_sentiments_df: pd.DataFrame, 
                  model: str,
                  title: str,
                  save_filepath = CURRENT_DIR,
                  algo = "width",
                  distance_min = 360,
                  prominence_min = 0.05,
                  width_min = 25,
                  ):
    """[summary]
    
    Uses find_peaks() from scipy.signal (using the parameter specified
    by 'algo') to identify peaks and troughs in one model's sentiment
    plot. Returns 

    Args:
        model (str): 'vader', 'textblob', 'distilbert', 'nlptown', 'roberta15lg' TODO: add the extra models everywhere else or remove from here
        title (str): title of text
        save_filepath (str): path (ending in '/') to the directory
            the resulting plot png should be stored in.
            Defaults to the current working directory.
        algo (str): "distance", "prominence", or "width". Defauls to "width".
        distance_min (int, only needed if algo="distance"): 
            Required minimum number of sentences (>= 1) between 
            neighboring peaks. The smallest peaks are removed  until 
            the condition is fulfilled for all remaining peaks.
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

    _ = plt.plot(x)
    _ = plt.plot(peaks, x[peaks], "^g", markersize=15, label='peak sentence#') # green triangles
    _ = plt.plot(valleys, x[valleys], "vr", markersize=15, label='valley sentence#') #red triangles
    for x_val in peaks: # x_val = index of a peak in x
        _ = plt.text(x_val, x[x_val], f'    {x_val}', horizontalalignment='left', size='medium', color='black', weight='semibold')
    for x_val in valleys:
        _ = plt.text(x_val, x[x_val], f'    {x_val}', horizontalalignment='left', size='medium', color='black', weight='semibold')
    _ = plt.title(f'{title} \n {algo}-based peak detection ({len(peaks)+len(valleys)} cruxes) \n {len(peaks)} Peaks & {len(valleys)} Valleys', fontsize=16)
    _ = plt.ylabel('Sentiment')
    _ = plt.xlabel('Sentence No.')
    _ = plt.legend(loc='best')
    _ = plt.grid(True, alpha=0.3)

    plt.show()
    plt.savefig(f"{save_filepath}{title}_{algo}_cruxes_plot.png")
    
    return peaks, x[peaks], valleys, x[valleys]


def crux_context(sentiment_df: pd.DataFrame, peaks: list, valleys: list, n=10):
    """Return sentences around each sentiment crux

    Args:
        sentiment_df (pd.DataFrame): dataframe with original raw text
        peaks (list): indices (sentence nums) of peaks
        valleys (list): indices (sentence nums) of troughs
        n (int, optional): Number of sentences around 
            each crux point to display. Defaults to 10.
            
    Returns:
        crux_context: a string displaying the n sentences around each peak or valley
    """
    
    
    halfwin = int(n/2)
    newline = '\n'

    crux_context = '=================================================='
    crux_context += '============     Peak Crux Points   =============='
    crux_context += '==================================================\n\n'

    for i, peak in enumerate(peaks): # Iterate through all peaks
        crux_sents_ls = []
        for sent_idx in range(peak-halfwin,peak+halfwin+1):
            sent_cur = sentiment_df.iloc[sent_idx].text_raw
            if sent_idx == peak: # If current sentence is the one at 
                                 # which the peak was identified, 
                                 # print it in all caps
                sent_str = sent_cur.upper()
            else:                # Otherwise, print the original sentence
                sent_str = sent_cur
            crux_sents_ls.append(sent_str)
    
        # context_ls = sentiment_df.iloc[apeak-halfwin:apeak+halfwin].text_raw
        crux_context += f"Peak #{i} at Sentence #{peak}:\n\n{newline.join(crux_sents_ls)}\n\n\n"

    crux_context += '=================================================='
    crux_context += '===========     Crux Valley Points    ============'
    crux_context += '==================================================\n\n'

    for i, valley in enumerate(valleys): # Iterate through all valleys
        crux_sents_ls = []
        for sent_idx in range(valley-halfwin,valley+halfwin+1):
            sent_cur = sentiment_df.iloc[sent_idx].text_raw
            if sent_idx == valley: # If current sentence is the one at 
                                    # which the valley was identified, 
                                    # print it in all caps
                sent_str = sent_cur.upper()
            else:                   # Otherwise, print the original sentence
                sent_str = sent_cur
            crux_sents_ls.append(sent_str)

        # context_ls = novel_df.iloc[avalley-halfwin:avalley+halfwin].text_raw
        crux_context += f"Valley #{i} at Sentence #{valley}:\n\n{newline.join(crux_sents_ls)}\n\n\n"


    return crux_context




