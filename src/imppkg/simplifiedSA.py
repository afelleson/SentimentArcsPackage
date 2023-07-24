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
# import string # defines constants such as ascii_letters (currently not
# used)

THIS_SOURCE_FILE_PATH = os.path.abspath(__file__)
THIS_SOURCE_FILE_DIRECTORY = os.path.dirname(THIS_SOURCE_FILE_PATH)

import numpy as np
# import modin.pandas as pd # Modin — uses multiple cores for operations on pandas dfs. I think this uses an old version of pandas, forcing a reinstall of pandas :( Also removes from functionality from df methods, including sort_values(). So no longer using this, for now.
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

# For gutenberg imports
import requests

# For segmenting by sentence
from pysbd.utils import PySBDFactory
import spacy
# Note: We are now using spacy/pysbd instead of nltk.
# nltk_download_dir = os.path.join(THIS_SOURCE_FILE_DIRECTORY, 'my_nltk_dir')
# import nltk
# nltk.download('punkt', download_dir=nltk_download_dir)
# nltk.data.load(os.path.join(nltk_download_dir, 'tokenizers','punkt','english.pickle'))
#     # Note: To support other langauges, add more nltk.data.load() commands like the one above; just change 'english' to the new language name
# from nltk.tokenize import sent_tokenize

# For text cleaning
from cleantext import clean # TODO: remove dependency (clean-text library) if possible
import contractions # TODO: remove dependency?

# For timeseries normalizations
from sklearn.preprocessing import StandardScaler # TODO: figure out if necessary or if you can do this manually

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
PARA_SEP = config.get('imports', 'paragraph_separation').encode('unicode_escape') # :/ can't use cuz doesn't equal r'\n{2,} and wasn't able to convert. dont really need tho.
CURRENT_DIR = os.getcwd()
# TODO: consider making TITLE a global variable. I don't like that because the user has to set it. Other options are passing it to every function and making a SAobject that has title as a member datum.
ALL_MODELS_LIST = ['vader', 'textblob', 'distilbert', 'sentimentr',
                   ]
        # TODO: add nlptown and roberta15lg after figuring out how to
        # source more compute power.
        # Note: sentimentR lexicons not included right now =
        # emojis_sentiment, hash_sentiment_emojis, 
        # hash_sentiment_slangsd, and hash_sentiment_loughran_mcdonald
        # (the last one is for financial texts).
        # 'sentimentr' runs lexicons 'sentimentr_jockersrinker',
        # 'sentimentr_jockers', 'sentimentr_huliu','sentimentr_nrc',
        # 'sentimentr_senticnet', 'sentimentr_sentiword',
        # 'sentimentr_loughran_mcdonald', 'sentimentr_socal_google'

# Custom Exceptions
class InputFormatException(Exception):
    pass


# If using pandas as pd:
#   Code to import a csv as a pd.df that can be passed to the model functions: 
#       df = pd.read_csv('saved_file.csv')
# Elif using modin.pandas as pd:
# def import_df(filepath: str) -> pd.DataFrame: 
#     # This function should exist (rather than having the user import their 
#     # df themself) only if we're using modin.pandas as pd instead of using 
#     # pandas. Reason: we want 'pd' in the pd.read_csv function call to be
#     # modin so it creates a modin pandas df. If the user does it in their 
#     # own code with pandas imported as pd, it'll create a regular pandas df.
#     # (Not completely sure about this / if this is the best way to do it.)
#       
#     file_extension = filepath.split('.')[-1]
#     if file_extension=="csv": # Technically, pd.read_csv() should also work with .txt files
#         return pd.read_csv(filepath)
#     else:
#         raise InputFormatException("Can only import a dataframe from a .csv")
#     # Could have requirements for filename formatting and get title from filename here, and be passing the text around in this whole program as a dict with title: content (str or df) or turn it into an object with member data title, etc as stated in another comment (go look at that one)

def uniquify(path: str) -> str:
    """Generate a unique filename for a file to be saved.
    
    Append (1), or (2), etc, to a file name if the file already exists.

    Args:
        path (str): Complete path to the file, including the extension

    Returns:
        path (str): Edited complete path to the file, including the 
            extension
    """
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path

def download_df(df_obj: pd.DataFrame, title: str, 
                save_filepath=CURRENT_DIR, 
                filename_suffix='_df', 
                nodate=True):
    """Write DataFrame object to csv file.

    Save DataFrame as a csv file named after the text title + the given 
    suffix in the provided directory. If a file with the same name
    already exists there, a number in parentheses is appended to the 
    file name.
    
    Args:
        df_obj (pd.DataFrame): DataFrame to save
        title (str): Text title
        save_filepath (str, optional): Path to folder in which to save
            the csv. Defaults to the current working directory.
        filename_suffix (str, optional): Text to append to the file 
            name, after the text title. Defaults to '_df'.
        nodate (bool, optional): Whether or not to append the date and 
            time to the file name. Defaults to True.

    Returns:
        err_code (int): Non-zero value indicateing error code, or zero 
            on success.
        err_msg (str or None): Human readable error message, or None on 
            success.
    """
    
    
    '''
    INPUT: DataFrame object and suffix to add to output csv filename
    '''
    # TODO: get name of df_obj and use that in the file name
    camel_title = ''.join([re.sub(r'[^\w\s]','',x).capitalize() for x in title.split()])
    if isinstance(df_obj, pd.DataFrame):
        if nodate:
            out_filename = camel_title.split('.')[0] + filename_suffix + ".csv"
        else:
            datetime_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            out_filename = camel_title.split('.')[0] + '_' + filename_suffix + datetime_str + ".csv"

        completepath = os.path.join(save_filepath,out_filename)
        df_obj.to_csv(uniquify(completepath), index=False)  # TODO: test
    else:
        raise TypeError('Expected pandas DataFrame as first argument; got ' + str(type(df_obj).__name__))


def upload_text(filepath: str) -> str:
    """Upload text from a raw text file into a string

    Args:
        filepath (str): Filepath to txt file to upload

    Raises:
        InputFormatException: If the file is not a .txt

    Returns:
        str: The imported raw text
    """

    filename_ext_str = filepath.split('.')[-1]
    if filename_ext_str == 'txt':
        # raw_text_str = uploaded[novel_filename_str].decode(TEXT_ENCODING)
        with open(filepath,'r') as f:
            raw_text_str = f.read()
            # TODO: figure out if decoding (using TEXT_ENCODING) would be useful here
            return raw_text_str
    else:
        raise InputFormatException("Must provide path to a plain text file (*.txt)")
        # TODO: Decide whether to use exceptions or error codes & be consistent

     # return as single-item dict with title as key instead? or a custom "SAtext" object with data members title & body?

def preview(something) -> str: # would make more sense as a method imo. could take in an SAtext object and be a method, or take in a dict to be able to print the file name
    """Produce a string showing the beginning and end of the text
    
    For use on a newly-created raw text string or cleaned text DataFrame
    for user verification.

    Args:
        something (str or pd.DataFrame): The text to be previewed

    Returns:
        stringToPeep (str): Annotated copy of the first and last few 
            sentences of the text
    """
    
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
    print(stringToPeep) # TODO: remove all print statements (at end of package development)
    return stringToPeep

# TODO: clean up & test or remove
# def gutenberg_import(title: str, Gutenberg_URL: str, 
#                     sentence_first_str = None, sentence_last_str = None) -> str:
#     #@title Enter the URL of your novel at ***gutenberg.net.au***
#     #@markdown Paste the URL to the ***HTML version*** (not plain text).

#     # title = 'Frankenstein by Mary Shelley'  #@param {type: "string"}

#     # Gutenberg_URL = 'https://gutenberg.net.au/ebooks/z00006.html'  #@param {type: "string"}
#     # the first sentence in the body of your novel: sentence_first_str
#     # the last sentence in the body of your novel: sentence_last_str


#     # Get raw HTML of novel from Gutenberg.net.au
#     response=requests.get(Gutenberg_URL)  # TODO: Pass the URL to the .get() method of the requests object
#     html = response.text

#     # Use HTML <p> tags to extract text into list of paragraphs
#     paragraphs = re.findall(r'<p>(.*?)</p>', html, flags=re.DOTALL)

#     if (len(paragraphs) < 3):
#         raise InputFormatException("Fewer than three paragraphs detected")

#     # TODO: figure out what this is doing and why (seems like it's just undoing what we did, plus the \r\n replacement)
#     # Concatenate all paragraphs into a single novel string
#     # For every paragraph, replace each hardcoded \r\n with a single space
#     paragraphs_flat = [re.sub(r'\r\n', ' ', paragraph) for paragraph in paragraphs]
#     # Concatenate all paragraphs into single strings separated by two \n
#     raw_text_str = '\n\n'.join(paragraphs_flat)
    
#     if (sentence_first_str is not None  and  sentence_last_str is not None): # using optional function args
#         # Remove header
#         raw_text_str = ' '.join(raw_text_str.partition(sentence_first_str)[1:])
#         # Remove footer
#         raw_text_str = ' '.join(raw_text_str.partition(sentence_last_str)[:2])
    
#     return(raw_text_str)

def replace_ellipses(raw_text_str:  str) -> str:
    ellipses_replaced_text = re.sub(r'\. \. \.|\.\.\.|\…', ' <ELLIPSIS> ', raw_text_str)
    # change mid-sentence ellipses to commas 
    
    
    # and change end-of-sentence ellipses (including ones before a line break!) to periods
    # How to deal with mid sentence ellipsis before a proper noun? I was thinking it’d be good enough to replace them with a period.
    # If there’s only one word before the ellipsis, make it a comma. Otherwise, a period is fine.
    # If sentence or quote begins with ellipsis, just remove the ellipsis.


def segment_sentences(raw_text_str:  str) -> list: # TODO: don't print/have a verification string if there aren't parameters to adjust here
    # Segment by sentence
    # sentences_list = sent_tokenize(raw_text_str) # Previous method,
    # using only nltk.tokenize for all sentence tokenization. Can
    # delete.
     
    # Add custom rules for spacy
    from spacy.language import Language

    def beginning_of_quote_component_func(doc):
        # Fixes issue where a beginning quotation mark was being treated as a separate sentence from the rest of the quote
        for i, token in enumerate(doc[:-1]):
            openingQuotes = ['“','"']
            if token.text in openingQuotes and doc[i + 1].is_alpha and doc[i + 1].text[0].isupper:
                doc[i + 1].is_sent_start = False
                doc[i].is_sent_start = True
        return doc

    @Language.component("beginning_of_quote_component")
    def beginning_of_quote_component(doc):
        return beginning_of_quote_component_func(doc)
    
    ellipses_replaced_text = re.sub(r'\. \. \.|\.\.\.|\…', ' <ELLIPSIS> ', raw_text_str)
    
    def ellipsis_lowercase_component_func(doc):
        # Marks ellipsis followed by a lowercase letter as not a sentence break
        for i, token in enumerate(doc[:-2]):
            print(str(token))
            if token.text == '<ELLIPSIS>':
                if doc[i + 1].is_space and doc[i + 2].text.is_lower:
                    print("YES\n")
                    doc[i].is_sent_start = False
                    doc[i + 1].is_sent_start = False
                    doc[i + 2].is_sent_start = False
                elif doc[i + 1].is_space and doc[i + 2].text[0].isupper:
                    doc[i].is_sent_start = False
                    doc[i + 1].is_sent_start = True # ?
                    doc[i + 2].is_sent_start = False # ?
                # idk if i should specify every other case. we've got
                # quotation marks, question marks, and all options
                # without a space between the ellipsis and the next thing.
        return doc

    @Language.component("ellipsis_lowercase_component")
    def ellipsis_lowercase_component(doc):
        return ellipsis_lowercase_component_func(doc)

    # Create spacy pipes
    nlp = spacy.blank('en')
    nlp.add_pipe('sentencizer')
    nlp.add_pipe("beginning_of_quote_component") # custom rule
    nlp.add_pipe("ellipsis_lowercase_component") # custom rule
    
    parags_ls = re.split(r'\n{2,}', ellipses_replaced_text) # Split text into paragraphs
    parags_ls = [x.strip() for x in parags_ls]
    
    sentences_list = []
    for para in parags_ls:
        para_no_newlines = re.sub('[\n]{1,}', ' ', para)
        
        # Round 1: PySBD
        doc = nlp(para_no_newlines)
        para_sents_pysbd_list = list(doc.sents)
        para_sents_pysbd_list = [str(x) for x in para_sents_pysbd_list] # temporary replacement for following line
        # para_sents_pysbd_list = [str(x).strip() for x in para_sents_pysbd_list]  # Strip leading/trailing whitespace
        
        # # Round 2: NLTK
            # On The Hunger Games, this only separates out opening
            # quotation marks and ellipses as their own sentences, which
            # is incorrect. End result is fine, but nltk is not adding
            # any beneficial functionality.
        # para_sents_nltk_list = [sent_tokenize(sent) for sent in para_sents_pysbd_list]
        # import itertools
        # para_sents_nltk_list = list(itertools.chain.from_iterable(para_sents_nltk_list))  # Flatten the list
        # # para_sents_nltk_list = sent_tokenize(para_no_newlines) # replacement for previous 3 lines if not using pysbd
        # para_sents_nltk_list = [str(x).strip() for x in para_sents_nltk_list] # Strip leading/trailing whitespace

        para_sents_list = [x for x in para_sents_pysbd_list if (len(x) > 1)] # Filter out empty and 1-character sentences

        sentences_list.extend(para_sents_list)
        

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
    sentences_list = [x for x in sentences_list if re.search('[a-zA-Z]', x)]
    
    num_sentences_removed = sentence_count - len(sentences_list)
    if (num_sentences_removed!=0):
        verificationString += f'\n\n{num_sentences_removed} empty and/or non-alphabetic sentences removed\n'
    # Q: How does sentence number & returning sentences around crux points still match up after doing this? Or do we not care exactly where the crux is in the original text? 
    # A: The "raw text" column in sentiment_df is the segmentedBySentences result

    print(verificationString) # TODO: same deal as before: have a separate verification function that returns this? return this in a list along with the actual return value? just print it?
    
    return sentences_list


def clean_string(dirty_str: str) -> str:
    #TODO: add options, and add more functions in here that take care of stuff clean-text doesn't, like emoticons
    """Clean a string

    Args:
        dirty_str (str): A raw string

    Returns:
        str: A cleaned string
    """

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
        lang="en"
    )

    # Replace all new lines/returns with single whitespace
    # clean_str = clean_str.replace('\n\r', ' ') # I think these are commented out bc clean() with no_line_breaks=True already does those
    # clean_str = clean_str.replace('\n', ' ')
    # clean_str = clean_str.replace('\r', ' ')
    clean_str = ' '.join(clean_str.split()) # remove leading, trailing, and *repeated* spaces

    return clean_str 


def create_clean_df(sentences_list: list[str], title: str, save = False, save_filepath = CURRENT_DIR) -> pd.DataFrame:
    """Create DataFrame of raw and cleaned sentences

    From a list of sentences, create a DataFrame with columns 
    'text_raw', 'cleaned_text', and 'cleaned_text_len'.
    
    Args:
        sentences_list (list of strings): A list of sentences.
        title (str): The text title
        save (bool, optional): Whether to save the DataFrame (as a csv). 
            Defaults to False.
        save_filepath (str, optional): Where to save the DataFrame. 
            Defaults to current working directory.

    Returns:
        pd.DataFrame: A DataFrame with columns 'sentence_num', 
            'text_raw', 'cleaned_text', and 'cleaned_text_len', intended to be
            passed to a sentiment analysis model function
    """
    
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
    """Turn raw text string into clean text DataFrame ready for analysis

    Args:
        raw_text_str (str): The raw text
        title (str): The text title
        save (bool, optional): Whether to save the DataFrame. Defaults 
            to False.
        save_filepath (str, optional): Where to save the DataFrame. 
            Defaults to CURRENT_DIR.

    Returns:
        pd.DataFrame: A DataFrame with columns 'sentence_num', 
            'text_raw', 'cleaned_text', and 'cleaned_text_len', intended to be
            passed to a sentiment analysis model function
    """
    sentences_list = segment_sentences(raw_text_str)
    clean_df = create_clean_df(sentences_list, title, save, save_filepath)
    return clean_df


def vader(sentiment_df: pd.DataFrame, title: str) ->  pd.DataFrame:
    # TODO: remove title from each of these models' params. Not doing
    # now bc not backwards compatible (Dev's code will break)
    print("vader")
    """ Run the vader sentiment analysis model.

    Run vader on the cleaned_text column of the passed DataFrame, and 
        create a new DataFrame with an appended 'sentiment' column.
        
    Args:
        sentiment_df (pd.DataFrame): A DataFrame with 'sentence_num', 
            'text_raw', and 'text_cleaned' columns.

    Returns:
        pd.DataFrame: 
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_vader_ls = [sid_obj.polarity_scores(asentence)['compound'] for asentence in sentiment_df['cleaned_text'].to_list()]
    
    # Create new VADER DataFrame to save results
    vader_df = sentiment_df[['sentence_num', 'text_raw', 'cleaned_text']].copy(deep=True)
    vader_df['sentiment'] = pd.Series(sentiment_vader_ls) 
        
    return vader_df
    # TODO: consider just appending these results to sentiment_df, and if someone wants the vader data only, they can subset that df. This would eliminate the need for combine_all_results or whatever in the main pipeline


def textblob(sentiment_df: pd.DataFrame, title: str) -> pd.DataFrame:
    print("textblob")
    from textblob import TextBlob
    sentiment_textblob_ls = [TextBlob(asentence).polarity for asentence in sentiment_df['cleaned_text'].to_list()]
        # From TextBlob docs: The sentiment property returns a named 
        # tuple of the form Sentiment(polarity, subjectivity). 
        # The polarity score is a float within the range [-1.0, 1.0]. 
        # The subjectivity is a float within  the range [0.0, 1.0], 
        # where 0.0 is very objective and 1.0 is very subjective.
    # sentiment_df['sentiment'] = sentiment_df['cleaned_text'].apply(lambda x : TextBlob(x).sentiment.polarity) # add textblob column to sentiment_df
    
    # change this to be sentiment instead of polarity for the new method
    sentiment_textblob_ls = [TextBlob(asentence).sentiment for asentence in sentiment_df['cleaned_text'].to_list()]
    # Create new TextBlob DataFrame to save results
    textblob_df = sentiment_df[['sentence_num', 'text_raw', 'cleaned_text']].copy(deep=True)
    # textblob_df['sentiment'] = pd.Series(sentiment_textblob_ls) #old
    # new
    textblob_df['sentiment'] = pd.Series([x.polarity for x in sentiment_textblob_ls]) 
    textblob_df['subjectivity'] = pd.Series([x.subjectivity for x in sentiment_textblob_ls]) 
    
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
    model_name = "distilbert-base-uncased-finetuned-sst-2-english" # Note: use a cased model for German
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
        # TODO: ask what these mean! the 'sentiment' column is just 0s and 1s; is that what we want?

    # TODO: decide where to put this
    # Ensure balance of sentiments
    # distilbert_df['distilbert'].unique()
    # _ = distilbert_df['label'].hist()

    return distilbert_df

def sentimentr(sentiment_df: pd.DataFrame, title: str):
    # 'sentimentr_jockersrinker','sentimentr_jockers',
    #    'sentimentr_huliu','sentimentr_nrc','sentimentr_senticnet',
    #    'sentimentr_sentiword','sentimentr_loughran_mcdonald',
    #    'sentimentr_socal_google'
    
    import rpy2.robjects as robjects
    r = robjects.r
    r_file_path = os.path.join(THIS_SOURCE_FILE_DIRECTORY, 'run_sentimentr.R')
    r['source'](r_file_path)
    get_sentimentr_rfunction = robjects.globalenv['get_sentimentr_values'] # note to self: if this doesn't work, you'll have to pass the restore function to the main one
    
    sentences_vec = robjects.StrVector(sentiment_df['cleaned_text'].to_list()) # Convert Python List of Strings to a R character vector
    sentimentr_rdf = get_sentimentr_rfunction(sentences_vec)

    # Convert rpy2.robjects.vectors.DataFrame to pandas.core.frame.DataFrame
    sentimentr_df = pd.DataFrame.from_dict({ key : np.asarray(sentimentr_rdf.rx2(key)) for key in sentimentr_rdf.names })
    
    sentimentr_df = pd.concat([sentiment_df[['sentence_num', 'text_raw', 'cleaned_text']].copy(deep=True), sentimentr_df.iloc[:,1:]], axis=1) # sentimentr_df = pd.concat([sentiment_df[['sentence_num', 'text_raw', 'cleaned_text']].copy(deep=True), sentimentr_df.loc[:, sentimentr_df.columns!='text_clean']], axis=1)

    return sentimentr_df

# TODO: get rid of this function? (I think Dev's code relies on it,
# currently.) Should we keep it in case people decide they want to run &
# compare more models later?
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
    """Run sentiment analysis model(s) on a given cleaned text DataFrame

    Args:
        sentiment_df (pd.DataFrame): A DataFrame with columns 
            'sentence_num', 'text_raw', and 'cleaned_text', where each
            row is a string (e.g., a sentence) to assign a sentiment to
        title (str): The text title
        models (list of strings, optional): A list of the sentiment
            analysis models to be run, with lowercase titles. Defaults 
            to ['vader', 'textblob', 'distilbert','sentimentr'].

    Returns:
        pd.DataFrame: sentiment_df with appended columns named after
            each model, containing the sentiment score assigned to each
            string by the model
    """
    all_sentiments_df = sentiment_df[['sentence_num','text_raw','cleaned_text']].copy(deep=True)
    if "vader" in models:
        all_sentiments_df['vader'] = vader(sentiment_df,title)['sentiment']
    if "textblob" in models:
        all_sentiments_df['textblob'] = textblob(sentiment_df,title)['sentiment']
    if "distilbert" in models:
        all_sentiments_df['distilbert'] = distilbert(sentiment_df,title)['sentiment']
    if "sentimentr" in models:
        all_sentiments_df = pd.concat([all_sentiments_df, sentimentr(sentiment_df,title).iloc[:, 5:].copy(deep=True)], axis=1)
    for user_model in models:
        if user_model not in ALL_MODELS_LIST:
            print(f"Warning: {user_model} model not found in list of accepted models. Check your spelling.")
    return all_sentiments_df

# This function works on a df containing multiple models, and it creates a new df with the same column names but new sentiment values.
# TODO: Also create functions that allow the user to input a df with only one model's sentiment values and append adjusted & normalized sentiments as new columns on the same df, in case they want to compare different adjustments & smoothing methods for the same model.
def plot_sentiments(all_sentiments_df: pd.DataFrame, 
                        title: str, 
                        models = ALL_MODELS_LIST,
                        adjustments="normalizedZeroMean", # TODO: add a 'rescale' option, where all points are rescaled from their model's original scale to -1 to 1
                        smoothing="sma",
                        plot = "save",
                        save_filepath=CURRENT_DIR, 
                        window_pct = 10,
                        ) -> pd.DataFrame:
    """Save a .png plot of raw or adjusted sentiments from the selected models.

    Save a .png plot of raw, normed, or normed & adjusted sentiments 
    from the selected models to the specified directory. Smooth 
    sentiment curves using the specified method before plotting. Also 
    return the points from the plot in the form of a [TODO].

    Args:
        all_sentiments_df (pd.DataFrame): Dataframe containing sentiment 
            values in columns named after the models in `models`
        title (str): Title of text
        models (list of strings): A list of the sentiment
            analysis models to be run, with lowercase titles. Must 
            contain models with the same timesereies length 
            (sentimentR cannot be included without adjustments). 
            Defaults to ['vader', 'textblob', 'distilbert','sentimentr']. 
        adjustments (str): "none" (plot raw sentiments), "normalizedZeroMean" 
            (normalize to mean=0, sd=1), "normalizedAdjMean" (normalize and add
            the scaled mean that woudld be computed by adjusting the 
            original scores so their range is exactly -1 to 1). Defaults
            to normalizedZeroMean.
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
        print("Warning: window percentage outside expected range (1-20%)")
    window_size = int(window_pct / 100 * all_sentiments_df.shape[0])

    camel_title = ''.join([re.sub(r'[^\w\s]','',x).capitalize() for x in title.split()])
    
    if adjustments == "raw":
        # Plot Raw Timeseries
        raw_rolling_mean = all_sentiments_df[['sentence_num','text_raw','cleaned_text']].copy(deep=True)
        raw_rolling_mean[models] = all_sentiments_df[models].rolling(window_size, center=True).mean()
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
        model (str): 'vader', 'textblob', 'distilbert', or 'sentimentr'
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

