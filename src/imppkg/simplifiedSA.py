"""
simplifiedSA.py

[TODO: description]

Version 1: Alex Felleson, August 2023
Based on an interactive python notebook by Jon Chun, Feb. 2023: https://github.com/jon-chun/sentimentarcs_simplified
"""

import configparser
import datetime
import re
import os

import pandas as pd
import spacy
from cleantext import clean
import contractions
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Note: Additional library imports are included within lesser-used 
# functions to improve efficiency.

THIS_SOURCE_FILE_PATH = os.path.abspath(__file__)
THIS_SOURCE_FILE_DIRECTORY = os.path.dirname(THIS_SOURCE_FILE_PATH)

# Read the toml file that contains settings that advanced users can edit (via ???)
# (Not currently using either of the imported global variables below)
# AUGTODO: get rid of this stuff in final version, but keep here
config = configparser.ConfigParser()
def get_filepath_in_src_dir(filename: str) -> str:
    result = os.path.join(THIS_SOURCE_FILE_DIRECTORY, filename)
    if not os.path.exists(result):
        raise Exception("Failed to create filepath in current directory")
    else:
        return result
config.read(get_filepath_in_src_dir('SA_settings.toml'))
TEXT_ENCODING = config.get('imports', 'text_encoding') # TODO: Use the chardet library to detect the file's character encoding instead, probably
PARA_SEP = config.get('imports', 'paragraph_separation').encode('unicode_escape') # Can't use because this isn't equivalent to r'\n{2,}' :(

# Set up matplotlib
plt.rcParams["figure.figsize"] = (20,10)
plt.rcParams["font.size"] = 22
# TODO: add more setup commands from old util file?

# Global variables
CURRENT_DIR = os.getcwd()
ALL_MODELS_LIST = ['vader', 
                   'textblob', 
                   'distilbert', 
                   'sentimentr',
                   ]
        # TODO: add nlptown and roberta15lg (from simplified notebook) 
        # after figuring out how to source more compute power.
# TODO: consider making TITLE a global variable. I don't like that because the user has to set it. Other options are passing it to every function or making a SAobject that has title as a member datum.


### FUNCTIONS ###

# If using modin.pandas as pd:
# Code to import a csv as a pd.df that can be passed to the model functions: 
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
#         raise ValueError("Can only import a dataframe from a .csv")
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
                save_filepath = CURRENT_DIR, 
                filename_suffix='_df', 
                date = False,
                ):
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
        date (bool, optional): Whether or not to append the date and 
            time to the file name. Defaults to False.

    Returns:
        err_code (int): Non-zero value indicateing error code, or zero 
            on success.
        err_msg (str or None): Human readable error message, or None on 
            success.
    """

    camel_title = ''.join([re.sub(r'[^\w\s]', '', x).capitalize() 
                           for x in title.split()])
    if isinstance(df_obj, pd.DataFrame):
        if not date:
            out_filename = camel_title.split('.')[0] + filename_suffix + ".csv"
        else:
            datetime_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            out_filename = camel_title.split('.')[0] + '_' + filename_suffix \
                + datetime_str + ".csv"

        completepath = os.path.join(save_filepath, out_filename)
        df_obj.to_csv(uniquify(completepath), index=False)
    else:
        raise TypeError('Expected pandas DataFrame as first argument; got ' + \
            str(type(df_obj).__name__))


def upload_text(filepath: str) -> str:
    """Upload text from a raw text file into a string.

    Args:
        filepath (str): Filepath to txt file to upload

    Raises:
        ValueError: If the file is not a .txt file

    Returns:
        str: The imported raw text
    """

    filename_ext_str = filepath.split('.')[-1]
    if filename_ext_str == 'txt':
        with open(filepath, 'r') as f:
            raw_text_str = f.read()
            # TODO: figure out if decoding (using TEXT_ENCODING) would
            # be useful here. Probably use the chardet library to detect 
            # the file's character encoding instead, if needed.
            return raw_text_str
    else:
        raise ValueError("Must provide path to a plain text file (.txt)")

     # return as single-item dict with title as key instead? or a custom "SAtext" object with data members title & body?

# TODO: delete? AUGTODO
def preview(something) -> str: # would make more sense as a method imo. could take in an SAtext object and be a method, or take in a dict to be able to print the file name
    """Print the beginning and end of a string or dataframe.
    
    For user verification of a newly-created raw text string or cleaned 
    text DataFrame.

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
            stringToView =     (f'  Length of text (in characters): {len(something)}\n' +
                                f'Entire text: \n' +
                                something)
        stringToView =     (f'  Length of text (in characters): {len(something)}\n' +
                '====================================\n\n' +
                f'Beginning of text:\n\n {something[:500]}\n\n\n' +
                '\n------------------------------------\n' +
                f'End of text:\n\n {something[-500:]}\n\n\n')
    elif type(something) == pd.DataFrame:
        if len(something)<20: # TODO: test this comparison value is exactly correct
            stringToView = ("Short dataframe. Here's the whole thing: \n" +
                            '\n'.join(something['cleaned_text'][0:9].astype(str)))
        stringToView = "First 10 sentences:\n\n"
        stringToView += '\n'.join(something['cleaned_text'][0:9].astype(str))
        stringToView += "\n\n ... \n\n"
        stringToView += "Last 10 sentences:\n\n"
        stringToView += '\n'.join(something['cleaned_text'][-11:-1].astype(str))
    else:
        TypeError("You may only preview a string or dataframe with a cleaned_text column")
        stringToView = ""
        
    stringToView += "\n\n"
    print(stringToView) # TODO: remove all print statements (at end of package development)
    return stringToView

def gutenberg_import(gutenberg_url: str, 
                     sentence_first_str = None, 
                     sentence_last_str = None,
                    ) -> str:
    """Import a raw text novel from Gutenberg.net.au.
    
    Extracts a novel from Gutenberg.net.au, reformats it into a string
    suited for input into other functions within this package,
    and provides options to remove header and footer content.

    Args:
        gutenberg_url (str): The full URL of the novel on Gutenberg.net.au.
        sentence_first_str (str, optional): A string that marks the 
            beginning of the relevant content to be extracted. Defaults 
            to None.
        sentence_last_str (str, optional): A string that marks the end
            of the relevant content to be extracted. Defaults to None.

    Raises:
        ValueError: Raised if fewer than three paragraphs are detected 
            in the novel.

    Returns:
        str: The reformatted raw text content of the novel, potentially 
            with header and footer removed.
    """
    
    import requests

    # Get raw HTML of novel from Gutenberg.net.au
    response = requests.get(gutenberg_url)
    html = response.text

    # Use HTML <p> tags to extract text into list of paragraphs
    paragraphs = re.findall(r'<p>(.*?)</p>', html, flags=re.DOTALL)

    if (len(paragraphs) < 20):
        raise ValueError("Fewer than 20 paragraphs detected. Check the input URL. "
                         "If the issue appears to be with the HTML formatting on the Gutenberg site, "
                         "you must format and import the text without using this "
                         "function (gutenberg_import()).")

    # Within each paragraph, replace each hardcoded \r and \n with a 
    # single space
    paragraphs_flat = [re.sub(r'\r', ' ', paragraph) 
                       for paragraph in paragraphs]
    paragraphs_flat = [re.sub(r'\n', ' ', paragraph) 
                       for paragraph in paragraphs_flat]
    # Remove leading, trailing, and *repeated* spaces
    paragraphs_flat = [' '.join(paragraph.split()) 
                       for paragraph in paragraphs_flat]
   
    # Join all paragraphs into a single string. Separate paragraphs with
    # two newline characters. (Expected raw text format for other
    # functions in this package.)
    raw_text_str = '\n\n'.join(paragraphs_flat)
    
    # Use optional function args
    # Remove header
    if sentence_first_str is not None:
        raw_text_str = ' '.join(raw_text_str.partition(sentence_first_str)[1:])
    # Remove footer
    if sentence_last_str is not None:
        raw_text_str = ' '.join(raw_text_str.partition(sentence_last_str)[:2])
    
    return(raw_text_str)


def segment_sentences(raw_text_str:  str, para_sep=r'\n{2,}') -> list:
    """Segment raw text into a list of sentences.
    
    Use the given (or default) paragraph separator to separate a string
    into paragraphs and then sentences. Includes some string cleaning 
    steps.

    Args:
        raw_text_str (str): The input raw text to be segmented into 
            sentences.
        para_sep (str, optional): A regular expression specifying the 
            paragraph separator. Defaults to r'\n{2,}'.

    Raises:
        ValueError: Raised if fewer than 5 paragraphs are detected.
            Indicates raw_text_str formatting that does not match the
            para_sep argument's value.

    Returns:
        List[str]: A list of segmented sentences from the input raw 
        text.
    """
    
    ellipses_replaced_text = re.sub(r'\. \. \.|\.\.\.|\…', 
                                    ' /ELLIPSIS/ ', 
                                    raw_text_str)
    
    # Split text into paragraphs
    parags_ls = re.split(para_sep, ellipses_replaced_text)
    
    if len(parags_ls) < 5:
        raise ValueError("Fewer than 5 paragraphs detected. Are paragraphs in "
                         "the raw text string separated by " + para_sep + "? "
                         "If not, specify the para_sep argument.")
    
    # Miscellaneous cleanup
    parags_ls = [x.replace('”“', '” “') for x in parags_ls]
    
    # Clean up miscellaneous line breaks and spaces
    # Remove single line breaks
    parags_ls = [x.replace('\n', ' ') for x in parags_ls]
    # Remove single line breaks
    parags_ls = [x.replace('\r', ' ') for x in parags_ls]
    # Remove leading, trailing, and *repeated* spaces
    parags_ls = [' '.join(x.split()) for x in parags_ls]
     
    # Add custom rules for spacy
    from spacy.language import Language

    def beginning_of_quote_component_func(doc):
        # Fixes issue where a beginning quotation mark was being treated 
        # as a separate sentence from the rest of the quote
        for i, token in enumerate(doc[:-1]):
            openingQuotes = ['“', '"']
            if (token.text in openingQuotes and 
                    doc[i + 1].is_alpha and 
                    doc[i + 1].text[0].isupper):
                doc[i + 1].is_sent_start = False
                doc[i].is_sent_start = True
        return doc

    @Language.component("beginning_of_quote_component")
    def beginning_of_quote_component(doc):
        return beginning_of_quote_component_func(doc)
    
    def sentence_ending_in_I_component_func(doc):
        # Fixes issue where "I." was not considered a sentence end
        # because I is a single letter.
        for i, token in enumerate(doc[:-1]):
            if token.text == "I.":
                doc[i + 1].is_sent_start = True
        return doc

    @Language.component("sentence_ending_in_I_component")
    def sentence_ending_in_I_component(doc):
        return sentence_ending_in_I_component_func(doc)
    
    # Create spacy sentence separation pipes
    nlp = spacy.blank('en')
    nlp.add_pipe('sentencizer')
    nlp.add_pipe("beginning_of_quote_component") # custom rule
    nlp.add_pipe("sentence_ending_in_I_component") # custom rule
    
    # from pysbd.utils import PySBDFactory
    # nlp.add_pipe(PySBDFactory(nlp)) 
        # If you're going to use this, you need to 
        # require spacy>=2.0.0,<3.0.0 in setup.cfg and change the other 
        # add_pipe() syntaxes back to how they were in the OG 
        # SentimentArcs code so they're compatible with those version of 
        # spaCy.
    
    sentences_list = []
    for para in parags_ls:
        
        # Round 1: spaCy
        doc = nlp(para)
        para_sents_spacy_list = list(doc.sents)
        # Strip leading/trailing whitespace
        para_sents_spacy_list = [str(x).strip() 
                                 for x in para_sents_spacy_list]  
        
        # # Round 2: NLTK
            # On The Hunger Games, this only separates out opening
            # quotation marks and ellipses as their own sentences, which
            # is incorrect. The end result is fine, but nltk is not 
            # adding any beneficial functionality. So we are now using 
            # spaCy only instead of nltk.
        # # Set up NLTK
        # import nltk
        # nltk_download_dir = os.path.join(THIS_SOURCE_FILE_DIRECTORY, 'my_nltk_dir')
        # nltk.download('punkt', download_dir=nltk_download_dir)
        # nltk.data.load(os.path.join(nltk_download_dir, 'tokenizers', 'punkt', 'english.pickle'))
        #     # Note: To support other langauges, add more nltk.data.load() commands like the one above; just change 'english' to the new language name
        # from nltk.tokenize import sent_tokenize

        # para_sents_nltk_list = [sent_tokenize(sent) for sent in para_sents_spacy_list]
        # import itertools
        # para_sents_nltk_list = list(itertools.chain.from_iterable(para_sents_nltk_list))  # Flatten the list
        # # para_sents_nltk_list = sent_tokenize(para) # replacement for previous 3 lines if not using spaCy
        # para_sents_nltk_list = [str(x).strip() for x in para_sents_nltk_list] # Strip leading/trailing whitespace

        ellipses_returned_para_sents = [re.sub(r' /ELLIPSIS/ ”', 
                                               r' . . .”', 
                                               x) 
                                        for x in para_sents_spacy_list]
        ellipses_returned_para_sents = [re.sub(r' /ELLIPSIS/ ', 
                                               r' . . . ', 
                                               x) 
                                        for x 
                                        in ellipses_returned_para_sents]
        # Remove empty and 1-character sentences
        para_sents_list = [x for x in ellipses_returned_para_sents 
                           if (len(x) > 1)] 
        # Remove sentences without any alphabetic characters
        para_sents_list = [x for x in para_sents_list 
                           if re.search('[a-zA-Z]', x)]

        sentences_list.extend(para_sents_list)
    
    return sentences_list


def clean_string(dirty_str: str, 
                 lowercase = True, expand_contractions = True
                 ) -> str:
    #TODO: add more functions in here that take care of stuff clean-text doesn't, like emoticons
    """Clean a string.

    Args:
        dirty_str (str): A raw string
        lowercase (bool): Whether to convert the text to lowercase.
            Defaults to True.
        expand_contractions (bool): Whether to expand contractions into
        separate words. Defaults to True.

    Returns:
        str: A cleaned string
    """
    
    # Replace ellipses
    mid_cleaning_str = re.sub(r'^ \. \. \.', r'', dirty_str) # If sentence begins with an ellipsis, remove the ellipsis.
    mid_cleaning_str = re.sub(r'^("|“) \. \. \.', r'', mid_cleaning_str) # If quote begins with an ellipsis, remove the ellipsis.
    mid_cleaning_str = re.sub(r' \. \. \.', r',', mid_cleaning_str) # Replace remaining ellipses with commas
    
    # Expand contractions
    if expand_contractions:
        mid_cleaning_str = contractions.fix(mid_cleaning_str)

    mid_cleaning_str = clean(mid_cleaning_str, # TODO: Determine if we want to keep this dependency (clean-text). Probably not bc it's not being maintained. Find alternative?
        fix_unicode = True,               # fix various unicode errors
        to_ascii = True,                  # transliterate to closest ASCII representation
        lower = lowercase,                # lowercase text
        no_line_breaks = False,           # fully strip line breaks as opposed to only normalizing them
        no_urls = False,                  # replace all URLs with a special token
        no_emails = False,                # replace all email addresses with a special token
        no_phone_numbers = False,         # replace all phone numbers with a special token
        no_numbers = False,               # replace all numbers with a special token
        no_digits = False,                # replace all digits with a special token
        no_currency_symbols = False,      # replace all currency symbols with a special token
        no_punct = False,                 # remove punctuation
        # replace_with_punct="",          # instead of removing punctuation, you may replace it
        # replace_with_url="<URL>",
        # replace_with_email="<EMAIL>",
        # replace_with_phone_number="<PHONE>",
        # replace_with_number="<NUMBER>",
        # replace_with_digit="0",
        # replace_with_currency_symbol="<CUR>",
        lang="en"
    )
    
    # Remove leading, trailing, and repeated spaces, just in case any 
    # made it through.
    clean_str = ' '.join(mid_cleaning_str.split())

    return clean_str 


def create_clean_df(sentences_list: list[str], 
                    lowercase = True, expand_contractions = True, 
                    title = "SentimentText", 
                    save = False, save_filepath = CURRENT_DIR,
                    ) -> pd.DataFrame:
    """Create DataFrame of raw and cleaned sentences.

    From a list of sentences, create a DataFrame with columns 
    'text_raw', 'cleaned_text', and 'cleaned_text_len'.
    
    Args:
        sentences_list (list of strings): A list of sentences.
        title (str, optional): The text title. Specify this if save =
            True. Defaults to "SentimentText."
        save (bool, optional): Whether to save the DataFrame (as a csv). 
            Defaults to False.
        save_filepath (str, optional): Where to save the DataFrame. 
            Defaults to current working directory.

    Returns:
        pd.DataFrame: A DataFrame with columns 'sentence_num', 
            'text_raw', 'cleaned_text', and 'cleaned_text_len' intended 
            to be passed to a sentiment analysis model function
    """
    
    # Create sentiment_df to hold text sentences and corresponding
    # sentiment values
    sentiment_df = pd.DataFrame({'text_raw': sentences_list})
    sentiment_df['text_raw'] = sentiment_df['text_raw'].astype('string')
    sentiment_df['text_raw'] = sentiment_df['text_raw'].str.strip()

    # Clean the 'text_raw' column and create the 'cleaned_text' column
    sentiment_df['cleaned_text'] = sentiment_df['text_raw'].apply(
        lambda x: 
        clean_string(x, lowercase = lowercase, 
                     expand_contractions = expand_contractions)
        ) # call clean_string(), defined above
    sentiment_df['text_raw_len'] = sentiment_df['text_raw'].apply(
        lambda x: len(x))
    sentiment_df['cleaned_text_len'] = sentiment_df['cleaned_text'].apply(
        lambda x: len(x))

    # Drop Sentence if Raw length < 1 (Double check)
    sentiment_df = sentiment_df.loc[sentiment_df['text_raw_len'] > 0]
    sentiment_df.shape

    # Fill any empty cleaned_text with a neutral word
    neutral_word = 'NEUTRALWORD'
    sentiment_df[sentiment_df['cleaned_text_len'] == 0]['cleaned_text'] = \
        neutral_word
    sentiment_df[sentiment_df['cleaned_text_len'] == 0]['cleaned_text_len'] = \
        11

    # Add Line Numbers
    sentence_nums = list(range(sentiment_df.shape[0]))
    sentiment_df.insert(0, 'sentence_num', sentence_nums)

    if save:
        download_df(sentiment_df, title, 
                    save_filepath = save_filepath, filename_suffix='_clean')
    
    return sentiment_df


def preprocess_text(raw_text_str: str, para_sep: str = r'\n{2,}', 
                    lowercase = True, expand_contractions = True,
                    title = "SentimentText", save = False, 
                    save_filepath = CURRENT_DIR,
                    )  -> pd.DataFrame:
    """Turn raw string into clean text DataFrame ready for analysis.

    Args:
        raw_text_str (str): A single string (the entire text to be
            analyzed) with paragraphs separated by para_sep.
        para_sep (str): A regular expression specifying the paragraph 
            separator. Defaults to r'\n{2,}' (two newline characters).
        title (str): The title of the text. Specify this if save =
            True. Defaults to "SentimentText." 
        save (bool, optional): Whether to download the resulting 
            DataFrame (as a csv). Defaults to False.
        save_filepath (str, optional): Where to download the DataFrame. 
            Defaults to the current working directory.

    Returns:
        pd.DataFrame: A DataFrame with columns 'sentence_num', 
            'text_raw', 'cleaned_text', and 'cleaned_text_len' intended 
            to be passed to a sentiment analysis model function
    """
    sentences_list = segment_sentences(raw_text_str, para_sep)
    clean_df = create_clean_df(sentences_list, 
                               lowercase = lowercase, 
                               expand_contractions = expand_contractions, 
                               title = title, save = save, 
                               save_filepath = save_filepath)
    return clean_df

def vader(sentiment_df: pd.DataFrame, title="") ->  pd.DataFrame:
    # TODO: remove title from each of these models' params. Not doing
    # now bc not backwards compatible (website backend code will break). AUGTODO
    """Run VADER sentiment analysis.

    Run VADER on the cleaned_text column of the passed DataFrame and 
        create a new DataFrame with an appended 'sentiment' column.
        For more details on VADER sentiment analysis, see 
        https://vadersentiment.readthedocs.io.
        
    Args:
        sentiment_df (pd.DataFrame): A DataFrame with 'sentence_num', 
            'text_raw', and 'text_cleaned' columns.

    Returns:
        pd.DataFrame: A DataFrame with 'sentence_num', 'text_raw', 
            'text_cleaned', and 'sentiment' columns, where the last
            column contains the VADER compound score (a number between
            -1 and 1) for each sentence.
    """
    # Note: The sentiment model functions create new DataFrames (instead 
    #   of adding a column to the DataFrame passed as an argument) in
    #   order to allow users to run multiple sentiment analysis models 
    #   in parallel
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sid_obj = SentimentIntensityAnalyzer()
    
    print("Running VADER sentiment analysis")
    sentiment_vader_ls = [sid_obj.polarity_scores(asentence)['compound'] 
                          for asentence 
                          in sentiment_df['cleaned_text'].to_list()]
    
    # Create new VADER DataFrame to save results
    vader_df = sentiment_df[['sentence_num', 'text_raw', 
                             'cleaned_text']].copy(deep=True)
    vader_df['sentiment'] = pd.Series(sentiment_vader_ls) 
        
    return vader_df

def textblob(sentiment_df: pd.DataFrame, title: str) -> pd.DataFrame:
    """Run TextBlob sentiment analysis.

    Run TextBlob on the cleaned_text column of the passed DataFrame and 
        create a new DataFrame with appended 'sentiment' and
        'subjectivity' columns. For more details on TextBlob sentiment 
        analysis, see https://textblob.readthedocs.io.
        
    Args:
        sentiment_df (pd.DataFrame): A DataFrame with 'sentence_num', 
            'text_raw', and 'text_cleaned' columns.

    Returns:
        pd.DataFrame: A DataFrame with 'sentence_num', 'text_raw', 
            'text_cleaned', 'sentiment', and 'subjectivity' columns, 
            where the last two columns contain the TextBlob polarity and
            subjectivity scores, respectively, for each sentence.  The 
            polarity score is a float within the range [-1.0, 1.0]. The 
            subjectivity is a float within  the range [0.0, 1.0], where 
            0.0 is very objective and 1.0 is very subjective.
    """
    from textblob import TextBlob
    
    print("Running TextBlob sentiment analysis")
    sentiment_textblob_ls = [TextBlob(asentence).sentiment 
                             for asentence 
                             in sentiment_df['cleaned_text'].to_list()]
    
    # Create new TextBlob DataFrame to save results
    textblob_df = sentiment_df[['sentence_num', 'text_raw', 
                                'cleaned_text']].copy(deep=True)
    textblob_df['sentiment'] = pd.Series([x.polarity 
                                          for x in sentiment_textblob_ls]) 
    textblob_df['subjectivity'] = pd.Series([x.subjectivity 
                                             for x in sentiment_textblob_ls]) 
    
    return textblob_df


# Create class for data preparation for transformer models
class SimpleDataset:
    """Class for data preparation for transformer models.

        For internal use only.
    """
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts
    
    def __len__(self):
        return len(self.tokenized_texts["input_ids"])
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}

def distilbert(sentiment_df: pd.DataFrame, title="") -> pd.DataFrame:
    """Run DistilBERT sentiment analysis.

    Run the "DistilBERT base uncased finetuned SST-2" model on the 
        cleaned_text column of the passed DataFrame and create a new 
        DataFrame with appended 'binary_sentiment', 'label', 'score', 
        and 'sentiment' columns. For more details on DistilBERT 
        sentiment analysis, see
        https://huggingface.co/docs/transformers/model_doc/distilbert 
        and
        huggingface.co/distilbert-base-uncased-finetuned-sst-2-english.
        
        
    Args:
        sentiment_df (pd.DataFrame): A DataFrame with 'sentence_num', 
            'text_raw', and 'text_cleaned' columns.

    Returns:
        pd.DataFrame: A DataFrame with 'sentence_num', 'text_raw', 
            'text_cleaned', 'binary_sentiment', 'label', 'score', and 
            'sentiment' columns, where the last four columns contain 
            the DistilBERT analysis results for each sentence. 
            'binary_sentiment' is a 0 or 1, representing 'negative' and
            'positive' sentiments, respectively. Those verbal labels are 
            given in the 'label' column. The 'score' for each sentence 
            reflects the model's confidence or certainty in its
            label/sentiment assignment. The 'sentiment' column contains
            an adjusted sentiment score between -1 and 1, which is equal
            to the score but negated for negatively labeled sentiments.
    """
    from transformers import AutoTokenizer
    from transformers import AutoModelForSequenceClassification, Trainer
    # Some of these imports might be needed in other transformer models 
    # to be added later (TODO)
    # from transformers import pipeline
    # from transformers import AutoModelForSeq2SeqLM, AutoModelWithLMHead
    # from transformers import BertTokenizer, BertForSequenceClassification
    # import sentencepiece
    
    print("Running DistilBERT sentiment analysis")
    # Load tokenizer and model, create trainer
    model_name = "distilbert-base-uncased-finetuned-sst-2-english" 
        # Note: If adding German models in the future, use a cased model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    trainer = Trainer(model=model)

    # Compute Sentiment Time Series
    clean_sentences_list = sentiment_df['cleaned_text'].to_list()

    # Tokenize texts and create prediction data set
    tokenized_texts = tokenizer(clean_sentences_list, 
                                truncation = True, padding = True)
    pred_dataset = SimpleDataset(tokenized_texts)

    # Run predictions
    prediction_results = trainer.predict(pred_dataset)

    # Transform sentiment predictions to labels
    sentiment_ls = np.argmax(prediction_results.predictions, axis=-1)
    labels_ls = pd.Series(sentiment_ls).map(model.config.id2label)
    
    # Calculate normalized scores (representing model confidence) 
    # using softmax and select the maximum score
    scores_ls = (np.exp(prediction_results[0])/np.exp(prediction_results[0])
                 .sum(-1, keepdims=True)).max(1)
    
    # Calculate adjusted sentiment scores based on score and 
    # positive/negative label
    adjusted_scores_ls = scores_ls * np.where(sentiment_ls == 1, 1, -1)

    # Create DataFrame with texts, predictions, labels, scores, and 
    # adjusted sentiments
    results_df = pd.DataFrame(list(zip(sentiment_ls, labels_ls, scores_ls, 
                                       adjusted_scores_ls)),
                               columns=['binary_sentiment', 'label', 'score', 
                                        'sentiment'])
    # Concatenate the calculated data DataFrame with the original 
    # distilbert_df
    distilbert_df = pd.concat([sentiment_df[['sentence_num', 
                                             'text_raw', 
                                             'cleaned_text',
                                             ]].copy(deep=True), 
                               results_df], axis = 1)

    return distilbert_df

def sentimentr(sentiment_df: pd.DataFrame, title="") -> pd.DataFrame:
    """Run SentimentR sentiment analysis.

    Use the SentimentR library in R to run a variety of lexical
    sentiment analysis models on the text.        
        
    Args:
        sentiment_df (pd.DataFrame): A DataFrame with 'sentence_num', 
            'text_raw', and 'text_cleaned' columns.

    Returns:
        pd.DataFrame: A DataFrame with 'sentence_num', 'text_raw', 
            'text_cleaned', 'sentimentr_jockersrinker', 
            'sentimentr_jockers', 'sentimentr_huliu', 'sentimentr_nrc', 
            'sentimentr_senticnet', 'sentimentr_sentiword', 
            'sentimentr_loughran_mcdonald', and 
            'sentimentr_socal_google' columns. Each sentiment score
            column is named after the lexicon used by SentimentR to
            assign sentiments to words and calculate a score for each
            sentence.
    """
    # Note: The sentimentR lexicons not included right now are
        # emojis_sentiment, hash_sentiment_emojis, and
        # hash_sentiment_slangsd beause the text is not being cleaned 
        # with those lexicons in mind.
    
    print("Running sentiment analyses with all all SentimentR lexicons")
    
    import rpy2.robjects as robjects
    r = robjects.r
    
    # Import the R file with the code to be run
    r_file_path = os.path.join(THIS_SOURCE_FILE_DIRECTORY, 'run_sentimentr.R')
    r['source'](r_file_path)
    
    # Import the get_sentimentr_values() function from the R code
    get_sentimentr_rfunction = robjects.globalenv['get_sentimentr_values']
    
    # Convert Python list of strings (cleaned sentences) to an R 
    # character vector
    sentences_vec = robjects.StrVector(sentiment_df['cleaned_text'].to_list())

    # Run the get_sentimentr_values() function from the R code, in R
    sentimentr_rdf = get_sentimentr_rfunction(sentences_vec)

    # Convert the resulting rpy2.robjects.vectors.DataFrame to a 
    # pandas.DataFrame
    sentimentr_df = pd.DataFrame.from_dict({
        key: np.asarray(sentimentr_rdf.rx2(key)) 
        for key in sentimentr_rdf.names})
    
    # Append entire sentimentr_df except the text_clean column 
    # to the passed DataFrame
    sentr_df = pd.concat([sentiment_df[['sentence_num', 'text_raw', 
                                        'cleaned_text']].copy(deep=True), 
                          sentimentr_df.iloc[:,1:]], axis = 1)

    return sentr_df

# TODO: get rid of this function? (I think the wesbite's code relies on it,
# currently, but it's not actually necessary there.) 
# Should we keep it in case people decide they want to run &
# compare more models later? -- No, because all it'd do is append a
# column to a dataframe, which the user can do on their own. AUGTODO:
# remove this function and pull request to the front end to stop using
# it. 
def combine_model_results(sentiment_df: pd.DataFrame, 
                          title = "", 
                          **kwargs) -> pd.DataFrame:
    # TODO: make these named params instead of freeform? as a check.
    '''
    Optional named args: vader = vader_df, textblob = textblob_df, 
                         distilbert = distilbert_df, nlptown = nlptown_df, 
                         roberta15lg = roberta15lg_df
    TODO: make sure this is working
    '''
    # Merge all dataframes into a new dataframe
    all_sentiments_df = sentiment_df[['sentence_num', 'text_raw',
                                      'cleaned_text']].copy(deep=True)
    for key, value in kwargs.items():
        try:
            all_sentiments_df[key] = value['sentiment']
        except:
            print(f'Warning: failed to append {key} sentiments\n')
    
    return all_sentiments_df

# AUGTODO remove title arg
def compute_sentiments(sentiment_df: pd.DataFrame, title = "", 
                       models = ALL_MODELS_LIST) -> pd.DataFrame:
    """Run sentiment analysis model(s) on a DataFrame of cleaned text.

    Args:
        sentiment_df (pd.DataFrame): A DataFrame with columns 
            'sentence_num', 'text_raw', and 'cleaned_text', where each
            row is a string (e.g., a sentence) to assign a sentiment to
        title (str, optional): The text title. This argument is not
            used.
        models (list of strings, optional): A list of the sentiment
            analysis models to be run, with lowercase titles. Defaults 
            to ['vader', 'textblob', 'distilbert', 'sentimentr'].

    Returns:
        pd.DataFrame: sentiment_df with appended columns named after
            each model, containing the sentiment score assigned to each
            string by the model
    """
    all_sentiments_df = sentiment_df[['sentence_num', 'text_raw',
                                      'cleaned_text']].copy(deep=True)
    if "vader" in models:
        all_sentiments_df['vader'] = vader(sentiment_df, title)['sentiment']
    if "textblob" in models:
        all_sentiments_df['textblob'] = textblob(sentiment_df, 
                                                 title)['sentiment']
    if "distilbert" in models:
        all_sentiments_df['distilbert'] = distilbert(sentiment_df, 
                                                     title)['sentiment']
    if "sentimentr" in models:
        all_sentiments_df = pd.concat([all_sentiments_df, 
                                       sentimentr(sentiment_df, title)
                                       .iloc[:, 5:].copy(deep=True)], 
                                      axis = 1)
    for user_model in models:
        if user_model not in ALL_MODELS_LIST:
            print(f"Warning: {user_model} model not found in list of accepted "
                  f"models. Check your spelling.")
            print(f"\nThe accepted models are:\n")
            for model in ALL_MODELS_LIST:
                print(model)
                print("\n")
    return all_sentiments_df

# This function works on a df containing multiple models, and it creates a new df with the same column names but new sentiment values.
# TODO: Also create functions that allow the user to input a df with only one model's sentiment values and append adjusted & normalized sentiments as new columns on the same df, in case they want to compare different adjustments & smoothing methods for the same model.
# TODO: make separate smoothing and plotting functions? to abstract within this one
def plot_sentiments(all_sentiments_df: pd.DataFrame, 
                    title: str, 
                    models = ALL_MODELS_LIST, #TODO this isn't going to work with sentimentR right now
                    adjustments = "normalizedZeroMean", # TODO: add a 'rescale' option, where all points are rescaled from their model's original scale to -1 to 1
                    smoothing = "sma",
                    plot = "save", # AUGTODO: change to save_plot and display_plot booleans
                    save_filepath = CURRENT_DIR, 
                    window_pct = 10,
                    ) -> pd.DataFrame:
    """Plot the raw or adjusted sentiments from the selected models.

    Save a .png plot of raw, normed, or normed & adjusted sentiments 
    from the selected models to the specified directory. Smooth 
    sentiment curves using the specified method before plotting. Also 
    return the points from the plot in the form of a [TODO].

    Args:
        all_sentiments_df (pd.DataFrame): Dataframe containing sentiment 
            values in columns named after the models in `models`
        title (str): Title of text
        models (list of strings): A list of the lowercase names of the 
            models to plot. These models' timeseries/sentiment 
            DataFrames must have the same length. Defaults to
            to ['vader', 'textblob', 'distilbert', 'sentimentr']. 
        adjustments (str): "none" (plot raw sentiments), 
            "normalizedZeroMean" (normalize to mean = 0, sd = 1), 
            "normalizedAdjMean" (normalize and add the scaled mean that 
            would be computed by adjusting the original scores so their 
            range is exactly -1 to 1). Defaults to normalizedZeroMean.
        smoothing (str): "sma" for a simple moving average (aka sliding 
            window with window size determined by window_pct), "lowess"
            for LOWESS smoothing using parameter = [TODO]
        plot (str): "display", "save", "both", or "none"
        save_filepath (str): path (ending in '/') to the directory
            the resulting plot png should be stored in.
            Defaults to the current working directory.
        window_pct (int): percentage of total text length to use as the
            window size for SMA smoothing

    Returns:
        TODO

    """
    
    if window_pct > 70 or window_pct <= 0:
        raise ValueError("Window percentage must be between 0 and 70")
    if window_pct > 20 or window_pct < 1:
        print("Warning: window percentage outside expected range (1-20)")
    window_size = int(window_pct/100 * all_sentiments_df.shape[0])

    camel_title = ''.join([re.sub(r'[^\w\s]', '', x).capitalize()
                           for x in title.split()])
    
    if adjustments == "raw":
        # Plot Raw Timeseries
        raw_rolling_mean = all_sentiments_df[['sentence_num', 'text_raw', 'cleaned_text']].copy(deep=True)
        raw_rolling_mean[models] = all_sentiments_df[models].rolling(window_size, center=True).mean()
        plt.figure().clear()
        ax = raw_rolling_mean[models].plot(grid=True, lw=3)
        ax.set_title(f'{title} Sentiment Analysis \n Raw Sentiment Timeseries')
        plt.xlabel('Sentence Number')
        plt.ylabel('Sentiment')
        if plot == "save" or plot == "both":
            completepath = os.path.join(save_filepath, f"{camel_title}_rawSentiments.png")
            plt.savefig(uniquify(completepath))
        if plot == "display" or plot == "both":
            plt.show()
        
        return raw_rolling_mean

    else:
        # Compute the mean of each raw sentiment timeseries 
        # and adjust to [-1.0, 1.0] range
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
        all_sentiments_norm_df = all_sentiments_df[['sentence_num', 'text_raw', 'cleaned_text']].copy(deep=True)
        all_sentiments_norm_df[models] = StandardScaler().fit_transform(all_sentiments_df[models])

        if adjustments == "normalizedZeroMean":
            # Plot Normalized Timeseries to same mean (Q: Is this mean 0? If not, change filename below.)
            norm_rolling_mean = all_sentiments_norm_df[['sentence_num', 'text_raw', 'cleaned_text']].copy(deep=True)
            norm_rolling_mean[models] = all_sentiments_norm_df[models].rolling(window_size, center=True).mean()
            plt.figure().clear()
            ax = norm_rolling_mean[models].plot(grid=True, lw=3)
            ax.set_title(f'{title} Sentiment Analysis \n Normalization: Standard Scaler')
            plt.xlabel('Sentence Number')
            plt.ylabel('Sentiment')
            if plot == "save" or plot == "both":
                completepath = os.path.join(save_filepath, f"{camel_title}_normalizedZeroMeanSentiments.png")
                plt.savefig(uniquify(completepath))
            if plot == "display" or plot == "both":
                plt.show()
            

            return norm_rolling_mean

        else: # adjustments == "normalizedAdjMean"
            # Plot StandardScaler + Original Mean
            # Plot Normalized Timeseries to their adjusted/rescaled original means
            all_sentiments_adjnorm_df = all_sentiments_df[['sentence_num', 'text_raw', 'cleaned_text']].copy(deep=True)
            for amodel in models:
                all_sentiments_adjnorm_df[amodel] = all_sentiments_norm_df[amodel] + models_adj_mean_dt[amodel]

            norm_adj_rolling_mean = all_sentiments_adjnorm_df[['sentence_num', 'text_raw', 'cleaned_text']].copy(deep=True)
            norm_adj_rolling_mean[models] = all_sentiments_adjnorm_df[models].rolling(window_size, center=True).mean()
            plt.figure().clear()
            ax = norm_adj_rolling_mean[models].plot(grid=True, lw=3)
            ax.set_title(f'{title} Sentiment Analysis \n Normalization: Standard Scaler + Scaled Mean Adjustment')
            plt.xlabel('Sentence Number')
            plt.ylabel('Sentiment')
            if plot == "save" or plot == "both":
                completepath = os.path.join(save_filepath, f"{camel_title}_normalizedAdjustedMeanSentiments.png")
                plt.savefig(uniquify(completepath))
            if plot == "display" or plot == "both":
                plt.show()

            return norm_adj_rolling_mean
        
    #TODO: add lowess option
    # from statsmodels.nonparametric.smoothers_lowess import lowess
    # y = current_sentiment_arc_df[selected_model.value].values
    # x = np.arange(current_sentiment_arc_df.shape[0]) # i think this is
    # just the number of rows in the df
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
                ) -> tuple[list[int], list[float], list[int], list[float]]:
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
    
    from scipy.signal import find_peaks
    
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
        completepath = os.path.join(save_filepath, f"{title} cruxes ({algo} algorithm).png")
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
                 n = 10) -> tuple[list, str]:
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
        crux_context_list.append(("peak", peak, peaks_list))

    crux_context_str += '=================================================='
    crux_context_str += '===========     Crux Valley Points    ============'
    crux_context_str += '==================================================\n\n'

    for i, valley in enumerate(valleys): # Iterate through all valleys
        crux_valleys_list = sentiment_df.iloc[valley-halfwindow-1:valley+halfwindow].text_raw
        crux_context_str += f"Valley #{i} at Sentence #{valley}:\n\n{newline.join(crux_valleys_list)}\n\n\n"
        crux_context_list.append(("valley", valley, crux_valleys_list))

    return crux_context_list, crux_context_str

