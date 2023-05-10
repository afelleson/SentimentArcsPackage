# Stuff to potentially add in the future, based on an OG SentimentArcs util

import spacy

from copy import deepcopy
from pysbd.utils import PySBDFactory
import re

# do this for lexical models
def text2lemmas(comment, lowercase, remove_stopwords):
    nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner'])
    stopwords_ls = ['a', 'the', 'an']
    if lowercase:
        comment = comment.lower()
    comment = nlp(comment)
    lemmatized = list()
    for word in comment:
        lemma = word.lemma_.strip()
        if lemma:
            if not remove_stopwords or lemma not in stopwords_ls:
                lemmatized.append(lemma)
    return " ".join(lemmatized)

# TODO: use part of this to add a col to the main df keeping track of where paragraph breaks are
def text_str2sents(text_str, pysbd_only=False):
  '''
  Given a long text string (e.g. a novel) and pysbd_only flag
  Return a list of every Sentence defined by (a) 2+ newlines as paragraph separators, 
                                            (b) SpaCy+PySBD Pipeline, and 
                                            (c) Optionally, NLTK sentence tokenizer
  '''

  parags_ls = []
  sents_ls = []

  global re
  global sent_tokenize
  global nlp

  nlp = spacy.blank('en')
  nlp.add_pipe(PySBDFactory(nlp))

  print(f'BEFORE stripping out headings len: {len(text_str)}')

  parags_ls = re.split(r'[\n]{2,}', text_str)

  parags_ls = [x.strip() for x in parags_ls]

  # Strip out non-printing characters
  parags_ls = [re.sub(f'[^{re.escape(string.printable)}]', '', x) for x in parags_ls]

  # Filter out empty lines Paragraphs
  parags_ls = [x for x in parags_ls if (len(x.strip()) >= global_vars.MIN_PARAG_LEN)]

  print(f'   Parag count before processing sents: {len(parags_ls)}')
  # FIRST PASS at Sentence Tokenization with PySBD

  for i, aparag in enumerate(parags_ls):
  

    aparag_nonl = re.sub('[\n]{1,}', ' ', aparag)
    doc = nlp(aparag_nonl)
    aparag_sents_pysbd_ls = list(doc.sents)
    print(f'pysbd found {len(aparag_sents_pysbd_ls)} Sentences in Paragraph #{i}')

    # Strip ofaparag_sents_pysbd_lsf whitespace from Sentences
    aparag_sents_pysbd_ls = [str(x).strip() for x in aparag_sents_pysbd_ls]

    # Filter out empty line Sentences
    aparag_sents_pysbd_ls = [x for x in aparag_sents_pysbd_ls if (len(x.strip()) > global_vars.MIN_SENT_LEN)]

    print(f'      {len(aparag_sents_pysbd_ls)} Sentences remain after cleaning')

    sents_ls += aparag_sents_pysbd_ls

  # (OPTIONAL) SECOND PASS as Sentence Tokenization with NLTK
  if pysbd_only == True:
    # Only do one pass of SpaCy/PySBD Sentence tokenizer
    # sents_ls += aparag_sents_pysbd_ls
    pass
  else:
    # Do second NLTK pass at Sentence tokenization if pysbd_only == False
    # Do second pass, tokenize again with NLTK to catch any Sentence tokenization missed by PySBD
    # corpus_sents_all_nltk_ls = []
    # sents_ls = []
    # aparag_sents_nltk_ls = []
    aparag_sents_pysbd_ls = deepcopy(sents_ls)
    sents_ls = []
    for asent in aparag_sents_pysbd_ls:
      print(f'Processing asent: {asent}')
      aparag_sents_nltk_ls = []
      aparag_sents_nltk_ls = sent_tokenize(asent)

      # Strip off whitespace from Sentences
      aparag_sents_nltk_ls = [str(x).strip() for x in aparag_sents_nltk_ls]

      # Filter out empty line Sentences
      aparag_sents_nltk_ls = [x for x in aparag_sents_nltk_ls if (len(x.strip()) > global_vars.MIN_SENT_LEN)]

      # corpus_sents_all_second_ls += aparag_sents_nltk_ls

      sents_ls += aparag_sents_nltk_ls

  print(f'About to return sents_ls with len = {len(sents_ls)}')
  
  return sents_ls
