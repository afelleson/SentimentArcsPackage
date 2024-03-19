# Quietly install packages (if needed) and load them
packagesList <- c("sentimentr",
                    "lexicon",
                    "data.table" # or "dplyr"
                  )
for (p in packagesList){
  if(! p %in% installed.packages()){
    install.packages(p, dependencies = TRUE)
  }
}
suppressMessages(invisible(lapply(packagesList, require, character.only = TRUE)))

# If SentimentR breaks a sentence from the input list into more than one sentence, 
# this function will average those sentences' sentiments (weighted by word count), 
# put them in one row, and flag it
restoreOldSentences = function(sentimentr_df) {
  # Group by element_id (index from original sentences vector),
  # average sentiments from sentences in the same group, and
  # flag averaged rows in the resulting data.table.
  # Use a weighted average, excluding NaNs 
  # (e.g., a sentence that's just a piece of punctuation),
  # and, finally, change NaNs to 0 at the end.
  
  dt <- as.data.table(sentimentr_df)
  dt <- dt[, 
           .(flag = if (.N > 1) 1 else 0, 
             sentiment_avg_with_nans = weighted.mean(sentiment, word_count, na.rm = TRUE)),
           by = element_id]
  dt[, .(flag=flag, sentiment_avg = ifelse(is.na(sentiment_avg_with_nans), 0, sentiment_avg_with_nans))]
}

# Calculate sentiments from a number of different models and aggregate the results
get_sentimentr_values <- function(sentences_vector) {
  
  get_sentences_obj <- get_sentences(sentences_vector)
  
  jockers_rinker <- sentiment(get_sentences_obj, 
                              polarity_dt=lexicon::hash_sentiment_jockers_rinker, 
                              hypen="", amplifier.weight=0.8, n.before=5, n.after=2,
                              adversative.weight=0.25, neutral.nonverb.like = FALSE, 
                              missing_value = NULL)
  
  jockers <- sentiment(get_sentences_obj, 
                       polarity_dt=lexicon::hash_sentiment_jockers, 
                       hypen="", amplifier.weight=0.8, n.before=5, n.after=2,
                       adversative.weight=0.25, neutral.nonverb.like = FALSE, 
                       missing_value = NULL)
  
  huliu <- sentiment(get_sentences_obj, 
                     polarity_dt=lexicon::hash_sentiment_huliu, 
                     hypen="", amplifier.weight=0.8, n.before=5, n.after=2,
                     adversative.weight=0.25, neutral.nonverb.like = FALSE, 
                     missing_value = NULL)
  
  nrc <- sentiment(get_sentences_obj, 
                   polarity_dt=lexicon::hash_sentiment_nrc, 
                   hypen="", amplifier.weight=0.8, n.before=5, n.after=2,
                   adversative.weight=0.25, neutral.nonverb.like = FALSE, 
                   missing_value = NULL)
  
  senticnet <- sentiment(get_sentences_obj, 
                         polarity_dt=lexicon::hash_sentiment_senticnet, 
                         hypen="", amplifier.weight=0.8, n.before=5, n.after=2,
                         adversative.weight=0.25, neutral.nonverb.like = FALSE, 
                         missing_value = NULL)
  
  sentiword <- sentiment(get_sentences_obj, 
                         polarity_dt=lexicon::hash_sentiment_sentiword, 
                         hypen="", amplifier.weight=0.8, n.before=5, n.after=2,
                         adversative.weight=0.25, neutral.nonverb.like = FALSE, 
                         missing_value = NULL)
  
  # loughran_mcdonald is for financial texts
  loughran_mcdonald <- sentiment(get_sentences_obj, 
                                 polarity_dt=lexicon::hash_sentiment_loughran_mcdonald, 
                                 hypen="", amplifier.weight=0.8, n.before=5, n.after=2,
                                 adversative.weight=0.25, neutral.nonverb.like = FALSE, 
                                 missing_value = 0)
  
  socal_google <- sentiment(get_sentences_obj, 
                            polarity_dt=lexicon::hash_sentiment_socal_google, 
                            hypen="", amplifier.weight=0.8, n.before=5, n.after=2,
                            adversative.weight=0.25, neutral.nonverb.like = FALSE, 
                            missing_value = NULL)
  
  sentimentr_df <- data.frame('text_clean' = sentences_vector,
                              'was_averaged' = restoreOldSentences(jockers_rinker)$flag,
                              'sentimentr_jockers_rinker' = restoreOldSentences(jockers_rinker)$sentiment_avg,
                              'sentimentr_jockers' = restoreOldSentences(jockers)$sentiment_avg,
                              'sentimentr_huliu' = restoreOldSentences(huliu)$sentiment_avg,
                              'sentimentr_nrc' = restoreOldSentences(nrc)$sentiment_avg,
                              'sentimentr_senticnet' = restoreOldSentences(senticnet)$sentiment_avg,
                              'sentimentr_sentiword' = restoreOldSentences(sentiword)$sentiment_avg,
                              'sentimentr_loughran_mcdonald' = restoreOldSentences(loughran_mcdonald)$sentiment_avg,
                              'sentimentr_socal_google' = restoreOldSentences(socal_google)$sentiment_avg
  )
  return(sentimentr_df)
}


### Usage ###

# mytext <- c(
#   "do you like it?  But I hate really bad dogs.",
#   "",
#   "I am the best friend.",
#   "Do you really like it?  I'm not a fan",
#   "It's like a tree.",
#   "i don't know . . ."
# )
# 
# results <- get_sentimentr_values(mytext)