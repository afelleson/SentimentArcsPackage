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
# this function will average those sentences' sentiments, put them in one row, and 
# flag it
restoreOldSentences = function(sentimentr_df) {
  # Group by element_id (index from original sentences vector),
  # average sentiments from sentences in the same group, and
  # flag averaged rows in the resulting data.table
  
  # dplyr version of the code (more readable):
  # sentimentr_df %>%
  #   group_by(element_id) %>%
  #   summarize(flag = if (n() > 1) 1 else 0,
  #             sentiment_avg = mean(sentiment, na.rm = TRUE))
  
  # data.table version (should be faster):
  dt <- as.data.table(sentimentr_df)
  dt[, 
     .(flag = if (.N > 1) 1 else 0, 
       sentiment_avg = mean(sentiment, na.rm = TRUE)), 
     by = element_id]
  
}

# Calculate sentiments from a number of different models and aggregate the results
get_sentimentr_values <- function(sentences_vector) {
  
  get_sentences_obj <- get_sentences(sentences_vector)
  
  jockersrinker <- sentiment(get_sentences_obj, polarity_dt=lexicon::hash_sentiment_jockers_rinker, 
                                        hypen="", amplifier.weight=0.8, n.before=5, n.after=2,
                                        adversative.weight=0.25, neutral.nonverb.like = FALSE, missing_value = 0)
  
  jockers <- sentiment(get_sentences_obj, polarity_dt=lexicon::hash_sentiment_jockers, 
                                  hypen="", amplifier.weight=0.8, n.before=5, n.after=2,
                                  adversative.weight=0.25, neutral.nonverb.like = FALSE, missing_value = 0)
  
  huliu <- sentiment(get_sentences_obj, polarity_dt=lexicon::hash_sentiment_huliu, 
                                hypen="", amplifier.weight=0.8, n.before=5, n.after=2,
                                adversative.weight=0.25, neutral.nonverb.like = FALSE, missing_value = 0)
  
  nrc <- sentiment(get_sentences_obj, polarity_dt=lexicon::hash_sentiment_nrc, 
                              hypen="", amplifier.weight=0.8, n.before=5, n.after=2,
                              adversative.weight=0.25, neutral.nonverb.like = FALSE, missing_value = 0)
  
  senticnet <- sentiment(get_sentences_obj, polarity_dt=lexicon::hash_sentiment_senticnet, 
                                    hypen="", amplifier.weight=0.8, n.before=5, n.after=2,
                                    adversative.weight=0.25, neutral.nonverb.like = FALSE, missing_value = 0)
  
  sentiword <- sentiment(get_sentences_obj, polarity_dt=lexicon::hash_sentiment_sentiword, 
                                    hypen="", amplifier.weight=0.8, n.before=5, n.after=2,
                                    adversative.weight=0.25, neutral.nonverb.like = FALSE, missing_value = 0)
  
  loughran_mcdonald <- sentiment(get_sentences_obj, polarity_dt=lexicon::hash_sentiment_loughran_mcdonald, 
                                            hypen="", amplifier.weight=0.8, n.before=5, n.after=2,
                                            adversative.weight=0.25, neutral.nonverb.like = FALSE, missing_value = 0)
  
  socal_google <- sentiment(get_sentences_obj, polarity_dt=lexicon::hash_sentiment_socal_google, 
                                       hypen="", amplifier.weight=0.8, n.before=5, n.after=2,
                                       adversative.weight=0.25, neutral.nonverb.like = FALSE, missing_value = 0)
  
  sentimentr_df <- data.frame('text_clean' = sentences_vector,
                              'was_averaged' = restoreOldSentences(jockersrinker)$flag,
                              'sentimentr_jockersrinker' = restoreOldSentences(jockersrinker)$sentiment_avg,
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
#   'do you like it?  But I hate really bad dogs',
#   '',
#   'I am the best friend.',
#   "Do you really like it?  I'm not a fan",
#   "It's like a tree."
# )

# results <- get_sentimentr_values(mytext)

