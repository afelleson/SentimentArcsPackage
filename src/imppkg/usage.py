# py -m pip install ~/SentimentArcsPackage (or whatever the path is to the clone of the SentimentArcsPackage repo)

import imppkg.simplifiedSA as SA

## every function that may raise an exception should be within a try except else block, like this:
#     try:
#         f = open(arg, 'r')
#     except OSError as error:
#         print('cannot open', arg)
#         print(f"Unexpected {error=}")
#     else:
#         print(arg, 'has', len(f.readlines()), 'lines')
#         f.close() 

# except Exception as error:
#     print(f"Unexpected {error=}, {type(error)=}") # print or log the exception
#     raise # raise it for the user to be able to catch (& see the standard traceback) as well

with open('input.txt', 'r') as file:
    text = file.read()
title = "Text Title"

sentiment_df = SA.preprocess_text(text, title)
# preview()

all_sentiments_df = SA.compute_sentiments(sentiment_df, title)

smoothed_sentiments_df = SA.plot_sentiments(all_sentiments_df, title,
                                            adjustments="normalizedAdjMean")

cruxes = SA.detect_peaks(smoothed_sentiments_df, 
                         'vader',
                         title,
                         algo = "width",
                         plot = "save",
                         save_filepath = "./plots/",
                         width_min = 25)

peak_xs, peak_ys, valley_xs, valley_ys = cruxes