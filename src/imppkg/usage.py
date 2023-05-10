# First install: py -m pip install ~/SentimentArcsPackage (or whatever the path is to your clone of the SentimentArcsPackage repo)
# pip install --upgrade ~/SentimentArcsPackage # to update the package (if it was edited and you pulled the new version) but do not force reinstall all its dependencies
# pip install --upgrade --force-reinstall ~/SentimentArcsPackage # to update your version of the package and uninstall & reinstall all of its dependencies

import imppkg.simplifiedSA as SA
# from imppkg.simplifiedSA import * # also works, but may run into namespace(?) issues

# ## TODO: every function that may raise an exception should be within a try except else block, like this:
# #     try:
# #         f = open(arg, 'r')
# #     except OSError as error:
# #         print('cannot open', arg)
# #         print(f"Unexpected {error: }")
# #     else:
# #         print(arg, 'has', len(f.readlines()), 'lines')
# #         f.close() 

# # except Exception as error:
# #     print(f"Unexpected {error=}, {type(error)=}") # print or log the exception
# #     raise # raise it for the user to be able to catch (& see the standard traceback) as well


def main():
    # Open a local file containing the raw text to analyze.
    with open('scollins_thehungergames1.txt', 'r') as file:
        text = file.read()
    title = "The Hunger Games" # Name the text document. This will be  
                               # used to label plots and name files 
                               # saved by this program

    # Segment text into sentences, clean it, and remove empty sentences
    sentiment_df = SA.preprocess_text(text, title)
    
    # Make sure the result looks how you expected
    SA.preview(sentiment_df)

    # Compute sentiment values for each sentence. 
    # Available models are "vader", "textblob", and "distbert".
    # Beware: distilbert takes a long time to run.
    all_sentiments_df = SA.compute_sentiments(sentiment_df, title, models=["vader","textblob"])

    # Smooth, normalize, and adjust the sentiments for plotting.
    # This returns a dataframe you can plot on your own, and it also
    # uses matplotlib to create a plot (showing all models) that you 
    # can choose to save or view.
    smoothed_sentiments_df = SA.plot_sentiments(all_sentiments_df, title,
                                                adjustments="normalizedAdjMean",
                                                models=["vader","textblob"])

    # For one model, identify the crux points (maxes & mins) using
    # a specific algorithm and parameter. This creates a plot as well.
    # It returns the x and y values of each other crux points on the plot,
    # separated out into peaks and valleys so you can label them differently
    # on your own plot if you'd like.
    cruxes = SA.find_cruxes(smoothed_sentiments_df, 
                            title=title,
                            model="vader",
                            algo = "width",
                            plot = "save",
                            save_filepath = "./plots/",
                            width_min = 25)


    peak_xs, peak_ys, valley_xs, valley_ys = cruxes
    
    # Return a list of sentences around each crux point and a string
    # containing the same info.
    crux_sents, crux_str = SA.crux_context(smoothed_sentiments_df, peak_xs, valley_xs, n=5)
    print(crux_str)
    




# Run main() only when this file is run (not when it's imported as a module)
if __name__ == "__main__":
    main()