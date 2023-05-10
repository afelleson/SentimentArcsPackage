# First install: py -m pip install ~/SentimentArcsPackage (or whatever the path is to your clone of the SentimentArcsPackage repo)
# pip install --upgrade ~/SentimentArcsPackage # to update the package (if it was edited and you pulled the new version) but do not force reinstall all its dependencies
# pip install --upgrade --force-reinstall ~/SentimentArcsPackage # to update your version of the package and uninstall & reinstall all of its dependencies

import imppkg.simplifiedSA as SA
# from imppkg.simplifiedSA import * # also works, but may run into namespace(?) issues

# ## every function that may raise an exception should be within a try except else block, like this:
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
    # Code to be executed when the script is run

    with open('scollins_thehungergames1.txt', 'r') as file:
        text = file.read()
    title = "Text Title"

    sentiment_df = SA.preprocess_text(text, title)
    print("\npreprocess_text done\n")
    
    SA.preview(sentiment_df)

    all_sentiments_df = SA.compute_sentiments(sentiment_df, title, models=["vader","textblob"])
    print("\ncompute_sentiments done\n")

    smoothed_sentiments_df = SA.plot_sentiments(all_sentiments_df, title,
                                                adjustments="normalizedAdjMean",
                                                models=["vader","textblob"])
    print("\nplot_sentiments done\n")
    
    SA.preview(smoothed_sentiments_df)
    
    print("\n2nd preview done\n")

    cruxes = SA.find_cruxes(smoothed_sentiments_df, 
                            model="vader",
    						title=title,
                            algo = "width",
                            plot = "save",
                            save_filepath = "./plots/",
                            width_min = 25)

    print("\nfind_cruxes done\n")

    peak_xs, peak_ys, valley_xs, valley_ys = cruxes
    
    print("\ncruxes unpacked successfully\n")

    crux_sents, crux_str = SA.crux_context(smoothed_sentiments_df, peak_xs, valley_xs, n=5)
    print(crux_str)
    
    print("\n\ndone!")




# Run main() only when this file is run (not when it's imported as a module)
if __name__ == "__main__":
    main()