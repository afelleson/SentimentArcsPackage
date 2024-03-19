# SimplifiedSentimentArcs
A Python package for running an ensemble of sentiment analysis models
and comparing their results. Wraps functions for text cleaning and the
VADER, TextBlob, DistilBERT, and SentimentR sentiment analysis models,
and provides a new function for plotting the results with various
adjustments.

Created for
[SentimentArcs_WebApp](https://github.com/DevAkre/SentimentArcs_WebApp)
and other uses.

## Installation
Clone this GitHub repository, or download it as a .zip and unzip it.
Use this console shell command to install the package:
```shell
$ python3 -m pip install /path/to/SentimentArcsPackage 
```
To reinstall after an update to the existing local copy of the package, run:
```shell
$ pip install --upgrade /path/to/SentimentArcsPackage
```

## Usage
Import and use within a python script, say my_script.py:
```python
import imppkg as sa

def main():
    with open("scollins_thehungergames.txt", "r") as file:
        text = file.read()

    title = "The Hunger Games"

    clean_df = sa.preprocess_text(text)

    distilbert_df = sa.compute_sentiments(clean_df, models=["distilbert"])

    sa.download_df(distilbert_df, title, filename_suffix="_distilbert_raw_sentiments")

    sentiment_results_df = sa.compute_sentiments(clean_df, title, models=["vader", "textblob", "sentimentr"])

    smoothed_no_adjustments_df = sa.plot_sentiments(sentiment_results_df, title,
                                                            adjustments="none", plot = "save")

    smoothed_zero_mean_df = sa.plot_sentiments(sentiment_results_df, title, models = ["vader", "textblob", "distilbert",
                                                       "sentimentr_jockers_rinker", "sentimentr_jockers", "sentimentr_huliu"],
                                                       plot = "display")

if __name__ == "__main__":
    main()
```
Then, in a console shell:
```shell
$ python3 /path/to/my_script.py
```

Or, import and use within an interactive python notebook through the interface of your choice (e.g., Google Colab, Jupyter Notebook) using the code in the main() function above.
