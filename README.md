# SentArcs
A wrapper for comparing results from an ensemble of sentiment analysis models.

## Installation
Download this GitHub repository as a .zip. Unzip it.
Use this console shell command to install SentimentArcsPackage :
```shell
$ python3 -m pip install /path/to/SentimentArcsPackage 
```
To update an existing local copy of the package, run:
```shell
$ pip install --upgrade /path/to/SentimentArcsPackage
```

## Usage
Import and use within a python script, say my_script.py:
```python
import imppkg as sa

def main():
    with open('scollins_thehungergames1.txt', 'r') as file:
        text = file.read()

    sentiment_df = sa.preprocess_text(text)

    textblob_df = sa.compute_sentiments(sentiment_df, models=["textblob"])

    sa.download_df(textblob_df, title="The Hunger Games", filename_suffix='_textblob_sentiments')

if __name__ == "__main__":
    main()
```
Then, in a console shell:
```shell
python3 /path/to/my_script.py
```

Or, import and use within an interactive python notebook through the interface of your choice (e.g., Google Colab, Anaconda) using the code in the main() function above.
