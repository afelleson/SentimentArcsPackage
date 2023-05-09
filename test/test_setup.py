import sys
import pytest
from imppkg.hello import doit
from imppkg.simplifiedSA import *


def test_always_pass():
    doit()
    assert True

def test_config(capfd):
    print(TEXT_ENCODING)
    out, err = capfd.readouterr() # for testing things printed to the console
    assert out == 'utf-8\n'
    
    
def test_config2(capfd):
    assert PARA_SEP == "\\n\\n"

import os

def test_all():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'scollins_thehungergames1.txt')
    
    print(current_dir)
    
    with open(file_path, 'r') as file:
        text = file.read()
        
    title = "Text Title"

    sentiment_df = preprocess_text(text, title)
    preview(sentiment_df)

    all_sentiments_df = compute_sentiments(sentiment_df, title)

    smoothed_sentiments_df = plot_sentiments(all_sentiments_df, title,
                                                adjustments="normalizedAdjMean")

    cruxes = find_cruxes(smoothed_sentiments_df, 
                         'vader',
                         title,
                         algo = "width",
                         plot = "save",
                         save_filepath = "./plots/",
                         width_min = 25)

    peak_xs, peak_ys, valley_xs, valley_ys = cruxes

    assert len(peak_xs) > 1



    # Tests from simplifiedSA ipynb:
    # save_df2csv_and_download(temp_df, '_bert-nlptown.txt')
    # clean_str("This \n\n\n is a very dirty DIRTY StrInG!!")

    # distilbert:
    # # Test
    # line_ls = ['I like that','That is annoying','This is great!','Wouldn´t recommend it.']
    # # Tokenize texts and create prediction data set
    # tokenized_texts = tokenizer(line_ls,truncation=True,padding=True)
    # pred_dataset = SimpleDataset(tokenized_texts)
    # # Run predictions
    # predictions = trainer.predict(pred_dataset)
    # # Transform predictions to labels
    # sentiment_ls = predictions.predictions.argmax(-1)
    # labels_ls = pd.Series(sentiment_ls).map(model.config.id2label)
    # scores_ls = (np.exp(predictions[0])/np.exp(predictions[0]).sum(-1,keepdims=True)).max(1)
    # # Create DataFrame with texts, predictions, labels, and scores
    # line_no_ls = list(range(len(sentiment_ls)))
    # distilbert_df = pd.DataFrame(list(zip(line_no_ls, line_ls,sentiment_ls,labels_ls,scores_ls)), columns=['line_no','line','sentiment','label','score'])
    # distilbert_df.head()