import sys
import pytest
from imppkg.hello import doit
from imppkg.simplifiedSA import test_func


def test_always_pass():
    doit()
    test_func()
    assert True


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