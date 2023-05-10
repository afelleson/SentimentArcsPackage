import sys
import pytest
import os
from imppkg.simplifiedSA import *

# def test_preprocess():
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     file_path = os.path.join(current_dir, 'scollins_thehungergames1.txt')
    
#     print(current_dir)
    
#     with open(file_path, 'r') as file:
#         text = file.read()
        
#     title = "Text Title"

#     sentiment_df = preprocess_text(text, title)
#     preview(sentiment_df)
#     assert True

# def test_big_plot():
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     file_path = os.path.join(current_dir, 'scollins_thehungergames1.txt')
    
#     print(current_dir)
    
#     with open(file_path, 'r') as file:
#         text = file.read()
        
#     title = "Text Title"

#     sentiment_df = preprocess_text(text, title)
    
#     all_sentiments_df = compute_sentiments(sentiment_df, title)

    #   download_df(all_sentiments_df, title)

#     assert True

# def test_smooth():
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     file_path = os.path.join(current_dir, 'scollins_thehungergames1.txt')
    
#     print(current_dir)
    
#     with open(file_path, 'r') as file:
#         text = file.read()
        
#     title = "Text Title"

#     sentiment_df = preprocess_text(text, title)
    
#     all_sentiments_df = compute_sentiments(sentiment_df, title)
    
#     smoothed_sentiments_df = plot_sentiments(all_sentiments_df, title,
#                                                 adjustments="normalizedAdjMean")

#     download_df(smoothed_sentiments_df, title)
    
#     assert True

# def test_cruxes():
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     file_path = os.path.join(current_dir, 'scollins_thehungergames1.txt')
    
#     print(current_dir)
    
#     with open(file_path, 'r') as file:
#         text = file.read()
        
#     title = "Text Title"

#     sentiment_df = preprocess_text(text, title)
    
#     all_sentiments_df = compute_sentiments(sentiment_df, title)
    
#     smoothed_sentiments_df = plot_sentiments(all_sentiments_df, title,
#                                                 adjustments="normalizedAdjMean")
    
#     cruxes = find_cruxes(smoothed_sentiments_df, 
#                          'vader',
#                          title,
#                          algo = "width",
#                          plot = "save",
#                          save_filepath = "./plots/",
#                          width_min = 25)

#     peak_xs, peak_ys, valley_xs, valley_ys = cruxes

#     assert len(peak_xs) > 1
