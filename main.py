import requests as requests
from bs4 import BeautifulSoup
from imdb import extract_imdb_pages2
from save import insert_colname_to_file

from reviewscrap import extract_review_pages, file_open
from preprocessing import read_csv, clean_data
from rnn import label_data, naive_bayes, cnn_model, rnn_model, logistic_regression_model
from multiprocessing import Pool




##Data Extraction
#Multiprocessing
pool = Pool(processes = 4)

#Scrap movie data
insert_colname_to_file("imdb_data.csv", ['title', 'id', 'reviewlink'])
pool.map(extract_imdb_pages2, range(1,10001, 50))


#Scrap movie reviews and rating
insert_colname_to_file("reviews_ratings.csv", ['rating','review'])
pool.map(extract_review_pages, file_open())

#Data Preprocessing 
data = read_csv('reviews_ratings.csv')

#Label data. rating > 8 : Positive, rating < 4 : Negative 5 < Neutral < 8
all_data = label_data(data)

#Modeling
print("Processing Logistic regression models...")
lr = logistic_regression_model(all_data)
print("Processing naive bayes...")
nb = naive_bayes(all_data)
print("Processing CNN models...")
cnn = cnn_model(all_data)
print("Processing RNN models...")
rnn = rnn_model(all_data)

#Comparison between models. 
print("Accuracy is ", 'lr: ', lr, 'nb : ', nb, 'cnn : ', cnn, 'rnn : ', rnn)