import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense, Dropout, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LogisticRegression



def label_data(data):

    # Choose 1, 2, 3, 4 as negative, 9,10 as positive. Label negative 0, positive 1. 
    positive_data = data[(data['rating'] > 8)]
    negative_data = data[(data['rating'] < 4)]

    #Create new label
    negative_data=negative_data.assign(label = [0]*len(negative_data))
    positive_data=positive_data.assign(label = [1]*len(positive_data))

    # Make the amount of positivity and negativity the same.
    number = len(negative_data)
    positive_data = positive_data.sample(number)

    #Combine positive and negative datasets

    all_data = pd.concat([positive_data, negative_data])

    return all_data


def naive_bayes(all_data):
    y = all_data['label']
    x = all_data['review']

     # Divide data into 30:70 for testing and training   
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state=1234)

    dtmvector = CountVectorizer()
    X_train_dtm = dtmvector.fit_transform(X_train)
    #print(X_train_dtm.shape)

    tfidf_transformer = TfidfTransformer()
    tfidfv = tfidf_transformer.fit_transform(X_train_dtm)
    #print(tfidfv.shape) 

    model = MultinomialNB()
    model.fit(tfidfv, y_train)

    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

    X_test_dtm = dtmvector.transform(X_test) #Test data -> DTM
    tfidfv_test = tfidf_transformer.transform(X_test_dtm) #DTM -> TF-IDF

    predicted = model.predict(tfidfv_test) #Predict test data 
    



    return accuracy_score(y_test, predicted)

def logistic_regression_model(all_data):
    y = all_data['label']
    x = all_data['review']

     # Divide data into 30:70 for testing and training   
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state=1234)

    dtmvector = CountVectorizer()
    X_train_dtm = dtmvector.fit_transform(X_train)
    #print(X_train_dtm.shape)

    tfidf_transformer = TfidfTransformer()
    tfidfv = tfidf_transformer.fit_transform(X_train_dtm)
    #print(tfidfv.shape) 

    model = LogisticRegression(class_weight = 'balanced')
    model.fit(tfidfv, y_train)


    X_test_dtm = dtmvector.transform(X_test) #Test data -> DTM
    tfidfv_test = tfidf_transformer.transform(X_test_dtm) #DTM -> TF-IDF

    predicted = model.predict(tfidfv_test)

    return accuracy_score(y_test, predicted)




def rnn_model(all_data):

    y = all_data['label']
    x = all_data['review']

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x) # X_data의 각 행에 토큰화를 수행
    sequences = tokenizer.texts_to_sequences(x) # 단어를 숫자값, 인덱스로 변환하여 저장

    word_to_index = tokenizer.word_index

    threshold = 2
    total_cnt = len(word_to_index) # 단어의 수
    rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
    total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
        if(value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value

    tokenizer = Tokenizer(num_words = total_cnt - rare_cnt + 1) #1번 이하로 등장하는 단어는 제외
    tokenizer.fit_on_texts(x)
    sequences = tokenizer.texts_to_sequences(x) # 단어를 숫자값, 인덱스로 변환하여 저장

    X_data = sequences
    #print('max of reviews : %d' % max(len(l) for l in X_data))
    #print('the avg of reviews : %f' % (sum(map(len, X_data))/len(X_data)))

    vocab_size = vocab_size = total_cnt - rare_cnt + 1
    #print('size of word collection: {}'.format((vocab_size)))

    n_of_train = int(len(sequences) * 0.5)
    n_of_test = int(len(sequences) - n_of_train)
    #print('The number of training data :',n_of_train)
    #print('The number of test data:',n_of_test)

    max_len = max(len(l) for l in X_data)
    #  fit to max_length 
    data = pad_sequences(X_data, maxlen = max_len)
    #print("데이터의 크기(shape): ", data.shape)

    # Divide data into 30:70 for testing and training   
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size= 0.3, random_state=1234)


    model = Sequential()
    model.add(Embedding(vocab_size, 32)) 
    model.add(SimpleRNN(32)) 
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

    
    return model.evaluate(X_test, y_test)[1]


def cnn_model(all_data):

    
    y = all_data['label']
    x = all_data['review']

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x) # X_data의 각 행에 토큰화를 수행
    sequences = tokenizer.texts_to_sequences(x) # 단어를 숫자값, 인덱스로 변환하여 저장

    word_to_index = tokenizer.word_index

    threshold = 2
    total_cnt = len(word_to_index) # 단어의 수
    rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
    total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
        if(value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value

    tokenizer = Tokenizer(num_words = total_cnt - rare_cnt + 1) #1번 이하로 등장하는 단어는 제외
    tokenizer.fit_on_texts(x)
    sequences = tokenizer.texts_to_sequences(x) # 단어를 숫자값, 인덱스로 변환하여 저장

    X_data = sequences
    #print('max of reviews : %d' % max(len(l) for l in X_data))
    #print('the avg of reviews : %f' % (sum(map(len, X_data))/len(X_data)))

    vocab_size = vocab_size = total_cnt - rare_cnt + 1
    #print('size of word collection: {}'.format((vocab_size)))

    n_of_train = int(len(sequences) * 0.5)
    n_of_test = int(len(sequences) - n_of_train)
   # print('The number of training data :',n_of_train)
    #print('The number of test data:',n_of_test)

    max_len = max(len(l) for l in X_data)
    #  fit to max_length 
    data = pad_sequences(X_data, maxlen = max_len)
    #print("데이터의 크기(shape): ", data.shape)

    # Divide data into 30:70 for testing and training   
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size= 0.3, random_state=1234)

    model = Sequential()
    model.add(Embedding(vocab_size, 256))
    model.add(Dropout(0.3))
    model.add(Conv1D(256, 3, padding='valid', activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 3)
    mc = ModelCheckpoint('best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

    model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['acc'])
    history = model.fit(X_train, y_train, epochs = 10, validation_data = (X_test, y_test), callbacks=[es, mc])

    loaded_model = load_model('best_model.h5')
    print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

    return loaded_model.evaluate(X_test, y_test)[1]