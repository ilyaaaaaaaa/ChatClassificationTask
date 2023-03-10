import flask
from flask import Flask, jsonify, request
from flask_restful import Api, Resource, reqparse
import pickle
import json
import numpy as np
import pandas as pd


# предобработка текста
# заготовим стоп-слова
from nltk.corpus import stopwords
from stop_words import get_stop_words
russian_stopwords = stopwords.words('russian')
russian_stopwords_2 = get_stop_words('russian')
russian_stopwords  = list(set(russian_stopwords) | set(russian_stopwords_2))

from string import punctuation
deleted_symols = punctuation + '0123456789'

from pymystem3 import Mystem
mystem = Mystem() 

def prepare(text, deleted_symols=deleted_symols, russian_stopwords=russian_stopwords, mystem=mystem):
    '''
    функция для предобработки текста
    '''
    # удалим знаки пунктуации и числа
    text = ''.join([char for char in text if str(char) not in deleted_symols])
    # удалим стоп-слова
    text = ' '.join([word for word in text.lower().split(' ') if word not in russian_stopwords])
    # лематизируем слова
    text = ''.join(mystem.lemmatize(text)).strip()
    return text


# загружаем модели
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('w2v.pickle', 'rb') as f:
    w2v = pickle.load(f)

    
# функции для векторизации текста и предикта   
def w2v_trans_word(word, w2v=w2v):
    '''
    вспомогательная функция для обработки слов, которых нет в словаре
    '''
    try:
        return w2v.wv.get_vector(word)
    except:
        return np.zeros(100)

def w2v_trans_doc(doc, w2v=w2v):
    '''
    функция для преобразования pd.Series в вектор w2v через метод apply
    '''
    return np.mean([w2v_trans_word(word, w2v) for word in doc.split()], axis=0)

def w2v_vectorizer(X_test, w2v=w2v):
    '''
    функция для преобразования pd.Series в вектор w2v через метод apply,
    а также для обработки ошибок сообщений, где оказались только стоп-слова
    '''
    return np.vstack([np.zeros(100) if np.isnan(np.std(doc)) else doc for doc in 
                     X_test.apply(w2v_trans_doc, w2v).tolist()])

def get_pred(doc, w2v=w2v):
    '''
    функция, чтобы принять на вход документ как строку, векторизовать его и получить вероятность класса
    '''
    vec = w2v_vectorizer(pd.Series(doc), w2v)
    prob = model.predict_proba(vec)[:,1][0]
    return prob
        

app = Flask(__name__)

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    doc = request.form.to_dict()['review_text']
    prob = get_pred(prepare(doc))
    pred = 'pos' if prob>=0.5 else 'neg'
    prob = round(prob*100)
    prob = prob if pred=='pos' else 100-prob
    return flask.render_template('predict.html', prediction=pred, prob=prob)

if __name__ == '__main__':
    app.run(debug=True, port=5000) # , host='localhost'
