import flask
from flask import Flask, jsonify, request
from flask_restful import Api, Resource, reqparse
import pickle
import json
import numpy as np
import pandas as pd

# загружаем модели
with open('/Users/ilya/Downloads/Text_classification_task/model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('/Users/ilya/Downloads/Text_classification_task/w2v.pickle', 'rb') as f:
    w2v = pickle.load(f)

    
# функции для предобработки текста и предикта   
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
    doc = request.form.to_dict()
    prob = get_pred(doc)
    pred = 'pos' if prob>=0.5 else 'neg'
    prob = round(prob*100)
    prob = prob if pred=='pos' else 100-prob
    return flask.render_template('predict.html', input_doc=doc, prediction=pred, prob=prob)

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
