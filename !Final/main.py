import numpy as np
import pipe
import dill
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import pandas as pd

app = FastAPI()

print('Загрузка модели', datetime.now())
with open('model_credit_risk.pkl', 'rb') as file:
    predict_model = dill.load(file)
trh = predict_model['metadata']['thr']
print('Модель загружена', datetime.now())

class Prediction(BaseModel):
    id: int
    pred: str


@app.get('/status')
def status():
    return "Всё работает"


@app.get('/version')
def version():
    return predict_model['metadata']


@app.get('/fit')
def fit():
    pipe.main()
    return 'Модель обучена'


@app.get('/predict')
def predict():
    # Чтение данных
    X = pipe.read_credit_history(11, 1)
    X.to_csv('test.csv', index=False)
       
    X = pd.read_csv('test.csv')
    print('dataset is readed')
    id = X.id
    X = X.reindex(columns=predict_model['metadata']['feature_names'])
    # Предсказание
    y_pred = predict_model['model'].predict_proba(X)[:, 1]

    result = (y_pred >= trh) * 1
    result = result.astype(np.int8)  # Преобразование в int8
    df = pd.DataFrame({'id': id, 'result': result})  
    df.to_csv('predict.csv', index=False)
    import json
    json_data = df.to_json(orient='records')

    return {
        json_data
    }
