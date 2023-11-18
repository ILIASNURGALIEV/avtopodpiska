import dill
import json
import pandas as pd
import pickle
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Form(BaseModel):
    session_id: str
    client_id: str
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str

@app.get('/status')
def status():
    return "I'm ok__"

@app.post('/dict')
def predict(form: Form):
    with open('data/sber_avtopodpiska_pipe_202308241046.pkl', 'rb') as file:
        model = dill.load(file)
    
    data_1 = form.dict()
    df = pd.DataFrame.from_dict([data_1])
    y = model.predict(df)

        
    return (f'{form.session_id}: {y[0]}')

