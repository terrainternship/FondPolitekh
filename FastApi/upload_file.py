import shutil
from fastapi import FastAPI, UploadFile, Request
from keras.models import load_model
import datetime as dt
from datetime import datetime
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np

from extra import graph_3m, graph_6m, graph_index_3m, graph_index_6m, \
    data_for_index, data_predict


model = load_model('conv_lstm_best.h5')
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.post("/upload/")
async def create_upload_files(request: Request,
                              files: list[UploadFile]):
    flag_load = False
    filename_load = ''
    filename = ''
    for file in files:
        with open(file.filename, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
            filename = file.filename
            filename_load = f'Файл {file.filename} загружен'
            flag_load = True

    df = data_predict(filename, model)
    df = data_for_index(df)

    show_3m = graph_3m(x_data=df.index, y_data=df.y)

    show_6m = graph_6m(x_data=df.index, y_data=df.y)

    index_instant_show_3m = graph_index_3m(
        x_data=df.index,
        y_data=df.instant_index,
        title_text='Индекс контрактования: '
                   'Мгновенный год-год (3-х месячный прогноз)')

    index_instant_show_6m = graph_index_6m(
        x_data=df.index,
        y_data=df.instant_index,
        title_text='Индекс контрактования: '
                   'Мгновенный год-год (6-и месячный прогноз)')

    index_month_show_3m = graph_index_3m(
        x_data=df.index,
        y_data=df.month_index,
        title_text='Индекс контрактования: '
                   'Текущий год-год (3-х месячный прогноз)')

    index_month_show_6m = graph_index_6m(
        x_data=df.index,
        y_data=df.month_index,
        title_text='Индекс контрактования: '
                   'Текущий год-год (6-и месячный прогноз)')

    index_year_show_3m = graph_index_3m(
        x_data=df.index,
        y_data=df.year_index,
        title_text='Индекс контрактования: '
                   'Длинный год-год (3-х месячный прогноз)')

    index_year_show_6m = graph_index_6m(
        x_data=df.index,
        y_data=df.year_index,
        title_text='Индекс контрактования: '
                   'Длинный год-год (6-и месячный прогноз)')

    return templates.TemplateResponse(
        "predict.html",
        context={"request": request,
                 'flag_load': flag_load,
                 'filename_load': filename_load,
                 'show_3m': show_3m,
                 'index_month_show_3m': index_month_show_3m,
                 'show_6m': show_6m,
                 'index_month_show_6m': index_month_show_6m,
                 'index_instant_show_3m': index_instant_show_3m,
                 'index_instant_show_6m': index_instant_show_6m,
                 'index_year_show_3m': index_year_show_3m,
                 'index_year_show_6m': index_year_show_6m
                 })


@app.get("/")
async def index(request: Request):

    return templates.TemplateResponse("index.html",
                                      context={"request": request,
                                               })
