from plotly import graph_objs as go
from plotly.utils import PlotlyJSONEncoder
import json
import datetime as dt
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import calendar


def data_predict(filename, model):

    df = pd.read_csv(filename, index_col='ds', parse_dates=True)
    # print(df.index.year)
    years_list = list(df.index.year.unique())
    days_in_month_dict = {'leap_year': {
        1: 31, 2: 29, 3: 31, 4: 30,
        5: 31, 6: 30, 7: 31, 8: 31,
        9: 30, 10: 31, 11: 30, 12: 31
    },
        'usual_year': {
            1: 31, 2: 28, 3: 31, 4: 30,
            5: 31, 6: 30, 7: 31, 8: 31,
            9: 30, 10: 31, 11: 30, 12: 31
        }
    }

    for each_year in years_list:
        if calendar.isleap(each_year):
            for each_month in days_in_month_dict['leap_year'].keys():
                for each_day in range(1, days_in_month_dict['leap_year'][
                                             each_month] + 1):
                    saled_cars = df[
                        (df.index.year.isin([each_year])) &
                        (df.index.month.isin([each_month])) &
                        (df.index.day.isin([each_day]))
                        ].shape[0]
                    if saled_cars == 0:
                        df_date = f'{each_year}-{each_month}-{each_day}'
                        df_date = pd.to_datetime(df_date)
                        df.loc[df_date] = [saled_cars]

        else:
            for each_month in days_in_month_dict['usual_year'].keys():
                for each_day in range(1, days_in_month_dict['usual_year'][
                                             each_month] + 1):
                    saled_cars = df[
                        (df.index.year.isin([each_year])) &
                        (df.index.month.isin([each_month])) &
                        (df.index.day.isin([each_day]))
                        ].shape[0]
                    if saled_cars == 0:
                        df_date = f'{each_year}-{each_month}-{each_day}'
                        df_date = pd.to_datetime(df_date)
                        df.loc[df_date] = [saled_cars]

    # df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.to_csv(filename)

    df = pd.read_csv(filename, index_col='ds')

    scaler = MinMaxScaler()
    scaler.fit(df)
    df[df.columns] = scaler.transform(df[df.columns])
    arr_for_pred = np.array(df[-365:]).copy()
    forecast = model.predict(arr_for_pred.reshape(1, 1, 365, 1))

    forecast = forecast.reshape((180, 1))
    forecast = pd.DataFrame(forecast)
    # start_date = df.index[-1] + dt.timedelta(days=1)
    start_date = datetime.strptime(df.index[-1],
                                   '%Y-%m-%d').date() + dt.timedelta(days=1)
    forecast['indx'] = [str(x).rstrip('00:00:00')[:-1] for x in
                        pd.date_range(start=start_date, periods=180)]

    forecast.set_index('indx', inplace=True, drop=True)
    forecast.index = forecast.index.astype('object')

    forecast.rename(columns={0: 'y'}, inplace=True)
    df[df.columns] = scaler.inverse_transform(
        df[df.columns])
    forecast[forecast.columns] = scaler.inverse_transform(
        forecast[forecast.columns])
    forecast['y'] = forecast['y'].round(decimals=0)

    df = pd.concat([df, forecast])

    return df


def graph_3m(x_data, y_data):

    sale_line_3m = go.Scatter(
            x=x_data[-450:-180],
            y=y_data[-450:-180],
            name='Продажи',
        )

    sale_line_3m.update(line_width=3)

    predict_line_3m = go.Scatter(
        x=x_data[-181:-90],
        y=y_data[-181:-90],
        name='Прогноз'
    )

    predict_line_3m.update(line_width=3)

    graphs_3m_layout = go.Layout(
        font={
            'family': 'Courier New, monospace',
            'size': 18
        },
        title={
            'text': 'Контракты (прогноз на 3 месяца)',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    graphs_3m_data = [
        sale_line_3m,
        predict_line_3m
    ]

    graphs_3m = go.Figure(
        data=graphs_3m_data,
        layout=graphs_3m_layout
    )

    show_3m = json.dumps(graphs_3m, cls=PlotlyJSONEncoder)

    return show_3m


def graph_6m(x_data, y_data):

    sale_line_6m = go.Scatter(
                x=x_data[-360:-180],
                y=y_data[-360:-180],
                name='Продажи',
            )

    sale_line_6m.update(line_width=3)

    predict_line_6m = go.Scatter(
            x=x_data[-181:],
            y=y_data[-181:],
            name='Прогноз'
        )

    predict_line_6m.update(line_width=3)

    graphs_6m_layout = go.Layout(
        font={
            'family': 'Courier New, monospace',
            'size': 18
        },
        title={
            'text': 'Контракты (прогноз на 6 месяцев)',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    graphs_6m_data = [
        sale_line_6m,
        predict_line_6m
    ]

    graphs_6m = go.Figure(
        data=graphs_6m_data,
        layout=graphs_6m_layout
    )

    show_6m = json.dumps(graphs_6m, cls=PlotlyJSONEncoder)
    return show_6m


def graph_index_3m(x_data, y_data,
                   title_text):
    month_index_line_3m = go.Scatter(
        x=x_data[-450:-180],
        y=y_data[-450:-180],
        name='Индекс продаж',
    )

    month_index_line_3m.update(line_width=3)

    predict_index_line_3m = go.Scatter(
        x=x_data[-181:-90],
        y=y_data[-181:-90],
        name='Индекс прогноза'
    )

    predict_index_line_3m.update(line_width=3)

    index_graphs_3m_layout = go.Layout(
        font={
            'family': 'Courier New, monospace',
            'size': 18
        },
        title={
            'text': title_text,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    index_graphs_3m_data = [
        month_index_line_3m,
        predict_index_line_3m
    ]

    index_graphs_3m = go.Figure(
        data=index_graphs_3m_data,
        layout=index_graphs_3m_layout
    )

    index_show_3m = json.dumps(index_graphs_3m, cls=PlotlyJSONEncoder)

    return index_show_3m


def graph_index_6m(x_data, y_data,
                   title_text):

    month_index_line_6m = go.Scatter(
        x=x_data[-360:-180],
        y=y_data[-360:-180],
        name='Индекс продаж',
    )

    month_index_line_6m.update(line_width=3)

    predict_index_line_6m = go.Scatter(
        x=x_data[-181:],
        y=y_data[-181:],
        name='Индекс прогноза'
    )

    predict_index_line_6m.update(line_width=3)

    index_graphs_6m_layout = go.Layout(
        font={
            'family': 'Courier New, monospace',
            'size': 18
        },
        title={
            'text': title_text,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    index_graphs_6m_data = [
        month_index_line_6m,
        predict_index_line_6m
    ]

    index_graphs_6m = go.Figure(
        data=index_graphs_6m_data,
        layout=index_graphs_6m_layout
    )

    index_show_6m = json.dumps(index_graphs_6m, cls=PlotlyJSONEncoder)

    return index_show_6m


def data_for_index(df):
    df = df
    df['ma_instant'] = df['y'].rolling(3).sum()
    df['ma_month'] = df['y'].rolling(30).sum()
    df['ma_year'] = df['y'].rolling(365).sum()

    date_list = ['ma_instant', 'ma_month', 'ma_year']

    for roll_date in date_list:
        for index_date in df[roll_date][365:].index:

            index_date = datetime.strptime(index_date, '%Y-%m-%d').date()
            old_index_date = str(index_date - dt.timedelta(weeks=52, days=1))
            old_index_date = df.index.get_loc(old_index_date)
            df[f'{roll_date}_prev_year'] = df[roll_date][old_index_date]
            df[f'{roll_date[3:]}_index'] = df[roll_date] / df[
                f'{roll_date}_prev_year'] * 100
            df[f'{roll_date[3:]}_index'] = df[
                f'{roll_date[3:]}_index'].round(decimals=0)

    return df
