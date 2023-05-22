import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import Dense, Flatten, ConvLSTM1D
from func_for_predict import compile_and_fit, dataset_to_windows
from datetime import datetime
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


saled_cars = pd.read_csv('saled_df_daily.csv', index_col=0)
# делаем нормализацию данных
# приводим все данные к диапазону от 0 до 1
scaler = MinMaxScaler()
scaler.fit(saled_cars)
saled_cars[saled_cars.columns] = scaler.transform(saled_cars[saled_cars.columns])
saled_cars, test_ds = saled_cars[:-180], saled_cars[-180:]
x_train, y_train = dataset_to_windows(window_size=180, dataset=saled_cars)
x_train = x_train.reshape((x_train.shape[0], 1, 365, 1))
# print(x_train.shape, y_train.shape)

lstm = Sequential([
    ConvLSTM1D(filters=120, kernel_size=6, input_shape=(1, 365, 1)),
    Flatten(),
    Dense(300, activation='relu'),
    Dense(200, activation='relu'),
    Dense(180)
])

history_lstm = compile_and_fit(lstm, batch_size=365, features=x_train,
                               labels=y_train)


test = np.array(saled_cars[-365:]).copy()
# print(test)
forecast = lstm.predict(test.reshape(1, 1, 365, 1))
# forecast = lstm.predict(test.reshape(1, 365, 1))
forecast = forecast.reshape((180, 1))
forecast = pd.DataFrame(forecast)
# forecast['indx'] = test_ds.index
start_date = datetime.strptime(saled_cars.index[-1], '%Y-%m-%d').date() + dt.timedelta(days=1)
forecast['indx'] = [str(x).rstrip('00:00:00') for x in pd.date_range(start=start_date, periods=180)]
forecast.set_index('indx', inplace=True, drop=True)
forecast.index = forecast.index.astype('object')
# forecast['euro_rate'] = test_ds['euro_rate']
forecast.rename(columns={0: 'y'}, inplace=True)
saled_cars[saled_cars.columns] = scaler.inverse_transform(saled_cars[saled_cars.columns])
forecast[forecast.columns] = scaler.inverse_transform(forecast[forecast.columns])
test_ds[test_ds.columns] = scaler.inverse_transform(test_ds[test_ds.columns])
forecast['y'] = forecast['y'].round(decimals=0)

print(forecast['y'].sum())

lstm.save('conv1d_lstm_20_05_2023.h5')

fig = plt.figure(figsize=(15, 8))

x = np.arange(15)

labels = list(forecast.index[:15])
plt.bar(x - 0.15, test_ds['y'][110:125], width=0.3, color='green',
        label='Реальные')
plt.bar(x + 0.15, forecast['y'][110:125], width=0.3, color='red',
        label='Прогноз')

for index, value in enumerate(test_ds['y'][110:125]):
    plt.text(x=index - 0.15, y=value + 0.15, s=str(round(value)), ha='center',
             fontsize=16)

for index, value in enumerate(forecast['y'][110:125]):
    plt.text(x=index + 0.15, y=value + 0.15, s=str(round(value)), ha='center',
             fontsize=16)

plt.xlabel('Даты продаж', fontsize=16)
plt.ylabel('Кол-во машин', fontsize=16)
plt.xticks(ticks=x, labels=labels)
plt.title(f'Сравнение реальных и прогнозных продаж в абсолютном выражении',
          fontsize=16)
plt.legend(fontsize=16)

plt.show()
