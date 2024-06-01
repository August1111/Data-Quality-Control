import numpy as np
import pandas as pd
import openpyxl
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

# Пример загрузки данных из CSV файла
data = pd.read_csv('C:/Users/artem/YandexDisk/ккд/flights_v_202406012145.csv')

# Преобразуем в даты
data['scheduled_departure'] = pd.to_datetime(data['scheduled_departure'])
data['scheduled_arrival'] = pd.to_datetime(data['scheduled_arrival'])
data['scheduled_duration'] = pd.to_timedelta(data['scheduled_duration'])
data['actual_departure'] = pd.to_datetime(data['actual_departure'])
data['actual_arrival'] = pd.to_datetime(data['actual_arrival'])
data['actual_duration'] = pd.to_timedelta(data['actual_duration'])


# Даты преобразуем в числа, секунды с 01 января 1970
data['scheduled_departure_ml'] = data['scheduled_departure'].apply(lambda x: x.timestamp())
data['scheduled_arrival_ml'] = data['scheduled_arrival'].apply(lambda x: x.timestamp())
data['scheduled_duration_ml'] = data['scheduled_duration'].dt.total_seconds()
data['actual_departure_ml'] = data['actual_departure'].apply(lambda x: x.timestamp())
data['actual_arrival_ml'] = data['actual_arrival'].apply(lambda x: x.timestamp())
data['actual_duration_ml'] = data['actual_duration'].dt.total_seconds()

# Разделение признаков на числовые и категориальные
numeric_features = ['scheduled_departure_ml', 'scheduled_arrival_ml', 'scheduled_duration_ml', 'actual_departure_ml', 'actual_arrival_ml', 'actual_duration_ml']
categorical_features = ['departure_airport','arrival_airport','aircraft_code']

# Создание трансформеров для числовых и категориальных данных
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

# Применение преобразований к числовым данным
data_numeric = data[numeric_features]
data_numeric_scaled = numeric_transformer.fit_transform(data_numeric)

# Применение преобразований к категориальным данным
data_categorical = data[categorical_features]
data_categorical_encoded = categorical_transformer.fit_transform(data_categorical).toarray()

# Объединение числовых и категориальных данных после преобразований
data_preprocessed = np.hstack((data_numeric_scaled, data_categorical_encoded))

# Обучение модели Isolation Forest
model = IsolationForest(contamination=0.03, random_state=42)
model.fit(data_preprocessed)

# Прогнозирование аномалий
data['anomaly'] = model.predict(data_preprocessed)

# # Визуализация аномалий - пока выключим
# plt.figure(figsize=(10, 6))
# colors = {1: 'blue', -1: 'red'}
# plt.scatter(data.index, data['actual_duration_ml'], c=data['anomaly'].map(colors))
# plt.xlabel('aircraft_code')
# plt.ylabel('actual_duration_ml')
# plt.title('Anomaly Detection using Isolation Forest')
# plt.show()

data_result = data [['scheduled_departure_ml', 'scheduled_arrival_ml', 'scheduled_duration_ml', 'actual_departure_ml'
    , 'actual_arrival_ml', 'actual_duration_ml','departure_airport','arrival_airport','aircraft_code','anomaly', 'flight_no']]
#
data_result.loc[data_result['anomaly'] == -1].to_excel('anomaly.xlsx', index=False)

# pd.set_option('display.max_columns', None)
# print(data_result.loc[data_result['anomaly'] == -1].sort_values(by='flight_no'))

