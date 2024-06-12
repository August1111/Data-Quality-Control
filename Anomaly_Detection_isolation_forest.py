import numpy as np
import pandas as pd
import openpyxl
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt

# Пример загрузки данных из CSV файла
# Isolation forest попытка №1 (два признака) В рейсе PG0013  за март 2017  у  10 полётов  увеличено реальное время прилёта  на 4 часа.
# data = pd.read_csv('C:/Users/artem/YandexDisk/ккд/ml_kkd_2_202406092045.csv')
#
# # Преобразуем в даты
# data['scheduled_departure'] = pd.to_datetime(data['scheduled_departure'])
# data['scheduled_arrival'] = pd.to_datetime(data['scheduled_arrival'])
# data['actual_departure'] = pd.to_datetime(data['actual_departure'])
# data['actual_arrival'] = pd.to_datetime(data['actual_arrival'])
#
#
#
# # Даты преобразуем в числа, секунды с 01 января 1970
# data['scheduled_departure_ml'] = data['scheduled_departure'].apply(lambda x: x.timestamp())
# data['scheduled_arrival_ml'] = data['scheduled_arrival'].apply(lambda x: x.timestamp())
# data['actual_departure_ml'] = data['actual_departure'].apply(lambda x: x.timestamp())
# data['actual_arrival_ml'] = data['actual_arrival'].apply(lambda x: x.timestamp())
#
# # Разделение признаков на числовые и категориальные
# # numeric_features = ['scheduled_departure_ml', 'scheduled_arrival_ml', 'scheduled_duration_ml', 'actual_departure_ml', 'actual_arrival_ml', 'actual_duration_ml']
# # categorical_features = ['departure_airport','arrival_airport','aircraft_code']
#
# numeric_features = ['actual_arrival']
# categorical_features = ['flight_no']
#
# # Создание трансформеров для числовых и категориальных данных
# numeric_transformer = StandardScaler()
# categorical_transformer = OneHotEncoder()
#
#
# # Применение преобразований к числовым данным
# data_numeric = data[numeric_features]
# data_numeric_scaled = numeric_transformer.fit_transform(data_numeric)
#
# # Применение преобразований к категориальным данным
# data_categorical = data[categorical_features]
# data_categorical_encoded = categorical_transformer.fit_transform(data_categorical).toarray()
#
# # Объединение числовых и категориальных данных после преобразований
# data_preprocessed = np.hstack((data_numeric_scaled, data_categorical_encoded))
#
# # Обучение модели Isolation Forest
# model = IsolationForest(contamination=0.03, random_state=42)
# model.fit(data_preprocessed)
#
# # Прогнозирование аномалий
# data['anomaly'] = model.predict(data_preprocessed)
#
# # # Визуализация аномалий - пока выключим
# # plt.figure(figsize=(10, 6))
# # colors = {1: 'blue', -1: 'red'}
# # plt.scatter(data.index, data['actual_duration_ml'], c=data['anomaly'].map(colors))
# # plt.xlabel('aircraft_code')
# # plt.ylabel('actual_duration_ml')
# # plt.title('Anomaly Detection using Isolation Forest')
# # plt.show()
#
# data_result = data [['scheduled_departure_ml', 'scheduled_arrival_ml', 'actual_departure_ml'
#     , 'actual_arrival_ml','departure_airport','arrival_airport','aircraft_code','anomaly', 'flight_no']]
# #
# data_result.loc[data_result['anomaly'] == -1].to_excel('anomaly.xlsx', index=False)
#
# # pd.set_option('display.max_columns', None)
# # print(data_result.loc[data_result['anomaly'] == -1].sort_values(by='flight_no'))

# Isolation forest попытка №2 (три признака) В рейсе PG0013  за март 2017  у  10 полётов  увеличено реальное время прилёта  на 4 часа.Загружаем только 3 показателя отлёт, прилёт, номер самолёта (объём данных в районе 9 мб)
# data = pd.read_csv('C:/Users/artem/YandexDisk/ккд/_select_actual_departure_actual_arrival_flight_no_from_public_ml_202406092145.csv')
#
# # Преобразуем в даты
# data['actual_departure_ml'] = pd.to_datetime(data['actual_departure'])
# data['actual_arrival_ml'] = pd.to_datetime(data['actual_arrival'])
#
# data['actual_departure_ml_2'] = data['actual_departure_ml'].apply(lambda x: x.timestamp())
# data['actual_arrival_ml_2'] = data['actual_arrival_ml'].apply(lambda x: x.timestamp())
#
# # Разделение признаков на числовые и категориальные
# numeric_features = ['actual_departure_ml_2','actual_arrival_ml_2']
# categorical_features = ['flight_no']
#
# # Создание трансформеров для числовых и категориальных данных
# numeric_transformer = StandardScaler()
# categorical_transformer = OneHotEncoder()
#
# # Применение преобразований к числовым данным
# data_numeric = data[numeric_features]
# data_numeric_scaled = numeric_transformer.fit_transform(data_numeric)
#
# # Применение преобразований к категориальным данным
# data_categorical = data[categorical_features]
# data_categorical_encoded = categorical_transformer.fit_transform(data_categorical).toarray()
#
# # Объединение числовых и категориальных данных после преобразований
# data_preprocessed = np.hstack((data_numeric_scaled, data_categorical_encoded))
#
# # Обучение модели Isolation Forest
# model = IsolationForest(contamination=0.015, random_state=42)
# model.fit(data_preprocessed)
#
# # Прогнозирование аномалий
# data['anomaly'] = model.predict(data_preprocessed)
#
# data_result = data [['actual_departure'
#     , 'actual_arrival','flight_no','anomaly']]
# #
# data_result.loc[data_result['anomaly'] == -1].to_excel('anomaly.xlsx', index=False)

# # Isolation forest попытка №3 (два признака) В рейсе PG0013  за март 2017  у  10 полётов  увеличено реальное время прилёта  на 4 часа.Загружаем только 3 показателя отлёт, прилёт, номер са
# data = pd.read_csv('C:/Users/artem/YandexDisk/ккд/_select_actual_arrival_actual_departure_as_length_flight_no_from_202406092336.csv')
#
# data['length_sec'] = pd.to_timedelta(data['length'])
#
# # Шаг 3: Применение функции к колонке времени
# data['length_sec'] = data['length_sec'].dt.total_seconds().astype(int)
#
# # Шаг 2: Преобразование номеров рейсов в числовой формат
# label_encoder = LabelEncoder()
# data['flight_number_encoded'] = label_encoder.fit_transform(data['flight_no'])
#
# # Шаг 3: Обучение модели Isolation Forest на всех данных
# model = IsolationForest(
#     contamination=0.003,  # Установите долю аномалий
#     n_estimators=150,  # Увеличьте количество деревьев для улучшения стабильности
#     max_samples=0.8,  # Используйте 80% выборок для каждого дерева
#     max_features=1.0,  # Используйте все признаки для каждого дерева
#     random_state=42
# )
# model.fit(data[['length_sec', 'flight_number_encoded']])
#
# # Прогнозирование аномалий
# data['anomaly'] = model.predict(data[['length_sec', 'flight_number_encoded']])
#
# # print(data.loc[data['anomaly'] == -1])
# print(data.loc[data['flight_no'] == 'PG0013'])


# Isolation forest попытка №4 ( три признака) В рейсе PG0013  за март 2017  у  10 полётов  увеличено реальное время прилёта  на 4 часа.Загружаем только 3 показателя отлёт, прилёт, номер са
data = pd.read_csv('C:/Users/artem/YandexDisk/ккд/_select_actual_arrival_date_actual_arrival_actual_departure_as_l_202406100018.csv')
data['actual_arrival_ml'] = pd.to_datetime(data['actual_arrival'])
data['length_sec'] = pd.to_timedelta(data['length'])

# Шаг 3: Применение функции к колонке времени
data['length_sec'] = data['length_sec'].dt.total_seconds().astype(int)
data['actual_arrival_ml_2'] = data['actual_arrival_ml'].apply(lambda x: x.timestamp())

# Шаг 2: Преобразование номеров рейсов в числовой формат
label_encoder = LabelEncoder()
data['flight_number_encoded'] = label_encoder.fit_transform(data['flight_no'])

# Шаг 3: Обучение модели Isolation Forest на всех данных
model = IsolationForest(
    contamination=0.00007,  # Установите долю аномалий
    n_estimators=150,  # Увеличьте количество деревьев для улучшения стабильности
    max_samples=0.8,  # Используйте 80% выборок для каждого дерева
    max_features=1.0,  # Используйте все признаки для каждого дерева
    random_state=42
)
model.fit(data[['actual_arrival_ml_2','length_sec', 'flight_number_encoded']])

# Прогнозирование аномалий
data['anomaly'] = model.predict(data[['actual_arrival_ml_2','length_sec', 'flight_number_encoded']])

data_result = data [['actual_arrival','length', 'flight_no', 'anomaly']]

print(data_result.loc[data_result['anomaly'] == -1])
# print(data.loc[data['flight_no'] == 'PG0013'])

