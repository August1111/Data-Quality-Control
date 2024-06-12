import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from tensorflow.keras.models import load_model

# Боевые данные
# Шаг 1: Загрузка и предобработка данных (например, нормализация)
autoencoder = load_model('autoencoder_model.h5')

data = pd.read_csv('C:/Users/artem/YandexDisk/ккд/ml_kkd_flights_202406121639.csv')

# Преобразуем в даты
data['scheduled_departure'] = pd.to_datetime(data['scheduled_departure'])
data['scheduled_arrival'] = pd.to_datetime(data['scheduled_arrival'])
data['actual_departure'] = pd.to_datetime(data['actual_departure'])
data['actual_arrival'] = pd.to_datetime(data['actual_arrival'])

# Даты преобразуем в числа, секунды с 01 января 1970
data['scheduled_departure'] = data['scheduled_departure'].apply(lambda x: x.timestamp())
data['scheduled_arrival'] = data['scheduled_arrival'].apply(lambda x: x.timestamp())
data['actual_departure'] = data['actual_departure'].apply(lambda x: x.timestamp())
data['actual_arrival'] = data['actual_arrival'].apply(lambda x: x.timestamp())

# Разделение признаков на числовые и категориальные
numeric_features = ['flight_id','scheduled_departure', 'scheduled_arrival', 'actual_departure', 'actual_arrival']
categorical_features = ['flight_no','departure_airport','status','arrival_airport','aircraft_code']

# Создание трансформеров для числовых и категориальных данных
numeric_transformer = StandardScaler()
categorical_transformer = LabelEncoder()

# Применение преобразований к числовым данным
data_numeric = data[numeric_features]
data_numeric_scaled = numeric_transformer.fit_transform(data_numeric)

# Применение преобразований к категориальным данным
label_encoders = {}
data_categorical_encoded = data[categorical_features]

for col in categorical_features:
    label_encoders[col] = LabelEncoder()
    data_categorical_encoded[col] = label_encoders[col].fit_transform(data[col])

# Объединение числовых и категориальных данных после преобразований
data_preprocessed = np.hstack((data_numeric_scaled, data_categorical_encoded))


# Предсказание (реконструкция данных) для  данных
data_reconstructions = autoencoder.predict(data_preprocessed)

# Рассчет среднеквадратичной ошибки для каждого наблюдения
new_mse = np.mean(np.power(data_preprocessed - data_reconstructions, 2), axis=1)

# Определение порога для обнаружения аномалий на основе обучающих данных
threshold = np.percentile(new_mse, 99.9)  # Установка порога на уровне 95-го перцентиля ошибки

# Обнаружение аномалий в тестовых данных
anomalies = new_mse > threshold

anomalous_data = data_preprocessed[anomalies]
print("Аномальные наблюдения:\n", anomalous_data)
print("Индексы аномалий:\n", np.where(anomalies)[0])
