import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

#Обучение
# Шаг 1: Загрузка и предобработка данных (например, нормализация)
data = pd.read_csv('C:/Users/artem/YandexDisk/ккд/ml_kkd_flights_202406121435.csv')

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

# Шаг 2: Разделение данных на обучающую и тестовую выборки
train_data, test_data = train_test_split(data_preprocessed, test_size=0.2, random_state=42)

# Шаг 3: Определение архитектуры автокодировщика
input_dim = data_preprocessed.shape[1]
encoding_dim = 5  # Размер узкого места (можно варьировать)

# Входной слой
input_layer = Input(shape=(input_dim,))

# Энкодер
encoder = Dense(10, activation='relu')(input_layer)
encoder = Dense(encoding_dim, activation='relu')(encoder)  # Узкое место

# Декодер
decoder = Dense(10, activation='relu')(encoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder)

# Создание и компиляция модели автокодировщика
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')

# Шаг 4: Обучение автокодировщика
autoencoder.fit(train_data, train_data, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)

# Применение модели на тестовых данных для оценки производительности
# Предсказание (реконструкция данных) для обучающих и тестовых данных
train_reconstructions = autoencoder.predict(train_data)
test_reconstructions = autoencoder.predict(test_data)

# Рассчет среднеквадратичной ошибки для каждого наблюдения
train_mse = np.mean(np.power(train_data - train_reconstructions, 2), axis=1)
test_mse = np.mean(np.power(test_data - test_reconstructions, 2), axis=1)

# Определение порога для обнаружения аномалий на основе обучающих данных
threshold = np.percentile(train_mse, 99.999)  # Установка порога на уровне 95-го перцентиля ошибки

# Сохранение модели и порога
autoencoder.save('autoencoder_model.h5')
#joblib.dump(threshold, 'anomaly_threshold.pkl')

