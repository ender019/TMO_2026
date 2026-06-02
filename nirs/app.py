import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Phone Price Predictor", layout="wide")

st.title("📱 Классификация ценового сегмента смартфонов")
st.write("Демонстрация работы модели SVC (Support Vector Classification)")

@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    return df

df = load_data()

# Настройки в боковой панели
st.sidebar.header("Параметры модели (Tuning)")
c_param = st.sidebar.slider("Регуляризация (C)", 0.1, 20.0, 10.0)
kernel_param = st.sidebar.selectbox("Ядро (Kernel)", ["linear", "rbf", "poly"])

# Подготовка данных
X = df.drop('price_range', axis=1)
y = df['price_range']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Обучение
model = SVC(C=c_param, kernel=kernel_param)
model.fit(X_train_scaled, y_train)

# Интерфейс предсказания
st.subheader("Введите характеристики телефона:")
col1, col2, col3 = st.columns(3)

with col1:
    ram = st.number_input("RAM (МБ)", 256, 4000, 2000)
    battery = st.number_input("Емкость батареи", 500, 2000, 1200)
with col2:
    px_height = st.number_input("Высота экрана (px)", 0, 2000, 500)
    px_width = st.number_input("Ширина экрана (px)", 500, 2000, 1000)
with col3:
    mobile_wt = st.number_input("Вес телефона", 80, 200, 150)
    int_memory = st.number_input("Внутренняя память (ГБ)", 2, 64, 32)

if st.button("📊 Предсказать цену"):
    # Берем средние значения для остальных признаков (которых нет в UI)
    input_data = X.mean().values.copy()
    feature_names = X.columns.tolist()
    
    # Заменяем значения на введенные пользователем
    input_dict = {
        "ram": ram, "battery_power": battery, 
        "px_height": px_height, "px_width": px_width,
        "mobile_wt": mobile_wt, "int_memory": int_memory
    }
    
    for i, name in enumerate(feature_names):
        if name in input_dict:
            input_data[i] = input_dict[name]
            
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)[0]
    
    price_labels = {0: "Бюджетный", 1: "Средний", 2: "Дорогой", 3: "Флагман"}
    st.success(f"Ценовой сегмент: **{price_labels[prediction]}**")