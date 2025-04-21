import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def classification_page():
    st.title("Clasificación")
    st.write("Clasificación de inmuebles en el cluster que le corresponde para análisis inmobiliario.")

    # Cargar los datos desde CSV
    df = pd.read_csv("data/compras_completo_limpio.csv")
    df_clusters = pd.read_csv("data/compras_clusters.csv")

    df["latitud"] = df["coordenadas"].str.split(",").str[0].str.strip().astype(float, errors='ignore')
    df["longitud"] = df["coordenadas"].str.split(",").str[1].str.strip().astype(float, errors='ignore')

    # Eliminar valores atípicos
    df = df[
        (df["superficie"].between(df["superficie"].quantile(0.05), df["superficie"].quantile(0.95))) &
        (df["superficie_util"].between(df["superficie_util"].quantile(0.01), df["superficie_util"].quantile(0.99))) &
        (df["precio"].between(df["precio"].quantile(0.02), df["precio"].quantile(0.98)))
    ]

    # Input del usuario
    input_data = {}
    columnas_clustering = ["precio", "antiguedad", "conservacion", "latitud", "longitud", "piscina", "baños",
                           "chimenea", "habitaciones", "superficie", "superficie_util", "garaje", "provincia",
                           "planta", "orientacion"]

    input_data['precio'] = st.slider("Introduce el precio", min_value=int(df['precio'].min()),
                                     max_value=int(df['precio'].max()), step=10)
    input_data['antiguedad'] = st.selectbox("Introduce la antigüedad", options=df['antiguedad'].unique())
    input_data['conservacion'] = st.selectbox("Introduce la conservación", options=df['conservacion'].unique())
    input_data['latitud'] = st.number_input("Introduce la latitud", min_value=float(df['latitud'].min()),
                                            max_value=float(df['latitud'].max()), format="%.6f")
    input_data['longitud'] = st.number_input("Introduce la longitud", min_value=float(df['longitud'].min()),
                                             max_value=float(df['longitud'].max()), format="%.6f")
    input_data['piscina'] = st.selectbox("¿Tiene piscina?", options=df['piscina'].unique())
    input_data['baños'] = st.slider("Introduce el número de baños", min_value=int(df['baños'].min()),
                                    max_value=int(df['baños'].max()), step=1)
    input_data['chimenea'] = st.selectbox("¿Tiene chimenea?", options=df['chimenea'].unique())
    input_data['habitaciones'] = st.slider("Introduce el número de habitaciones",
                                           min_value=int(df['habitaciones'].min()),
                                           max_value=int(df['habitaciones'].max()), step=1)
    input_data['superficie'] = st.slider("Introduce la superficie", min_value=int(df['superficie'].min()),
                                         max_value=int(df['superficie'].max()), step=10)
    input_data['superficie_util'] = st.slider("Introduce la superficie útil",
                                              min_value=int(df['superficie_util'].min()),
                                              max_value=int(df['superficie_util'].max()), step=10)
    input_data['garaje'] = st.selectbox("¿Tiene garaje?", options=df['garaje'].unique())
    input_data['provincia'] = st.selectbox("Introduce la provincia", options=df['provincia'].unique())
    input_data['planta'] = st.slider("Introduce la planta", min_value=int(df['planta'].min()),
                                     max_value=int(df['planta'].max()), step=1)
    input_data['orientacion'] = st.selectbox("Introduce la orientación", options=df['orientacion'].unique())

    if st.button("Predecir Cluster"):
        input_df = pd.DataFrame([input_data])

        # Variables categóricas
        cat_cols = ['antiguedad', 'conservacion', 'piscina', 'chimenea', 'garaje', 'provincia', 'orientacion']
        for col in cat_cols:
            df[col] = df[col].astype(str)
            df_clusters[col] = df_clusters[col].astype(str)

        # Codificar input y clusters
        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(pd.concat([df[col], df_clusters[col]]))
            input_df[col] = le.transform(input_df[col])
            df_clusters[col] = le.transform(df_clusters[col])
            label_encoders[col] = le

        # Escalado
        scaler = StandardScaler()
        input_df_scaled = scaler.fit_transform(input_df)

        # Entrenamiento del modelo
        X = df_clusters[columnas_clustering]
        y = df_clusters['cluster']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predicción
        pred = model.predict(input_df_scaled)
        st.write(f"El inmueble pertenece al cluster: {pred[0]}")

        # Precisión
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"Precisión del modelo: {acc:.2f}")

if __name__ == "__main__":
    classification_page()
