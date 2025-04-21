import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import mysql.connector
import plotly.express as px
import pickle
import numpy as np

def regression_page():
    st.title("Regresión")
    st.write("Predicción del precio de inmuebles basado en sus características y cluster.")

    # Conexión a la base de datos MySQL
    df = pd.read_csv("data/alquileres_completo_limpio.csv")

    df["latitud"] = df["coordenadas"].str.split(",").str[0].str.strip().astype(float, errors='ignore')
    df["longitud"] = df["coordenadas"].str.split(",").str[1].str.strip().astype(float, errors='ignore')

    # Eliminar valores atípicos en la superficie (percentil 1 y 99)
    q1_superficie = df["superficie"].quantile(0.03)
    q99_superficie = df["superficie"].quantile(0.97)
    df = df[(df["superficie"] >= q1_superficie) & (df["superficie"] <= q99_superficie)]

    # Eliminar valores atípicos en la superficie_util (percentil 1 y 99)
    q1_superficie = df["superficie_util"].quantile(0.01)
    q99_superficie = df["superficie_util"].quantile(0.99)
    df = df[(df["superficie_util"] >= q1_superficie) & (df["superficie_util"] <= q99_superficie)]

    # Eliminar valores atípicos en el precio (percentil 1 y 99)
    q1_precio = df["precio"].quantile(0.05)
    q99_precio = df["precio"].quantile(0.95)
    df = df[(df["precio"] >= q1_precio) & (df["precio"] <= q99_precio)]

    query = "SELECT * FROM alquiler_clusters"
    df_clusters = pd.read_sql(query, conn)
    conn.close()

    clusters = df_clusters['cluster']

    # Introducción de variables
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

    if st.button("Predecir Precio"):
        # Convertir input_data a DataFrame
        input_df = pd.DataFrame([input_data])

        # Convertir todas las columnas categóricas a cadenas
        for column in ['antiguedad', 'conservacion', 'piscina', 'chimenea', 'garaje', 'provincia', 'orientacion']:
            df[column] = df[column].astype(str)
            df_clusters[column] = df_clusters[column].astype(str)

        # Codificar las variables categóricas en input_df
        label_encoders = {}
        for column in ['antiguedad', 'conservacion', 'piscina', 'chimenea', 'garaje', 'provincia', 'orientacion']:
            le = LabelEncoder()
            le.fit(pd.concat([df[column], df_clusters[column]]))  # Ajustar con todas las etiquetas posibles
            input_df[column] = le.transform(input_df[column])
            label_encoders[column] = le

        # Escalar los datos de entrada con el mismo scaler utilizado para los datos de entrenamiento
        scaler = StandardScaler()
        scaler.fit(df_clusters[columnas_clustering])  # Ajustar el scaler con los datos de entrenamiento
        input_df_escalado = scaler.transform(input_df)
        input_df_escalado = pd.DataFrame(input_df_escalado,
                                         columns=input_df.columns)  # Mantener los nombres de las características

        # Dividir en datos de entrenamiento y prueba
        X = df_clusters[columnas_clustering]
        y = df_clusters['cluster']

        # Codificar las variables categóricas en el DataFrame original
        for column in ['antiguedad', 'conservacion', 'piscina', 'chimenea', 'garaje', 'provincia', 'orientacion']:
            df_clusters[column] = label_encoders[column].transform(df_clusters[column])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar el modelo con los datos y clusters
        modelo_cluster = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo_cluster.fit(X_train, y_train)

        # Realizar la predicción del cluster
        cluster_predicho = modelo_cluster.predict(input_df_escalado)[0]
        st.write(f"El inmueble pertenece al cluster: {cluster_predicho}")

        # Filtrar datos del cluster predicho
        df_cluster = df_clusters[df_clusters["cluster"] == cluster_predicho]

        # Variables predictoras y objetivo
        X = df_cluster.drop(columns=["precio", "cluster"])
        y = df_cluster["precio"]

        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Crear y entrenar el modelo de regresión
        modelo_regresion = RandomForestRegressor(n_estimators=100, random_state=42)
        modelo_regresion.fit(X_train, y_train)

        # Realizar la predicción del precio
        input_df_escalado = scaler.transform(input_df)  # Escalar los datos de entrada nuevamente
        input_df_escalado = pd.DataFrame(input_df_escalado,
                                         columns=X_train.columns)  # Asegurarse de que las columnas coincidan
        precio_predicho = modelo_regresion.predict(input_df_escalado)[0]
        st.write(f"El precio estimado del inmueble es: {precio_predicho:.2f} €")

        # Evaluar el modelo
        y_pred = modelo_regresion.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        medae = median_absolute_error(y_test, y_pred)
        st.write(f'MedAE: {medae:.2f}')

        # Aplicar logaritmo al precio
        #df_clusters["preciolog"] = np.log1p(df_clusters["precio"])

        # Crear el gráfico con Plotly Express
        #fig = px.scatter(df_clusters, x="superficie", y="preciolog", color="cluster",
        #                 title="Distribución de Precios vs Superficie por Cluster",
        #                 labels={"superficie": "Superficie (m²)", "preciolog": "Precio (€)(log)"},
        #                 opacity=0.5, color_continuous_scale="viridis")

        # Añadir el punto del input del usuario al gráfico
        #input_df["preciolog"] = np.log1p(input_df["precio"])
        #fig.add_scatter(x=input_df["superficie"], y=input_df["preciolog"], mode='markers', name='Input del Usuario',
        #                marker=dict(color='red', size=10))

        # Mostrar el gráfico en Streamlit
        #st.plotly_chart(fig)

        # Guardar el modelo
        with open(f"data/modelo_regresion_alq_cluster_{cluster_predicho}.pkl", "wb") as archivo:
            pickle.dump(modelo_regresion, archivo)

if __name__ == "__main__":
    regression_page()