import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import mysql.connector
import numpy as np

def classification_page():
    st.title("Clasificación")
    st.write("Clasificación de inmuebles en el cluster que le corresponde para análisis inmobiliario.")

    # Conexión a la base de datos MySQL
    conn = mysql.connector.connect(
        host=st.secrets["database"]["host"],
        user=st.secrets["database"]["user"],
        password=st.secrets["database"]["password"],
        database=st.secrets["database"]["name"],
        auth_plugin='mysql_native_password'
    )
    query = "SELECT * FROM general_alquileres"
    df = pd.read_sql(query, conn)

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

    if st.button("Predecir Cluster"):
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

        # Escalar los datos de entrada
        scaler = StandardScaler()
        input_df_escalado = scaler.fit_transform(input_df)

        # Dividir en datos de entrenamiento y prueba
        X = df_clusters[columnas_clustering]
        y = df_clusters['cluster']

        # Codificar las variables categóricas en el DataFrame original
        for column in ['antiguedad', 'conservacion', 'piscina', 'chimenea', 'garaje', 'provincia', 'orientacion']:
            df_clusters[column] = label_encoders[column].transform(df_clusters[column])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar el modelo con los datos y clusters
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Realizar la predicción
        prediction = model.predict(input_df_escalado)
        st.write(f"El inmueble pertenece al cluster: {prediction[0]}")

        # Evaluar el modelo
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f'Precisión del modelo: {accuracy:.2f}')

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

if __name__ == "__main__":
    classification_page()