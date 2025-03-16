import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import mysql.connector

from app.pages.database import cargar_datos_bd

def clustering_page():
    st.title("Clustering")
    st.write("Preparación de clusters para análisis inmobiliario.")

    # Cargar datos desde la base de datos
    query = "SELECT * FROM general_compras"
    df = cargar_datos_bd(query)

    #df.drop(columns=["superficie_solar"], inplace=True)

    st.dataframe(df)

    # Eliminar valores atípicos en la superficie (percentil 1 y 99)
    q1_superficie = df["superficie"].quantile(0.05)
    q99_superficie = df["superficie"].quantile(0.95)
    df = df[(df["superficie"] >= q1_superficie) & (df["superficie"] <= q99_superficie)]

    # Eliminar valores atípicos en la superficie_util (percentil 1 y 99)
    q1_superficie = df["superficie_util"].quantile(0.01)
    q99_superficie = df["superficie_util"].quantile(0.99)
    df = df[(df["superficie_util"] >= q1_superficie) & (df["superficie_util"] <= q99_superficie)]

    # Eliminar valores atípicos en el precio (percentil 1 y 99)
    q5_precio = df["precio"].quantile(0.02)
    q95_precio = df["precio"].quantile(0.98)
    df = df[(df["precio"] >= q5_precio) & (df["precio"] <= q95_precio)]

    df["antiguedad"] = df["antiguedad"].replace("Sin especificar", "10-20")
    df["conservacion"] = df["conservacion"].replace("Sin especificar", "En buen estado")

    df["latitud"] = df["coordenadas"].str.split(",").str[0].str.strip().astype(float, errors='ignore')
    df["longitud"] = df["coordenadas"].str.split(",").str[1].str.strip().astype(float, errors='ignore')

    # Filtrar las columnas numéricas del DataFrame original
    columnas_numericas = df.select_dtypes(include=['number']).columns

    # Crear el mapa de correlación con Plotly
    corr_matrix = df[columnas_numericas].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    fig.update_layout(title='Mapa de correlación entre las columnas numéricas',
        width=800,
        height=600
    )

    st.plotly_chart(fig)

    # Calculamos la correlación de cada columna con 'precio'
    corr_with_target = df[columnas_numericas].corr()['precio'].drop('precio')

    # Convertimos la serie a un DataFrame para facilitar la visualización
    corr_df = corr_with_target.reset_index()
    corr_df.columns = ['Columna', 'Correlación']
    corr_df = corr_df.sort_values(by='Correlación', ascending=True)

    # Creamos el gráfico
    fig = px.bar(corr_df, x='Columna', y='Correlación', color='Correlación',
                 color_continuous_scale='RdBu_r', range_color=[-1, 1])
    fig.update_layout(
        title='Correlación de las columnas numéricas con la objetivo "precio"',
        width=800,
        height=600
    )

    st.plotly_chart(fig)

    ##REVISAR
    #st.write("A simple vista podemos observar que las columnas numéricas con mayor relación al precio "
    #         "son las superficies que pueden ser redundantes al proporcionar el mismo tipo de información "
    #         "y por el contrario las que menos serian planta, precio por metro cuadrado que en principio"
    #         "deberia ser relevante y habitaciones")


    st.subheader("Analisis de algunas categorias cualitativas y su influencia en el precio")

    # Crear diagramas de cajas para las columnas especificadas en relación con el precio
    columnas_cualitativas_influyentes = ["piscina", "garaje", "chimenea"]

    # Sustituir los valores "Sin especificar" por "no" en las columnas "piscina" y "garaje"
    df['piscina'] = df['piscina'].replace('Sin especificar', 'No')
    df['garaje'] = df['garaje'].replace('Sin especificar', 'No')
    df['chimenea'] = df['chimenea'].replace('Sin especificar', 'No')

    for col in columnas_cualitativas_influyentes:
        df[col] = df[col].astype(str)  # Convertir a string para asegurar que se trate como categórica
        fig_box = px.box(df, x=col, y='precio', log_y=True, title=f'Diagrama de Cajas: {col} vs Precio')
        st.plotly_chart(fig_box)

        # Calcular y mostrar las medianas para cada categoría de la columna
        categorias = df[col].unique()
        for categoria in categorias:
            mediana_precio = df[df[col] == categoria]['precio'].median()
            st.write(f"Mediana de precio cuando {col} es {categoria}: {mediana_precio}")

            #Aquí quedaría bien una conclusion de lo que se ve en cada boxplot como resumen
            #Quitaria aire y dejaria las otras 3, al menos para alquileres influyen bastante, aire no

    columnas_clustering = ["precio", "antiguedad", "conservacion", "latitud", "longitud", "piscina", "baños", "chimenea",
                           "habitaciones","superficie", "superficie_util", "garaje", "provincia", "planta", "orientacion"]

    # Crear una copia del DataFrame con las columnas seleccionadas
    df2 = df[columnas_clustering].copy()

    # Aplicar LabelEncoder a las columnas categóricas
    label_encoder = LabelEncoder()
    for column in ['antiguedad', 'conservacion', 'piscina', 'chimenea', 'garaje', 'provincia', 'planta', 'orientacion']:
        df2[column] = label_encoder.fit_transform(df2[column])

    # Calcular la correlación de cada columna con 'precio'
    corr_with_target = df2.corr()['precio'].drop('precio')

    # Convertir la serie a un DataFrame para facilitar la visualización
    corr_df = corr_with_target.reset_index()
    corr_df.columns = ['Columna', 'Correlación']
    corr_df = corr_df.sort_values(by='Correlación', ascending=True)

    # Crear el gráfico de barras
    fig = px.bar(corr_df, x='Columna', y='Correlación', color='Correlación',
                 color_continuous_scale='RdBu_r', range_color=[-1, 1])
    fig.update_layout(
        title='Correlación de las columnas numéricas con la objetivo "precio"',
        width=800,
        height=600
    )
    st.plotly_chart(fig)

    st.subheader("Metodo del codo para escalar los datos y elegir el número de clusters")

    # Verificar que no haya valores None antes de escalar los datos
    if df2.isnull().values.any():
        df2 = df2.fillna(0)  # O cualquier otro valor que consideres apropiado

    # Escalado de datos para el método del codo
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(df2)
    inertia = []
    K_range = range(2, 10)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(datos_escalados)
        inertia.append(kmeans.inertia_)

    # Crear el gráfico del método del codo
    fig_codo = px.line(x=list(K_range), y=inertia, markers=True)
    fig_codo.update_layout(
        xaxis_title='Número de clusters (K)',
        yaxis_title='Inercia',
        width=800,
        height=600
    )
    st.plotly_chart(fig_codo)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df2['cluster'] = kmeans.fit_predict(datos_escalados)

    st.write("Clusters generados")

    ##Ver cuantos hay por Cluster
    st.dataframe(df2["cluster"].value_counts())

    # Aplicar logaritmo al precio
    #df2["preciolog"] = np.log1p(df2["precio"])

    # Crear el gráfico con Plotly Express
    #fig = px.scatter(df2, x="superficie", y="precio", color="cluster",
    #                 title="Distribución de Precios vs Superficie por Cluster",
    #                 labels={"superficie": "Superficie (m²)", "precio": "Precio (€)(log)"},
    #                 opacity=0.5, color_continuous_scale="viridis")

    # Mostrar el gráfico en Streamlit
    #st.plotly_chart(fig)

    ##Probar quitando logs y 3 clusters para el plot


    ##Guardar el df nuevo con los clusters en MySQL
    conn = mysql.connector.connect(
        host=st.secrets["database"]["host"],
        user=st.secrets["database"]["user"],
        password=st.secrets["database"]["password"],
        database=st.secrets["database"]["name"],
        auth_plugin='mysql_native_password'
    )
    cursor = conn.cursor()

    # Crear tabla si no existe
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS compras_clusters (
                id INT AUTO_INCREMENT PRIMARY KEY,
                precio FLOAT,
                antiguedad INT,
                conservacion INT,
                latitud FLOAT,
                longitud FLOAT,
                piscina INT,
                baños INT,
                chimenea INT,
                habitaciones INT,
                superficie FLOAT,
                superficie_util FLOAT,
                garaje INT,
                provincia INT,
                planta INT,
                orientacion INT,
                cluster INT,
                UNIQUE KEY unique_key (precio, antiguedad, conservacion, latitud, longitud, piscina, baños, chimenea, habitaciones, superficie, superficie_util, garaje, provincia, planta, orientacion, cluster)
            )
        """)

    # Insertar datos en la tabla
    for _, row in df2.iterrows():
        cursor.execute("""
                INSERT INTO compras_clusters (precio, antiguedad, conservacion, latitud, longitud, piscina, baños, chimenea, habitaciones, superficie, superficie_util, garaje, provincia, planta, orientacion, cluster)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    precio = VALUES(precio),
                    antiguedad = VALUES(antiguedad),
                    conservacion = VALUES(conservacion),
                    latitud = VALUES(latitud),
                    longitud = VALUES(longitud),
                    piscina = VALUES(piscina),
                    baños = VALUES(baños),
                    chimenea = VALUES(chimenea),
                    habitaciones = VALUES(habitaciones),
                    superficie = VALUES(superficie),
                    superficie_util = VALUES(superficie_util),
                    garaje = VALUES(garaje),
                    provincia = VALUES(provincia),
                    planta = VALUES(planta),
                    orientacion = VALUES(orientacion),
                    cluster = VALUES(cluster)
            """, tuple(row[columnas_clustering].tolist() + [int(row['cluster'])]))

    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    clustering_page()
