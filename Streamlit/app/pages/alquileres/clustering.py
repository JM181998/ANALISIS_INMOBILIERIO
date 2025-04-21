import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clustering_page():
    st.title("Clustering")
    st.write("Preparación de clusters para análisis inmobiliario.")

    # Cargar datos desde archivo CSV
    df = pd.read_csv("data/alquileres_completo_limpio.csv")

    # Eliminar valores atípicos
    q1_superficie = df["superficie"].quantile(0.03)
    q99_superficie = df["superficie"].quantile(0.97)
    df = df[(df["superficie"] >= q1_superficie) & (df["superficie"] <= q99_superficie)]

    q1_superficie_util = df["superficie_util"].quantile(0.01)
    q99_superficie_util = df["superficie_util"].quantile(0.99)
    df = df[(df["superficie_util"] >= q1_superficie_util) & (df["superficie_util"] <= q99_superficie_util)]

    q5_precio = df["precio"].quantile(0.05)
    q95_precio = df["precio"].quantile(0.95)
    df = df[(df["precio"] >= q5_precio) & (df["precio"] <= q95_precio)]

    # Reemplazar valores "Sin especificar"
    df["antiguedad"] = df["antiguedad"].replace("Sin especificar", "10-20")
    df["conservacion"] = df["conservacion"].replace("Sin especificar", "En buen estado")

    # Separar coordenadas
    df["latitud"] = df["coordenadas"].str.split(",").str[0].str.strip().astype(float, errors='ignore')
    df["longitud"] = df["coordenadas"].str.split(",").str[1].str.strip().astype(float, errors='ignore')

    # Mapa de correlación numérico
    columnas_numericas = df.select_dtypes(include=['number']).columns
    corr_matrix = df[columnas_numericas].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    fig.update_layout(title='Mapa de correlación entre columnas numéricas', width=800, height=600)
    st.plotly_chart(fig)

    # Correlación con "precio"
    corr_with_target = df[columnas_numericas].corr()['precio'].drop('precio')
    corr_df = corr_with_target.reset_index()
    corr_df.columns = ['Columna', 'Correlación']
    corr_df = corr_df.sort_values(by='Correlación', ascending=True)
    fig = px.bar(corr_df, x='Columna', y='Correlación', color='Correlación',
                 color_continuous_scale='RdBu_r', range_color=[-1, 1])
    fig.update_layout(title='Correlación con "precio"', width=800, height=600)
    st.plotly_chart(fig)

    st.subheader("Análisis de categorías cualitativas")

    columnas_cualitativas_influyentes = ["piscina", "garaje", "chimenea"]
    for col in columnas_cualitativas_influyentes:
        df[col] = df[col].replace('Sin especificar', 'No').astype(str)
        fig_box = px.box(df, x=col, y='precio', log_y=True, title=f'Diagrama de Cajas: {col} vs Precio')
        st.plotly_chart(fig_box)
        for categoria in df[col].unique():
            mediana_precio = df[df[col] == categoria]['precio'].median()
            st.write(f"Mediana de precio cuando {col} es {categoria}: {mediana_precio}")

    columnas_clustering = ["precio", "antiguedad", "conservacion", "latitud", "longitud", "piscina", "baños", "chimenea",
                           "habitaciones", "superficie", "superficie_util", "garaje", "provincia", "planta", "orientacion"]

    df2 = df[columnas_clustering].copy()

    label_encoder = LabelEncoder()
    for column in ['antiguedad', 'conservacion', 'piscina', 'chimenea', 'garaje', 'provincia', 'planta', 'orientacion']:
        df2[column] = label_encoder.fit_transform(df2[column])

    # Gráfico de correlación para clustering
    corr_with_target = df2.corr()['precio'].drop('precio')
    corr_df = corr_with_target.reset_index()
    corr_df.columns = ['Columna', 'Correlación']
    corr_df = corr_df.sort_values(by='Correlación', ascending=True)
    fig = px.bar(corr_df, x='Columna', y='Correlación', color='Correlación',
                 color_continuous_scale='RdBu_r', range_color=[-1, 1])
    fig.update_layout(title='Correlación con "precio" (datos transformados)', width=800, height=600)
    st.plotly_chart(fig)

    st.subheader("Método del codo")

    if df2.isnull().values.any():
        df2 = df2.fillna(0)

    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(df2)
    inertia = []
    K_range = range(2, 10)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(datos_escalados)
        inertia.append(kmeans.inertia_)

    fig_codo = px.line(x=list(K_range), y=inertia, markers=True)
    fig_codo.update_layout(xaxis_title='Número de clusters (K)', yaxis_title='Inercia', width=800, height=600)
    st.plotly_chart(fig_codo)

    # Aplicar KMeans final
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df2['cluster'] = kmeans.fit_predict(datos_escalados)

    st.write("Clusters generados")
    st.dataframe(df2["cluster"].value_counts())

    # Guardar los datos en archivo CSV
    df2.to_csv("data/alquiler_clusters.csv", index=False)
    st.success("Clusters guardados en 'data/alquiler_clusters.csv'")

if __name__ == "__main__":
    clustering_page()
