import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

def clustering_page():
    st.title("Clustering")
    st.write("Preparación de clusters para análisis inmobiliario.")

    # Cargar datos
    df = pd.read_csv("data/compras_completo_limpio.csv")

    # Limpieza inicial
    df["antiguedad"] = df["antiguedad"].replace("Sin especificar", "10-20")
    df["conservacion"] = df["conservacion"].replace("Sin especificar", "En buen estado")
    df["piscina"] = df["piscina"].replace("Sin especificar", "No")
    df["chimenea"] = df["chimenea"].replace("Sin especificar", "No")
    df["garaje"] = df["garaje"].replace("Sin especificar", "No")

    # Coordenadas
    df["latitud"] = df["coordenadas"].str.split(",").str[0].str.strip().astype(float)
    df["longitud"] = df["coordenadas"].str.split(",").str[1].str.strip().astype(float)

    # Outliers
    df = df[
        df["superficie"].between(df["superficie"].quantile(0.05), df["superficie"].quantile(0.95)) &
        df["superficie_util"].between(df["superficie_util"].quantile(0.01), df["superficie_util"].quantile(0.99)) &
        df["precio"].between(df["precio"].quantile(0.02), df["precio"].quantile(0.98))
    ]

    # Análisis correlaciones
    st.subheader("Mapa de correlación numérica")
    columnas_numericas = df.select_dtypes(include='number').columns
    corr_matrix = df[columnas_numericas].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    fig_corr.update_layout(width=800, height=600)
    st.plotly_chart(fig_corr)

    st.subheader("Correlación con el precio")
    corr_precio = corr_matrix['precio'].drop('precio')
    fig_precio = px.bar(
        corr_precio.sort_values(), x=corr_precio.index, y=corr_precio.values,
        labels={'x': 'Variable', 'y': 'Correlación'}, color=corr_precio.values,
        color_continuous_scale='RdBu_r', range_color=[-1, 1]
    )
    fig_precio.update_layout(width=800, height=600)
    st.plotly_chart(fig_precio)

    st.subheader("Análisis de variables cualitativas")
    for col in ["piscina", "garaje", "chimenea"]:
        fig_box = px.box(df, x=col, y='precio', log_y=True, title=f'{col.capitalize()} vs Precio')
        st.plotly_chart(fig_box)
        for valor in df[col].unique():
            mediana = df[df[col] == valor]['precio'].median()
            st.write(f"Mediana para {col} = {valor}: {mediana:,.0f} €")

    # Clustering
    columnas_clustering = [
        "precio", "antiguedad", "conservacion", "latitud", "longitud", "piscina", "baños", "chimenea",
        "habitaciones", "superficie", "superficie_util", "garaje", "provincia", "planta", "orientacion"
    ]
    df_cluster = df[columnas_clustering].copy()

    # Codificación
    le = LabelEncoder()
    for col in df_cluster.select_dtypes(include='object').columns:
        df_cluster[col] = le.fit_transform(df_cluster[col].astype(str))

    # Escalado
    scaler = StandardScaler()
    X = scaler.fit_transform(df_cluster.fillna(0))

    # Método del codo
    st.subheader("Método del codo")
    inertia = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    fig_codo = px.line(x=list(range(2, 10)), y=inertia, markers=True,
                       labels={'x': 'Número de Clusters', 'y': 'Inercia'})
    fig_codo.update_layout(width=800, height=500)
    st.plotly_chart(fig_codo)

    # Clustering final con k=3
    kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_cluster['cluster'] = kmeans_final.fit_predict(X)

    st.subheader("Distribución de Clusters")
    st.dataframe(df_cluster['cluster'].value_counts())

    # Guardar CSV
    df_cluster.to_csv("data/compras_clusters.csv", index=False)
    st.success("Clustering guardado en: `data/compras_clusters.csv`")

if __name__ == "__main__":
    clustering_page()
