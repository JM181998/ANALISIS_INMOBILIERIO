import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import plotly.express as px

def regression_page():
    st.title("Regresión")
    st.write("Predicción del precio de inmuebles basado en sus características y cluster.")

    # Cargar los datos desde CSV
    df = pd.read_csv("data/compras_completo_limpio.csv")
    df_clusters = pd.read_csv("data/compras_clusters.csv")

    # Separar latitud y longitud
    df["latitud"] = pd.to_numeric(df["coordenadas"].str.split(",").str[0].str.strip(), errors='coerce')
    df["longitud"] = pd.to_numeric(df["coordenadas"].str.split(",").str[1].str.strip(), errors='coerce')

    # Eliminar valores atípicos
    df = df[
        df["superficie"].between(df["superficie"].quantile(0.05), df["superficie"].quantile(0.95)) &
        df["superficie_util"].between(df["superficie_util"].quantile(0.01), df["superficie_util"].quantile(0.99)) &
        df["precio"].between(df["precio"].quantile(0.02), df["precio"].quantile(0.98))
    ]

    # Variables de entrada
    input_data = {}
    columnas_clustering = ["precio", "antiguedad", "conservacion", "latitud", "longitud", "piscina", "baños",
                           "chimenea", "habitaciones", "superficie", "superficie_util", "garaje", "provincia",
                           "planta", "orientacion"]

    input_data['precio'] = st.slider("Introduce el precio", int(df['precio'].min()), int(df['precio'].max()), step=10)
    input_data['antiguedad'] = st.selectbox("Introduce la antigüedad", sorted(df['antiguedad'].dropna().unique()))
    input_data['conservacion'] = st.selectbox("Introduce la conservación", sorted(df['conservacion'].dropna().unique()))
    input_data['latitud'] = st.number_input("Introduce la latitud", float(df['latitud'].min()), float(df['latitud'].max()), format="%.6f")
    input_data['longitud'] = st.number_input("Introduce la longitud", float(df['longitud'].min()), float(df['longitud'].max()), format="%.6f")
    input_data['piscina'] = st.selectbox("¿Tiene piscina?", sorted(df['piscina'].dropna().unique()))
    input_data['baños'] = st.slider("Introduce el número de baños", int(df['baños'].min()), int(df['baños'].max()), step=1)
    input_data['chimenea'] = st.selectbox("¿Tiene chimenea?", sorted(df['chimenea'].dropna().unique()))
    input_data['habitaciones'] = st.slider("Introduce el número de habitaciones", int(df['habitaciones'].min()), int(df['habitaciones'].max()), step=1)
    input_data['superficie'] = st.slider("Introduce la superficie", int(df['superficie'].min()), int(df['superficie'].max()), step=10)
    input_data['superficie_util'] = st.slider("Introduce la superficie útil", int(df['superficie_util'].min()), int(df['superficie_util'].max()), step=10)
    input_data['garaje'] = st.selectbox("¿Tiene garaje?", sorted(df['garaje'].dropna().unique()))
    input_data['provincia'] = st.selectbox("Introduce la provincia", sorted(df['provincia'].dropna().unique()))
    input_data['planta'] = st.slider("Introduce la planta", int(df['planta'].min()), int(df['planta'].max()), step=1)
    input_data['orientacion'] = st.selectbox("Introduce la orientación", sorted(df['orientacion'].dropna().unique()))

    if st.button("Predecir Precio"):
        input_df = pd.DataFrame([input_data])

        # Codificación
        cat_cols = ['antiguedad', 'conservacion', 'piscina', 'chimenea', 'garaje', 'provincia', 'orientacion']
        label_encoders = {}
        for col in cat_cols:
            df[col] = df[col].astype(str)
            df_clusters[col] = df_clusters[col].astype(str)
            input_df[col] = input_df[col].astype(str)

            le = LabelEncoder()
            le.fit(pd.concat([df[col], df_clusters[col]]).unique())
            df_clusters[col] = le.transform(df_clusters[col])
            input_df[col] = le.transform(input_df[col])
            label_encoders[col] = le

        # Escalado
        scaler = StandardScaler()
        scaler.fit(df_clusters[columnas_clustering])
        input_df_escalado = pd.DataFrame(scaler.transform(input_df[columnas_clustering]), columns=columnas_clustering)

        # Clasificación del cluster
        X = df_clusters[columnas_clustering]
        y = df_clusters['cluster']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        modelo_cluster = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo_cluster.fit(X_train, y_train)

        cluster_predicho = modelo_cluster.predict(input_df_escalado)[0]
        st.write(f"El inmueble pertenece al cluster: {cluster_predicho}")

        # Regresión por cluster
        df_cluster = df_clusters[df_clusters["cluster"] == cluster_predicho]
        X = df_cluster.drop(columns=["precio", "cluster"])
        y = df_cluster["precio"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        modelo_regresion = RandomForestRegressor(n_estimators=100, random_state=42)
        modelo_regresion.fit(X_train, y_train)

        input_df_escalado = input_df_escalado.reindex(columns=X.columns)
        precio_predicho = modelo_regresion.predict(input_df_escalado)[0]
        st.success(f"El precio estimado del inmueble es: {precio_predicho:.2f} €")

        # Métricas
        y_pred = modelo_regresion.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        medae = median_absolute_error(y_test, y_pred)
        st.write(f'MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, MedAE: {medae:.2f}')

        # Visualización
        df_clusters["preciolog"] = np.log1p(df_clusters["precio"])
        fig = px.scatter(df_clusters, x="superficie", y="preciolog", color="cluster",
                         title="Distribución de Precios vs Superficie por Cluster",
                         labels={"superficie": "Superficie (m²)", "preciolog": "Precio (€)(log)"},
                         opacity=0.5)
        input_df["preciolog"] = np.log1p(input_df["precio"])
        fig.add_scatter(x=input_df["superficie"], y=input_df["preciolog"], mode='markers',
                        name='Input del Usuario', marker=dict(color='red', size=10))
        st.plotly_chart(fig)

        # Guardar modelo entrenado
        with open(f"data/modelo_regresion_compras_cluster_{cluster_predicho}.pkl", "wb") as archivo:
            pickle.dump(modelo_regresion, archivo)

if __name__ == "__main__":
    regression_page()
