import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

def classification_page():
    st.title("Clasificación")
    st.write("Clasificación de inmuebles en el cluster que le corresponde para análisis inmobiliario.")

    # Cargar datos
    df = pd.read_csv("data/alquileres_completo_limpio.csv")

    # Extraer latitud y longitud
    df["latitud"] = df["coordenadas"].str.split(",").str[0].str.strip().astype(float, errors='ignore')
    df["longitud"] = df["coordenadas"].str.split(",").str[1].str.strip().astype(float, errors='ignore')

    # Limpiar valores atípicos
    df = df[(df["superficie"].between(df["superficie"].quantile(0.03), df["superficie"].quantile(0.97)))]
    df = df[(df["superficie_util"].between(df["superficie_util"].quantile(0.01), df["superficie_util"].quantile(0.99)))]
    df = df[(df["precio"].between(df["precio"].quantile(0.05), df["precio"].quantile(0.95)))]

    # Variables usadas para clustering
    columnas_clustering = ["precio", "antiguedad", "conservacion", "latitud", "longitud", "piscina", "baños",
                           "chimenea", "habitaciones", "superficie", "superficie_util", "garaje", "provincia",
                           "planta", "orientacion"]

    # Codificar variables categóricas
    df_encoded = df.copy()
    label_encoders = {}
    for col in ['antiguedad', 'conservacion', 'piscina', 'chimenea', 'garaje', 'provincia', 'orientacion']:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

    # Escalar los datos
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df_encoded[columnas_clustering])

    # Clustering (KMeans)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_encoded['cluster'] = kmeans.fit_predict(X_cluster)

    # UI - Inputs
    input_data = {}
    for col in columnas_clustering:
        if df[col].dtype == "object":
            input_data[col] = st.selectbox(f"Selecciona {col}", options=df[col].unique())
        elif df[col].dtype == "float64" or df[col].dtype == "int64":
            if col in ['latitud', 'longitud']:
                input_data[col] = st.number_input(f"Introduce {col}", 
                                                  float(df[col].min()), float(df[col].max()), format="%.6f")
            else:
                input_data[col] = st.slider(f"Introduce {col}", 
                                            int(df[col].min()), int(df[col].max()), step=1)

    if st.button("Predecir Cluster"):
        # Preparar entrada para el modelo
        input_df = pd.DataFrame([input_data])

        # Codificar la entrada
        for col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col].astype(str))

        # Escalar
        input_scaled = scaler.transform(input_df)

        # Entrenar modelo de clasificación
        X = df_encoded[columnas_clustering]
        y = df_encoded['cluster']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predecir
        prediction = model.predict(input_scaled)
        st.success(f"El inmueble pertenece al cluster: {prediction[0]}")

        # Precisión
        accuracy = accuracy_score(y_test, model.predict(X_test))
        st.write(f"Precisión del modelo: {accuracy:.2f}")

if __name__ == "__main__":
    classification_page()
