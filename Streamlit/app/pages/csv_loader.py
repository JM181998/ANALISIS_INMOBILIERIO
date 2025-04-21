import streamlit as st
import pandas as pd
from PIL import Image

def main():
    st.title("Visualización de Datos desde CSV")

    # Cargar y mostrar CSV de alquileres
    try:
        df_alquileres = pd.read_csv('data/alquileres_completo_limpio.csv')
        st.subheader("Datos de Alquileres")
        with st.expander("Mostrar tabla de alquileres"):
            st.dataframe(df_alquileres)
    except FileNotFoundError:
        st.error("No se encontró el archivo 'alquileres_completo_limpio.csv'.")

    # Cargar y mostrar CSV de compras
    try:
        df_compras = pd.read_csv('data/compras_completo_limpio.csv', low_memory=False)
        st.subheader("Datos de Compras")
        with st.expander("Mostrar tabla de compras"):
            st.dataframe(df_compras)
    except FileNotFoundError:
        st.error("No se encontró el archivo 'compras_completo_limpio.csv'.")

    # Mostrar imagen del modelo entidad-relación (opcional)
    try:
        image = Image.open('app/images/db.png')
        st.image(image, caption='Modelo entidad-relación entre nuestras tablas', use_container_width=True)
    except FileNotFoundError:
        st.warning("No se encontró la imagen del modelo entidad-relación.")

