import pandas as pd
import streamlit as st
from PIL import Image

@st.cache_data
def cargar_datos_csv(path):
    return pd.read_csv(path, low_memory=False)

def main():
    st.title("Visualización de Datos desde CSV")

    df_alquileres = cargar_datos_csv('data/alquileres_completo_limpio.csv')
    df_compras = cargar_datos_csv('data/compras_completo_limpio.csv')

    st.subheader("Datos completos de inmuebles en alquiler:")
    with st.expander("Mostrar tabla de alquileres"):
        st.dataframe(df_alquileres)

    st.subheader("Datos completos de inmuebles en venta:")
    with st.expander("Mostrar tabla de compras"):
        st.dataframe(df_compras)

    image = Image.open('app/images/db.png')
    st.image(image, caption='Modelo entidad-relación entre nuestras tablas', use_container_width=True, width=300)
