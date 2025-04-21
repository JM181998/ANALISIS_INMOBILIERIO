import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def comparador_page():
    st.title("Comparador de Inmuebles en Venta")
    st.write("Busca y compara inmuebles en diferentes zonas de España.")
    st.write("Visualiza las diferencias entre las principales características de forma clara y directa.")

    # Cargar y preparar datos
    df = pd.read_csv("data/compras_completo_limpio.csv")

    columnas_importantes = [
        'identificador', 'superficie', 'superficie_util', 'planta',
        'baños', 'precio_m2', 'precio', 'habitaciones',
        'provincia', 'href', 'nombre',
        'comunidad_autonoma', 'area'
    ]

    df = df[columnas_importantes].dropna(subset=[
        'superficie', 'baños', 'precio_m2', 'habitaciones', 'planta'
    ]).copy()

    # Transformaciones logarítmicas
    df['precio_log'] = np.log1p(df['precio'])
    df['precio_m2_log'] = np.log1p(df['precio_m2'])
    df['superficie_log'] = np.log1p(df['superficie'])
    df['superficie_util_log'] = np.log1p(df['superficie_util'])

    # Dataset para radar
    radar_cols = ['superficie_log', 'superficie_util_log', 'precio_m2_log', 'precio_log', 'habitaciones', 'baños', 'planta']
    df_radar = df[['identificador'] + radar_cols].copy()

    # Filtros
    st.subheader("Filtro de Ubicación")
    ccaa = st.selectbox("Comunidad Autónoma", sorted(df['comunidad_autonoma'].dropna().unique()))
    df_filtrado = df[df['comunidad_autonoma'] == ccaa]

    provincias = st.multiselect("Provincia", sorted(df_filtrado['provincia'].dropna().unique()))
    if provincias:
        df_filtrado = df_filtrado[df_filtrado['provincia'].isin(provincias)]

    areas = st.multiselect("Área", sorted(df_filtrado['area'].dropna().unique()))
    if areas:
        df_filtrado = df_filtrado[df_filtrado['area'].isin(areas)]

    if df_filtrado.empty:
        st.warning("No hay inmuebles disponibles con esos filtros.")
        return

    st.write("### Inmuebles disponibles:")
    st.dataframe(df_filtrado)

    # Selección de identificadores
    st.subheader("Selecciona los inmuebles a comparar")
    opciones = df_filtrado['identificador'].unique()
    id1 = st.selectbox("Inmueble 1", opciones)
    id2 = st.selectbox("Inmueble 2", opciones)

    if id1 == id2:
        st.warning("Selecciona dos inmuebles diferentes para comparar.")
        return

    # Extraer datos para radar plot
    piso1 = df_radar[df_radar['identificador'] == id1].drop(columns='identificador').iloc[0]
    piso2 = df_radar[df_radar['identificador'] == id2].drop(columns='identificador').iloc[0]
    labels = radar_cols + [radar_cols[0]]

    r1 = piso1.tolist() + [piso1.tolist()[0]]
    r2 = piso2.tolist() + [piso2.tolist()[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r1,
        theta=labels,
        fill='toself',
        name=f'{id1} ({df_filtrado[df_filtrado["identificador"] == id1]["provincia"].values[0]})'
    ))
    fig.add_trace(go.Scatterpolar(
        r=r2,
        theta=labels,
        fill='toself',
        name=f'{id2} ({df_filtrado[df_filtrado["identificador"] == id2]["provincia"].values[0]})'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False, range=[-2, 15])),
        showlegend=True
    )
    st.plotly_chart(fig)

    # Comparativa tabular
    st.subheader("Comparativa Tabular")
    comparativa = df[df['identificador'].isin([id1, id2])][[
        'identificador', 'nombre', 'area', 'provincia', 'comunidad_autonoma',
        'precio', 'precio_m2', 'superficie', 'superficie_util',
        'planta', 'habitaciones', 'baños', 'href'
    ]]
    st.dataframe(comparativa)

if __name__ == "__main__":
    comparador_page()
