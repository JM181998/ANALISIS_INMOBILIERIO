import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from app.pages.database import cargar_datos_csv

# Cargar datos desde la base de datos
df = pd.read_csv("data/alquileres_completo_limpio.csv")

def comparador_page():
    st.title("Comparador de Inmuebles en alquiler")
    st.write("En esta página podras buscar y comparar inmuebles de diferentes zonas de España.")
    st.write("Además contamos con una herramienta con la que podrás observar de forma visual las diferencias entre las principales caracteristicas que puede ofrecerte cada uno de ellos.")

    ##GRAFICO DE RADAR PARA COMPARAR INMUEBLES##
    columnas_importantes = ['identificador', 'superficie', 'superficie_util', 'planta',
                            'baños', 'precio_m2', 'precio', 'habitaciones', 'provincia', 'href', 'nombre',
                            'comunidad_autonoma']
    df_importante = df[columnas_importantes].copy()

    df_importante.dropna(subset=['superficie', 'baños', 'precio_m2', 'habitaciones', 'planta'], inplace=True)

    # Aplicar transformación logarítmica en precio, precio_m2, superficie y superficie_util
    df_importante['precio_log'] = np.log1p(df_importante['precio'])
    df_importante['precio_m2_log'] = np.log1p(df_importante['precio_m2'])
    df_importante['superficie_log'] = np.log1p(df_importante['superficie'])
    df_importante['superficie_util_log'] = np.log1p(df_importante['superficie_util'])

    # Combinamos los datos transformados
    columnas_para_transformar = ['superficie_log', 'superficie_util_log', 'precio_m2_log']
    df_transformed = df_importante[columnas_para_transformar].copy()
    df_transformed = df_transformed.reset_index(drop=True)
    df_importante = df_importante.reset_index(drop=True)
    df_transformed['precio'] = df_importante['precio_log']
    df_transformed['habitaciones'] = df_importante['habitaciones']
    df_transformed['baños'] = df_importante['baños']
    df_transformed['planta'] = df_importante['planta']
    df_transformed['identificador'] = df_importante['identificador'].values

    # Título de la aplicación
    st.write('### Selector de inmuebles')

    # Filtro de CCAA
    ccaa_seleccionada = st.selectbox('Selecciona la Comunidad Autonoma', df['comunidad_autonoma'].unique(), index=0)

    # Filtrar el DataFrame por la CCAA seleccionada
    df_ccaa_filtrado = df[df['comunidad_autonoma'] == ccaa_seleccionada]

    # Filtro de provincias
    provincias = st.multiselect(
        'En qué provincia quieres buscar',
        df_ccaa_filtrado['provincia'].unique(),
        placeholder='Selecciona una o varias provincias'
    )

    # Filtrar el DataFrame por provincias seleccionadas
    if provincias:
        df_filtrado = df_ccaa_filtrado[df_ccaa_filtrado['provincia'].isin(provincias)]
    else:
        df_filtrado = df_ccaa_filtrado

        # Filtro de áreas
    areas = st.multiselect(
        'En qué área concreta quieres buscar',
        df_filtrado['area'].unique(),
        placeholder='Selecciona una o varias áreas'
    )

    # Filtrar el DataFrame por áreas seleccionadas
    if areas:
        df_filtrado = df_filtrado[df_filtrado['area'].isin(areas)]

    # Eliminamos columnas completamente vacías
    df_filtrado = df_filtrado.dropna(axis=1, how='all')

    # Mostramos el DataFrame filtrado solo si no está vacío
    if not df_filtrado.empty:
        st.write('Datos filtrados por ubicación:')
        st.dataframe(df_filtrado)
    else:
        st.write('No hay datos disponibles para las provincias seleccionadas.')

    # Pedimos que seleccione los identificadores de los inmuebles que se quieren comparar
    id1 = st.selectbox('Selecciona el primer identificador', df_filtrado['identificador'].unique())
    id2 = st.selectbox('Selecciona el segundo identificador', df_filtrado['identificador'].unique())

    # Filtramos los datos seleccionados
    piso1 = df_transformed[df_transformed['identificador'] == id1].drop(columns=['identificador'])
    piso2 = df_transformed[df_transformed['identificador'] == id2].drop(columns=['identificador'])

    # Añadir el primer valor al final de la lista de valores para cerrar el polígono
    valores_piso1 = piso1.iloc[0].tolist()
    valores_piso1.append(valores_piso1[0])

    valores_piso2 = piso2.iloc[0].tolist()
    valores_piso2.append(valores_piso2[0])

    # Añadir el primer eje al final de la lista de ejes para cerrar el polígono
    ejes = piso1.columns.tolist()
    ejes.append(ejes[0])

    # Verificación de los valores y ejes
    ##st.write("Valores de 'piso1':", valores_piso1)
    #st.write("Valores de 'piso2':", valores_piso2)
    #st.write("Ejes:", ejes)

    # Creamos el gráfico de radar
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=valores_piso1,
        theta=ejes,
        fill='toself',
        name=f'{id1} - {df_filtrado[df_filtrado["identificador"] == id1]["provincia"].values[0]}',
        #line = dict(color='blue')
    ))

    fig.add_trace(go.Scatterpolar(
        r=valores_piso2,
        theta=ejes,
        fill='toself',
        name=f'{id2} - {df_filtrado[df_filtrado["identificador"] == id2]["provincia"].values[0]}',
        #line = dict(color='red')

    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=False,
                range=[-3, 10]  # Ajuste del rango del eje radial
            )
        ),
        showlegend=True
    )

    st.plotly_chart(fig)

    # Mostramos el DataFrame comparativo
    piso1_original = df[df['identificador'] == id1][columnas_importantes]
    piso2_original = df[df['identificador'] == id2][columnas_importantes]
    df_comparativo = pd.concat([piso1_original, piso2_original], ignore_index=True)

    # Reordenamos las columnas segun su importancia
    columnas_ordenadas = [
        'identificador', 'nombre', 'provincia', 'comunidad_autonoma', 'precio', 'precio_m2',
        'superficie', 'superficie_util', 'planta',
        'habitaciones', 'baños', 'href'
    ]
    df_comparativo = df_comparativo[columnas_ordenadas]

    st.write('Comparativa de características originales:')
    st.dataframe(df_comparativo)

if __name__ == "__main__":
    comparador_page()