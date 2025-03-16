import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import folium
import geopandas as gpd
import json

from app.pages.database import cargar_datos_bd
#from scripts.data_processing import filtrar_outliers_precio, filtrar_outliers_superficie

# Cargar datos desde la base de datos
query = "SELECT * FROM general_alquileres"
df = cargar_datos_bd(query)

# Eliminar filas donde el valor de la columna agencia sea "Agencia no disponible"
#df = df[df['agencia'] != "Agencia no disponible"]

def eda_page():
    st.title("Análisis exploratorio de los datos de Alquileres")
    st.write("Aquí puedes ver la presentación de los datos obtenidos y como se relacionan entre sí.")
    #MAPA COROPLÉTICO
    def mapa_coropletico(df):
        st.markdown("## Mapa Coroplético de Pisos/Casas en alquiler por Provincia")
        # Cargamos el archivo GeoJSON de las provincias de España
        geojson_path = "data/spanish_provinces.geojson"
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        gdf_spain = gpd.GeoDataFrame.from_features(geojson_data["features"])
        # Agrupamos por provincia y contamos el número de pisos/casas
        df_grouped = df.groupby('provincia').size().reset_index(name='counts')
        # Creamos un diccionario de mapeo para asegurarnos de que coincidan los nombres de las provincias en ambos archivos
        mapping = {
            'A Coruña': 'A Coruña',
            'Alava': 'Araba',
            'Albacete': 'Albacete',
            'Alicante': 'Alacant',
            'Almeria': 'Almería',
            'Asturias': 'Asturias',
            'Avila': 'Ávila',
            'Badajoz': 'Badajoz',
            'Barcelona': 'Barcelona',
            'Burgos': 'Burgos',
            'Caceres': 'Cáceres',
            'Cadiz': 'Cádiz',
            'Cantabria': 'Cantabria',
            'Castellon': 'Castelló',
            'Ciudad Real': 'Ciudad Real',
            'Cordoba': 'Córdoba',
            'Cuenca': 'Cuenca',
            'Girona': 'Girona',
            'Granada': 'Granada',
            'Guadalajara': 'Guadalajara',
            'Guipuzcoa': 'Gipuzcoa',
            'Huelva': 'Huelva',
            'Huesca': 'Huesca',
            'Islas Baleares': 'Illes Balears',
            'Jaen': 'Jaén',
            'La Rioja': 'La Rioja',
            'Las Palmas': 'Las Palmas',
            'Leon': 'León',
            'Lleida': 'Lleida',
            'Lugo': 'Lugo',
            'Madrid': 'Madrid',
            'Malaga': 'Málaga',
            'Melilla': 'Melilla',
            'Murcia': 'Murcia',
            'Navarra': 'Navarra',
            'Ourense': 'Ourense',
            'Palencia': 'Palencia',
            'Pontevedra': 'Pontevedra',
            'Salamanca': 'Salamanca',
            'Santa Cruz de Tenerife': 'Santa Cruz de Tenerife',
            'Segovia': 'Segovia',
            'Sevilla': 'Sevilla',
            'Soria': 'Soria',
            'Tarragona': 'Tarragona',
            'Teruel': 'Teruel',
            'Toledo': 'Toledo',
            'Valencia': 'València',
            'Valladolid': 'Valladolid',
            'Vizcaya': 'Bizkaia',
            'Zamora': 'Zamora',
            'Zaragoza': 'Zaragoza'
        }
        # Normalizar los nombres de las provincias en ambos DataFrames y aplicar el mapeo
        gdf_spain['provincia_normalized'] = gdf_spain['provincia'].map(mapping)
        df_grouped['provincia_normalized'] = df_grouped['provincia'].map(mapping)
        # Unimos los datos con el GeoDataFrame
        gdf_spain = gdf_spain.merge(df_grouped, left_on='provincia_normalized', right_on='provincia_normalized', how='left')
        # Creamos el mapa coroplético con Folium
        m = folium.Map(location=[40.416775, -3.703790], zoom_start=6)

        folium.Choropleth(
            geo_data=geojson_data,
            name='choropleth',
            data=df_grouped,
            columns=['provincia_normalized', 'counts'],
            key_on='feature.properties.provincia',
            fill_color='Oranges',
            fill_opacity=0.9,
            line_opacity=0.2,
            legend_name='Cantidad de pisos/casas en alquiler por provincia'
        ).add_to(m)

        folium.LayerControl().add_to(m)
        # Renderizamos el mapa directamente en Streamlit
        st.components.v1.html(m._repr_html_(), height=525)
    mapa_coropletico(df)

    st.write("")
    st.write("")

    #ANALISIS DE OULIERS EN PRECIO
    def analizar_outliers_precio(df):
        st.markdown("## Análisis de valores atípicos de inmuebles en alquiler")
        # Transformamos la columna precio a su valor logarítmico
        df['log_precio'] = np.log(df['precio'])
        # Calculamos el Z-Score e identificamos outliers
        df['z_score'] = (df['log_precio'] - df['log_precio'].mean()) / df['log_precio'].std()
        df['outlier'] = df['z_score'].apply(lambda x: 'Outlier' if np.abs(x) > 3 else 'Normal')
        # Creamos el histograma con Plotly
        fig = px.histogram(df, x='log_precio', color='outlier', title='Histograma de precios en escala logarítmica',
                           color_discrete_map={'Outlier': 'red', 'Normal': 'green'})
        # Seleccionamos algunas etiquetas redondas para mostrar en el eje x
        precios_redondos = [50, 100,250, 500, 1000, 2500, 5000, 10000, 20000, 40000, 80000]
        tickvals = np.log(precios_redondos)
        ticktext = precios_redondos
        # Actualizamos las etiquetas del eje x para mostrar el precio real
        fig.update_layout(
            xaxis_title='Precio',
            yaxis_title='Frecuencia',
            xaxis=dict(
                tickmode='array',
                tickvals=tickvals,
                ticktext=ticktext
            )
        )
        st.plotly_chart(fig)
    analizar_outliers_precio(df)

    st.write("")
    st.write("")

    ##FRAN
    def scatter_segmentado(df):

        st.markdown("## Relación precio/superficie segmentado por Comunidad Autonoma de los inmuebles en alquiler")

        Q1 = df['superficie'].quantile(0.05)
        Q3 = df['superficie'].quantile(0.95)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filtrar outliers y superficies inferiores a 30 m²
        filtered_df = df[
            (df['superficie'] >= lower_bound) & (df['superficie'] <= upper_bound) & (df['superficie'] >= 30)]

        # Aplicar logaritmo al precio
        # filtered_df["preciolog"] = np.log1p(filtered_df["precio"])

        # Crear scatter plot con Plotly Express
        fig = px.scatter(filtered_df, x='superficie', y='precio', color='comunidad_autonoma',
                         title='Relación entre Precio y Superficie segmentado por Comunidad Autónoma',
                         labels={'superficie': 'Superficie (m²)', 'precio': 'Precio (€)'},
                         opacity=0.5, color_continuous_scale='viridis')
        st.plotly_chart(fig)

    scatter_segmentado(df)

    st.write("")
    st.write("")

    ##ENRIQUE
    def plot_precio_promedio(df):
        if 'antiguedad' in df.columns and 'precio' in df.columns:
            # Eliminar valores nulos
            df_filtered = df.dropna(subset=['antiguedad', 'precio'])

            # Mapeo de las categorías de antigüedad
            antiguedad_map = {
                '< 5': 'Menos de 5 años',
                '5-10': 'Entre 5 y 10 años',
                '10-20': 'Entre 10 y 20 años',
                '20-30': 'Entre 20 y 30 años',
                '30-50': 'Entre 30 y 50 años',
                '> 50': 'Más de 50 años'
            }

            # Aplicar el mapeo a la columna de antigüedad
            df_filtered['antiguedad'] = df_filtered['antiguedad'].map(antiguedad_map)

            # Contar la media de precio por categoría de antigüedad
            precio_promedio = df_filtered.groupby('antiguedad')['precio'].mean()

            # Orden deseado de las columnas
            columnas_ordenadas = [
                "Menos de 5 años", "Entre 5 y 10 años", "Entre 10 y 20 años",
                "Entre 20 y 30 años", "Entre 30 y 50 años", "Más de 50 años"
            ]

            # Reindexar el DataFrame para que siga el orden deseado
            precio_promedio = precio_promedio.reindex(columnas_ordenadas)

            # Mostrar los datos en Streamlit
            st.write("### Precio Promedio por Antigüedad")

            # Graficar con Plotly Express
            fig = px.bar(precio_promedio, x=precio_promedio.index, y=precio_promedio.values,
                         labels={'x': 'Antigüedad', 'y': 'Precio Promedio'})

            fig.update_layout(xaxis_title='Antigüedad', yaxis_title='Precio Promedio')

            # Mostrar la gráfica en Streamlit
            st.plotly_chart(fig)
        else:
            st.error("Las columnas 'antiguedad' o 'precio' no existen en el DataFrame.")
    plot_precio_promedio(df)

    st.write("")
    st.write("")

    ##JUANMA
    def plot_precio_por_provincia_y_comunidad(df):
        if 'provincia' in df.columns and 'precio' in df.columns:
            df_limpio = df.dropna(subset=['provincia', 'precio'])
            precio_por_provincia = df_limpio.groupby('provincia')['precio'].mean().sort_values(ascending=False)
            media_precio_provincia = df_limpio['precio'].mean()

            st.write("### Precio Promedio por Provincia")
            fig_provincia = px.bar(precio_por_provincia, x=precio_por_provincia.index, y=precio_por_provincia.values,
                                   labels={'x': 'Provincia', 'y': 'Precio Promedio'})

            fig_provincia.add_shape(type='line', x0=-0.5, y0=media_precio_provincia, x1=len(precio_por_provincia) - 0.5,
                                    y1=media_precio_provincia,
                                    line=dict(color='red', dash='dash'), name=f'Media: {media_precio_provincia:.2f}')

            fig_provincia.update_layout(
                xaxis_title='Provincia', yaxis_title='Precio Promedio',
                xaxis=dict(tickangle=45)
            )

            st.plotly_chart(fig_provincia)
        else:
            st.error("Las columnas 'provincia' o 'precio' no existen en el DataFrame.")

        st.write("")

        if 'comunidad_autonoma' in df.columns and 'precio' in df.columns:
            df_limpio = df.dropna(subset=['comunidad_autonoma', 'precio'])
            precio_por_comunidad = df_limpio.groupby('comunidad_autonoma')['precio'].mean().sort_values(ascending=False)
            media_precio_comunidad = df_limpio['precio'].mean()

            st.write("### Precio Promedio por Comunidad Autónoma")
            fig_comunidad = px.bar(precio_por_comunidad, x=precio_por_comunidad.index, y=precio_por_comunidad.values,
                                   labels={'x': 'Comunidad Autónoma', 'y': 'Precio Promedio'})

            fig_comunidad.add_shape(type='line', x0=-0.5, y0=media_precio_comunidad, x1=len(precio_por_comunidad) - 0.5,
                                    y1=media_precio_comunidad,
                                    line=dict(color='red', dash='dash'), name=f'Media: {media_precio_comunidad:.2f}')

            fig_comunidad.update_layout(
                xaxis_title='Comunidad Autónoma', yaxis_title='Precio Promedio',
                xaxis=dict(tickangle=45)
            )

            st.plotly_chart(fig_comunidad)
        else:
            st.error("Las columnas 'comunidad_autonoma' o 'precio' no existen en el DataFrame.")

    plot_precio_por_provincia_y_comunidad(df)

if __name__ == "__main__":
    eda_page()