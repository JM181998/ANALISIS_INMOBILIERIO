import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
import json
import geopandas as gpd

# Cargar datos desde CSV
df = pd.read_csv("data/compras_completo_limpio.csv")

def eda_page():
    st.title("Análisis exploratorio de los datos de compras")
    st.write("Aquí puedes ver la presentación de los datos obtenidos y como se relacionan entre sí.")

    # MAPA COROPLÉTICO
    def mapa_coropletico(df):
        st.markdown("## Mapa Coroplético de Pisos/Casas en alquiler por Provincia")
        geojson_path = "data/spanish_provinces.geojson"
        try:
            with open(geojson_path, 'r') as f:
                geojson_data = json.load(f)
        except FileNotFoundError:
            st.error("El archivo 'spanish_provinces.geojson' no se encontró.")
            return

        gdf_spain = gpd.GeoDataFrame.from_features(geojson_data["features"])
        df_grouped = df.groupby('provincia').size().reset_index(name='counts')

        mapping = {
            'A Coruña': 'A Coruña', 'Alava': 'Araba', 'Albacete': 'Albacete', 'Alicante': 'Alacant',
            'Almeria': 'Almería', 'Asturias': 'Asturias', 'Avila': 'Ávila', 'Badajoz': 'Badajoz',
            'Barcelona': 'Barcelona', 'Burgos': 'Burgos', 'Caceres': 'Cáceres', 'Cadiz': 'Cádiz',
            'Cantabria': 'Cantabria', 'Castellon': 'Castelló', 'Ciudad Real': 'Ciudad Real',
            'Cordoba': 'Córdoba', 'Cuenca': 'Cuenca', 'Girona': 'Girona', 'Granada': 'Granada',
            'Guadalajara': 'Guadalajara', 'Guipuzcoa': 'Gipuzcoa', 'Huelva': 'Huelva', 'Huesca': 'Huesca',
            'Islas Baleares': 'Illes Balears', 'Jaen': 'Jaén', 'La Rioja': 'La Rioja', 'Las Palmas': 'Las Palmas',
            'Leon': 'León', 'Lleida': 'Lleida', 'Lugo': 'Lugo', 'Madrid': 'Madrid', 'Malaga': 'Málaga',
            'Melilla': 'Melilla', 'Murcia': 'Murcia', 'Navarra': 'Navarra', 'Ourense': 'Ourense',
            'Palencia': 'Palencia', 'Pontevedra': 'Pontevedra', 'Salamanca': 'Salamanca',
            'Santa Cruz de Tenerife': 'Santa Cruz de Tenerife', 'Segovia': 'Segovia', 'Sevilla': 'Sevilla',
            'Soria': 'Soria', 'Tarragona': 'Tarragona', 'Teruel': 'Teruel', 'Toledo': 'Toledo',
            'Valencia': 'València', 'Valladolid': 'Valladolid', 'Vizcaya': 'Bizkaia', 'Zamora': 'Zamora',
            'Zaragoza': 'Zaragoza'
        }

        gdf_spain['provincia_normalized'] = gdf_spain['provincia'].map(mapping)
        df_grouped['provincia_normalized'] = df_grouped['provincia'].map(mapping)

        gdf_spain = gdf_spain.merge(df_grouped, on='provincia_normalized', how='left')

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
        st.components.v1.html(m._repr_html_(), height=525)

    mapa_coropletico(df)

    st.write("")

    # OUTLIERS PRECIO
    def analizar_outliers_precio(df):
        df = df.copy()
        st.markdown("## Análisis de valores atípicos en inmuebles en venta")
        df['log_precio'] = np.log(df['precio'])
        df['z_score'] = (df['log_precio'] - df['log_precio'].mean()) / df['log_precio'].std()
        df['outlier'] = df['z_score'].apply(lambda x: 'Outlier' if np.abs(x) > 3 else 'Normal')

        fig = px.histogram(df, x='log_precio', color='outlier',
                           title='Histograma de precios en escala logarítmica',
                           color_discrete_map={'Outlier': 'red', 'Normal': 'green'})

        precios_redondos = [1000, 5000, 20000, 80000, 250000, 1000000, 5000000, 20000000]
        tickvals = np.log(precios_redondos)

        fig.update_layout(
            xaxis_title='Precio',
            yaxis_title='Frecuencia',
            xaxis=dict(tickmode='array', tickvals=tickvals, ticktext=precios_redondos)
        )

        st.plotly_chart(fig)

    analizar_outliers_precio(df)

    st.write("")

    # SCATTER SEGMENTADO
    def scatter_segmentado(df):
        st.markdown("## Relación precio/superficie segmentado por Comunidad Autonoma de los inmuebles en venta")

        Q1 = df['superficie'].quantile(0.05)
        Q3 = df['superficie'].quantile(0.95)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        filtered_df = df[(df['superficie'] >= lower_bound) & (df['superficie'] <= upper_bound) & (df['superficie'] >= 30)]

        fig = px.scatter(filtered_df, x='superficie', y='precio', color='comunidad_autonoma',
                         labels={'superficie': 'Superficie (m²)', 'precio': 'Precio (€)'},
                         opacity=0.5)

        st.plotly_chart(fig)

    scatter_segmentado(df)

    st.write("")

    # PRECIO PROMEDIO POR ANTIGÜEDAD
    def plot_precio_promedio(df):
        if 'antiguedad' in df.columns and 'precio' in df.columns:
            df_filtered = df.dropna(subset=['antiguedad', 'precio']).copy()

            antiguedad_map = {
                '< 5': 'Menos de 5 años', '5-10': 'Entre 5 y 10 años', '10-20': 'Entre 10 y 20 años',
                '20-30': 'Entre 20 y 30 años', '30-50': 'Entre 30 y 50 años', '> 50': 'Más de 50 años'
            }

            df_filtered['antiguedad'] = df_filtered['antiguedad'].map(antiguedad_map)

            precio_promedio = df_filtered.groupby('antiguedad')['precio'].mean()
            columnas_ordenadas = [
                "Menos de 5 años", "Entre 5 y 10 años", "Entre 10 y 20 años",
                "Entre 20 y 30 años", "Entre 30 y 50 años", "Más de 50 años"
            ]
            precio_promedio = precio_promedio.reindex(columnas_ordenadas)

            st.write("### Precio Promedio por Antigüedad")
            fig = px.bar(precio_promedio, x=precio_promedio.index, y=precio_promedio.values,
                         title='Relación entre Precio Promedio y Antigüedad',
                         labels={'x': 'Antigüedad', 'y': 'Precio Promedio'})
            fig.update_layout(xaxis_title='Antigüedad', yaxis_title='Precio Promedio')

            st.plotly_chart(fig)
        else:
            st.error("Las columnas 'antiguedad' o 'precio' no existen en el DataFrame.")

    plot_precio_promedio(df)

    st.write("")

    # PRECIO PROMEDIO POR PROVINCIA Y COMUNIDAD
    def plot_precio_por_provincia_y_comunidad(df):
        if 'provincia' in df.columns and 'precio' in df.columns:
            df_limpio = df.dropna(subset=['provincia', 'precio'])
            precio_por_provincia = df_limpio.groupby('provincia')['precio'].mean().sort_values(ascending=False)
            media_precio_provincia = df_limpio['precio'].mean()

            st.write("### Precio Promedio por Provincia")
            fig_provincia = px.bar(precio_por_provincia, x=precio_por_provincia.index, y=precio_por_provincia.values,
                                   labels={'x': 'Provincia', 'y': 'Precio Promedio'})

            fig_provincia.add_shape(type='line', x0=-0.5, y0=media_precio_provincia,
                                    x1=len(precio_por_provincia) - 0.5, y1=media_precio_provincia,
                                    line=dict(color='red', dash='dash'))

            fig_provincia.update_layout(xaxis_title='Provincia', yaxis_title='Precio Promedio',
                                        xaxis=dict(tickangle=45))

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

            fig_comunidad.add_shape(type='line', x0=-0.5, y0=media_precio_comunidad,
                                    x1=len(precio_por_comunidad) - 0.5, y1=media_precio_comunidad,
                                    line=dict(color='red', dash='dash'))

            fig_comunidad.update_layout(xaxis_title='Comunidad Autónoma', yaxis_title='Precio Promedio',
                                        xaxis=dict(tickangle=45))

            st.plotly_chart(fig_comunidad)
        else:
            st.error("Las columnas 'comunidad_autonoma' o 'precio' no existen en el DataFrame.")

    plot_precio_por_provincia_y_comunidad(df)

if __name__ == "__main__":
    eda_page()
