import streamlit as st
from PIL import Image

def main():
    # Imagen de título con tres columnas y usar la columna del medio para centrar la imagen
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        image = Image.open('app/images/home_logo.png')
        st.image(image, caption='Proyecto Fin de Bootcamp HAB.', use_container_width=True)

    st.markdown("""
        ## Bienvenidos a la App de Análisis de Datos de Pisos.com
        Esta aplicación está diseñada para proporcionar un análisis exhaustivo de los datos de compra, venta y alquiler de propiedades en la página Pisos.com. A continuación, se presenta una breve descripción de las secciones disponibles en la app:

        ### Secciones de la App
        - **Exploratory Data Analysis**: Visualizaciones y análisis exploratorio de los datos.
        - **Compra/Venta**: Herramientas y modelos para analizar propiedades en compra/venta.
        - **Alquileres**: Herramientas y modelos para analizar propiedades en alquiler.
        - **Base de Datos**: Descripción de la arquitectura de la base de datos utilizada en el proyecto.
        - **Sobre nosotros**: Información sobre los integrantes del proyecto, con enlaces a LinkedIn y Github.

        ### Cómo Navegar
        Utiliza el menú de la izquierda para navegar entre las diferentes secciones de la app. Cada sección contiene visualizaciones interactivas y explicaciones detalladas para ayudarte a entender mejor los datos y los modelos utilizados.

        ### Objetivo del Proyecto
        El objetivo de este proyecto es proporcionar una herramienta interactiva y fácil de usar para analizar los datos de propiedades en Pisos.com, ayudando a los usuarios a tomar decisiones informadas sobre compra, venta y alquiler de propiedades.

        ¡Esperamos que disfrutes explorando los datos y modelos en esta app!
        """)

if __name__ == "__main__":
    main()