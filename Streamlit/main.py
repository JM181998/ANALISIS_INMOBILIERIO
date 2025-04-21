import streamlit as st
from app.pages import home, csv_loader as database, about_us
from app.pages.compra_venta import eda as cv_eda, comparador as cv_comparador, clustering as cv_clustering, classification as cv_classification, regression as cv_regression
from app.pages.alquileres import eda as alq_eda, comparador as alq_comparador, clustering as alq_clustering, classification as alq_classification, regression as alq_regression

def main():
    st.set_page_config(page_title="Análisis de Datos de Pisos.com", layout="wide")

    # Sidebar
    st.sidebar.title("Navegación")
    main_page = st.sidebar.selectbox("Selecciona una sección principal", [
        "Página Principal", 
        "Compra/Venta", 
        "Alquileres", 
        "Base de Datos", 
        "Sobre nosotros"
    ], index=0)

    if main_page == "Página Principal":
        home.main()
    elif main_page == "Compra/Venta":
        sub_page = st.sidebar.selectbox("Selecciona una subpágina", [
            "Análisis de datos", "Comparador", "Clustering", "Clasificación", "Regresión"])
        if sub_page == "Análisis de datos":
            cv_eda.eda_page()
        elif sub_page == "Comparador":
            cv_comparador.comparador_page()
        elif sub_page == "Clustering":
            cv_clustering.clustering_page()
        elif sub_page == "Clasificación":
            cv_classification.classification_page()
        elif sub_page == "Regresión":
            cv_regression.regression_page()
    elif main_page == "Alquileres":
        sub_page = st.sidebar.selectbox("Selecciona una subpágina", [
            "Análisis de datos", "Comparador", "Clustering", "Clasificación", "Regresión"])
        if sub_page == "Análisis de datos":
            alq_eda.eda_page()
        elif sub_page == "Comparador":
            alq_comparador.comparador_page()
        elif sub_page == "Clustering":
            alq_clustering.clustering_page()
        elif sub_page == "Clasificación":
            alq_classification.classification_page()
        elif sub_page == "Regresión":
            alq_regression.regression_page()
    elif main_page == "Base de Datos":
        database.main()
    elif main_page == "Sobre nosotros":
        about_us.main()

if __name__ == "__main__":
    main()
