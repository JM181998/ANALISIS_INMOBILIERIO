import streamlit as st

def main():
    participants = [
        {
            "name": "Enrique Gómez",
            "role": "Creación de modelos y clustering.",
            "bio": "https://www.linkedin.com/in/enrique-manuel-gómez-murciego"
        },
        {
            "name": "Jorge Gazulla",
            "role": "Limpieza y Streamlit.",
            "bio": "https://www.linkedin.com/in/jorge-gazulla"
        },
        {
            "name": "Juan Manuel Fuentes",
            "role": "Extracción de datos y creación de modelos.",
            "bio": "https://es.linkedin.com/in/juanma-fuentes"
        },
        {
            "name": "Fran Polo",
            "role": "Modelado y gestión de base de datos.",
            "bio": "https://www.linkedin.com/in/franpolog"
        },
        {
            "name": "Carolina Merlo",
            "role": "Análisis exploratorio de datos.",
            "bio": "https://www.linkedin.com/in/carolina-merlo"
        }
    ]

    st.title("Sobre Nosotros")
    for participant in participants:
        st.subheader(participant['name'])
        st.text(f"Rol: {participant['role']}")
        st.text(f"{participant['bio']}")
        st.markdown("---")

if __name__ == "__main__":
    main()