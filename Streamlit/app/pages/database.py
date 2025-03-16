import streamlit as st
import mysql.connector
import pandas as pd
from PIL import Image


def cargar_datos_bd(query):
    """
    Carga los datos desde una base de datos MySQL usando st.secrets.
    """
    conn = mysql.connector.connect(
        host=st.secrets["database"]["host"],
        user=st.secrets["database"]["user"],
        password=st.secrets["database"]["password"],
        database=st.secrets["database"]["name"],
        auth_plugin='mysql_native_password'  # Especificar el plugin de autenticación
    )
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def crear_base_de_datos():
    """
    Crea la base de datos en MySQL Workbench si no existe.
    """
    conn = mysql.connector.connect(
        host=st.secrets["database"]["host"],
        user=st.secrets["database"]["user"],
        password=st.secrets["database"]["password"],
        auth_plugin='mysql_native_password'  # Especificar el plugin de autenticación
    )
    cursor = conn.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS analisis_inmobiliario")
    conn.close()


def inferir_tipo_sql(dtype):
    if pd.api.types.is_int64_dtype(dtype):
        return 'INT'
    elif pd.api.types.is_float_dtype(dtype):
        return 'FLOAT'
    else:
        return 'VARCHAR(255)'


def generar_schema(tabla, df):
    columnas = []
    for col, dtype in zip(df.columns, df.dtypes):
        tipo_sql = inferir_tipo_sql(dtype)
        columnas.append(f'`{col}` {tipo_sql}')
    return f'CREATE TABLE {tabla} (\n    ' + ',\n    '.join(columnas) + '\n);'


def crear_tablas():
    df_alquileres = pd.read_csv('data/alquileres_completo_limpio.csv')
    df_compras = pd.read_csv('data/compras_completo_limpio.csv', low_memory=False)

    sql_schema_alquileres = generar_schema('general_alquileres', df_alquileres)
    sql_schema_compras = generar_schema('general_compras', df_compras)

    #st.write("Esquema de la tabla general_alquileres:")
    #st.write(sql_schema_alquileres)
    #st.write("Esquema de la tabla general_compras:")
    #st.write(sql_schema_compras)

    connection = mysql.connector.connect(
        host=st.secrets["database"]["host"],
        user=st.secrets["database"]["user"],
        password=st.secrets["database"]["password"],
        database=st.secrets["database"]["name"],
        auth_plugin='mysql_native_password'  # Especificar el plugin de autenticación
    )

    cursor = connection.cursor()

    try:
        cursor.execute("DROP TABLE IF EXISTS general_alquileres")
        cursor.execute("DROP TABLE IF EXISTS general_compras")

        cursor.execute(sql_schema_alquileres)
        st.write("Tabla general_alquileres creada exitosamente.")

        cursor.execute(sql_schema_compras)
        st.write('Tabla general_compras creada exitosamente')

        connection.commit()

    except mysql.connector.Error as err:
        st.write(f'Error: {err}')

    cursor.close()
    connection.close()


def insertar_datos(tabla, df):
    connection = mysql.connector.connect(
        host=st.secrets["database"]["host"],
        user=st.secrets["database"]["user"],
        password=st.secrets["database"]["password"],
        database=st.secrets["database"]["name"],
        auth_plugin='mysql_native_password'  # Especificar el plugin de autenticación
    )

    cursor = connection.cursor()

    columnas = ', '.join([f"`{col}`" for col in df.columns])
    valores = ', '.join(['%s'] * len(df.columns))
    query = f"INSERT INTO {tabla} ({columnas}) VALUES ({valores})"

    for _, row in df.iterrows():
        #st.write(f"Insertando fila: {tuple(row)}")
        try:
            cursor.execute(query, tuple(row))
        except Exception as e:
            st.write(f"❌ Error al insertar fila: {e}")
            st.write(row)

    connection.commit()
    st.write(f"✅ {len(df)} registros insertados en '{tabla}'.")

    cursor.close()
    connection.close()
    st.write('Conexión cerrada')


def main():
    st.title("Importación de Datos desde MySQL")

    # Crear la base de datos y las tablas
    crear_base_de_datos()
    crear_tablas()

    # Insertar datos en las tablas
    df_alquileres = pd.read_csv('data/alquileres_completo_limpio.csv')
    insertar_datos('general_alquileres', df_alquileres)

    df_compras = pd.read_csv('data/compras_completo_limpio.csv', low_memory=False)
    insertar_datos('general_compras', df_compras)

    # Cargar datos de la tabla general_alquileres
    query_alquileres = "SELECT * FROM general_alquileres"
    df_alquileres = cargar_datos_bd(query_alquileres)
    st.subheader("Datos completos de nuestra base de datos de inmuebles en alquiler:")
    with st.expander("Mostrar tabla de alquileres"):
        st.dataframe(df_alquileres)

    # Cargar datos de la tabla general_compras
    query_compras = "SELECT * FROM general_compras"
    df_compras = cargar_datos_bd(query_compras)
    st.subheader("Datos completos de nuestra base de datos de inmuebles en venta:")
    with st.expander("Mostrar tabla de compras"):
        st.dataframe(df_compras)

    image = Image.open('app/images/db.png')
    st.image(image, caption='Modelo entidad-relación entre nuestras tablas', use_container_width=True, width=300)


if __name__ == "__main__":
    main()