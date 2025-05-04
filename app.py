import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from urllib.request import urlopen, URLError, HTTPError

st.set_page_config(page_title="No Shows - Predicción", layout="centered")

# --- Función para cargar el modelo (desde Hugging Face) y el scaler (local) ---
@st.cache_resource
def load_model_and_scaler():
    try:
        model_url = "https://huggingface.co/felipeocampo/no-shows/resolve/main/modelnoshows.joblib"
        with urlopen(model_url) as model_file:
            model = joblib.load(model_file)

        scaler = joblib.load("scaler.joblib")  # scaler local
        return model, scaler

    except HTTPError as e:
        if e.code == 429:
            st.error("⚠️ Hugging Face está limitando las descargas. Intenta más tarde.")
        else:
            st.error(f"❌ Error HTTP al descargar el modelo: {e}")
        st.stop()

    except URLError as e:
        st.error(f"❌ No se pudo acceder al modelo remoto: {e.reason}")
        st.stop()

    except FileNotFoundError:
        st.error("❌ El archivo 'scaler.joblib' no está en el repositorio.")
        st.stop()

    except Exception as e:
        st.error(f"❌ Error inesperado: {str(e)}")
        st.stop()

# --- Función de preprocesamiento ---
def preprocesar_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.dropna().reset_index(drop=True)

    columnas_id = ['ID', 'Paciente', 'Nº documento']
    columnas_extra = ['Interlocutor', 'Un.org.planificada']
    columnas_a_remover = columnas_id + columnas_extra + ['Tipo de cita']

    df_ids = df[columnas_id + columnas_extra].copy()
    df_modelo = df.drop(columns=columnas_a_remover, errors='ignore')

    df_modelo = df_modelo.rename(columns={
        "Edad": "Age",
        "Género": "Sex",
        "Tipo aseguradora": "Insurance Type",
        "Número de diagnósticos": "Number of Diseases",
        "Hospitalización reciente": "Recent Hospitalization",
        "Número de medicamentos": "Number of Medications",
        "Hora": "Hour",
        "Día de la semana": "Day",
        "Mes": "Month",
        "Nº intervalo": "Creation to Assignment Interval",
        "Asistencias previas": "Number of Previous Attendance",
        "Inasistencias previas": "Number of Previous Non-Attendance"
    })

    return df_modelo, df_ids

# --- Navegación lateral ---
st.sidebar.title("🧭 Navegación")
seccion = st.sidebar.radio("Selecciona módulo:", ["🔧 Preprocesamiento", "📈 Predicción"])

# --- Preprocesamiento ---
if seccion == "🔧 Preprocesamiento":
    st.title("🔧 Preprocesamiento de Datos")

    uploaded_file = st.file_uploader("Sube un archivo .xlsx", type=["xlsx", "XLSX"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.subheader("Vista previa del archivo original:")
        st.dataframe(df.head())

        df_modelo, df_ids = preprocesar_dataframe(df)

        st.success("✅ Preprocesamiento completado.")
        st.subheader("Variables para el modelo:")
        st.dataframe(df_modelo.head())

        # Exportar datos preprocesados
        output = BytesIO()
        df_modelo.to_excel(output, index=False)
        output.seek(0)

        st.download_button(
            label="📥 Descargar datos para predicción",
            data=output,
            file_name="datos_preprocesados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# --- Predicción ---
elif seccion == "📈 Predicción":
    st.title("📈 Predicción con Modelo")

    uploaded_file = st.file_uploader("Sube el archivo preprocesado (.xlsx)", type=["xlsx", "XLSX"])

    if uploaded_file:
        df_modelo = pd.read_excel(uploaded_file)
        model, scaler = load_model_and_scaler()

        columnas_esperadas = [
            'Age', 'Sex', 'Insurance Type', 'Number of Diseases',
            'Recent Hospitalization', 'Number of Medications', 'Hour', 'Day',
            'Month', 'Creation to Assignment Interval',
            'Number of Previous Attendance', 'Number of Previous Non-Attendance'
        ]
        faltantes = [col for col in columnas_esperadas if col not in df_modelo.columns]

        if faltantes:
            st.error(f"❌ Faltan columnas necesarias para el modelo: {faltantes}")
            st.stop()

        df_modelo = df_modelo[columnas_esperadas]

        X_scaled = scaler.transform(df_modelo)
        pred = model.predict(X_scaled)

        df_modelo["Predicción"] = pred
        df_modelo["Predicción"] = df_modelo["Predicción"].replace({0: "Inasistencia", 1: "Asistencia"})

        st.success("✅ Predicción completada.")
        st.dataframe(df_modelo)

        # Descargar archivo con predicción
        output = BytesIO()
        df_modelo.to_excel(output, index=False)
        output.seek(0)

        st.download_button(
            label="📥 Descargar archivo con predicciones",
            data=output,
            file_name="predicciones_resultado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

