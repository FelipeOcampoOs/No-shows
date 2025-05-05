import streamlit as st
import pandas as pd
import joblib
import gzip
from io import BytesIO

st.set_page_config(page_title="Predicción de Inasistencia", layout="centered")
st.title("🩺 Predicción de inasistencias médicas")

# --- Cargar modelo y scaler desde archivos locales ---
@st.cache_resource
def load_model_and_scaler():
    try:
        with gzip.open("modelnoshows.joblib.gz", "rb") as f:
            model = joblib.load(f)

        scaler = joblib.load("scaler.joblib")
        return model, scaler

    except Exception as e:
        st.error(f"❌ Error al cargar modelo o scaler: {str(e)}")
        st.stop()

# --- Preprocesamiento de datos ---
def preprocesar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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

# --- Subida de archivo Excel ---
uploaded_file = st.file_uploader("📁 Sube tu archivo .xlsx con citas médicas", type=["xlsx", "XLSX"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        df_modelo, df_ids = preprocesar(df)

        columnas_esperadas = [
            'Age', 'Sex', 'Insurance Type', 'Number of Diseases',
            'Recent Hospitalization', 'Number of Medications', 'Hour', 'Day',
            'Month', 'Creation to Assignment Interval',
            'Number of Previous Attendance', 'Number of Previous Non-Attendance'
        ]
        df_modelo = df_modelo[columnas_esperadas]

        # --- Cargar modelo y predecir ---
        model, scaler = load_model_and_scaler()
        X_scaled = scaler.transform(df_modelo)
        pred = model.predict(X_scaled)

        # --- Agregar predicción ---
        df["Predicción"] = pred
        df["Predicción"] = df["Predicción"].replace({0: "Inasistencia", 1: "Asistencia"})"})

        # --- Preparar archivo para descarga ---
        output = BytesIO()
        df_ids.to_excel(output, index=False)
        output.seek(0)

        st.success("✅ Archivo procesado correctamente.")
        st.download_button(
            label="📥 Descargar archivo con predicciones",
            data=output,
            file_name="predicciones_resultado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"❌ Error durante el procesamiento: {str(e)}")


