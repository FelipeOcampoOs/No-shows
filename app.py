import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
import gzip

st.set_page_config(page_title="Predicción de Inasistencia Médica", layout="centered")
st.title("🩺 Predicción de Inasistencia Médica")

# --- Cargar modelo y scaler ---
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

# --- Preprocesamiento ---
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

# --- Archivo de entrada ---
uploaded_file = st.file_uploader("📁 Sube tu archivo .xlsx con las citas médicas", type=["xlsx", "XLSX"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("Vista previa del archivo original:")
    st.dataframe(df.head())

    try:
        df_modelo, df_ids = preprocesar_dataframe(df)
    except Exception as e:
        st.error(f"❌ Error durante el preprocesamiento: {str(e)}")
        st.stop()

    st.success("✅ Preprocesamiento completado.")

    columnas_esperadas = [
        'Age', 'Sex', 'Insurance Type', 'Number of Diseases',
        'Recent Hospitalization', 'Number of Medications', 'Hour', 'Day',
        'Month', 'Creation to Assignment Interval',
        'Number of Previous Attendance', 'Number of Previous Non-Attendance'
    ]
    faltantes = [col for col in columnas_esperadas if col not in df_modelo.columns]
    if faltantes:
        st.error(f"❌ Faltan columnas requeridas para el modelo: {faltantes}")
        st.stop()

    df_modelo = df_modelo[columnas_esperadas]

    # Cargar modelo y hacer predicción
    model, scaler = load_model_and_scaler()
    X_scaled = scaler.transform(df_modelo)
    pred = model.predict(X_scaled)

    # Agregar predicción
    df_ids["Predicción"] = pred
    df_ids["Predicción"] = df_ids["Predicción"].replace({0: "Inasistencia", 1: "Asistencia"})

    # Mostrar y permitir descarga
    st.success("✅ Predicción completada.")
    st.subheader("Resultados:")
    st.dataframe(df_ids)

    output = BytesIO()
    df_ids.to_excel(output, index=False)
    output.seek(0)

    st.download_button(
        label="📥 Descargar archivo con predicciones",
        data=output,
        file_name="predicciones_resultado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

