import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from urllib.request import urlopen, URLError, HTTPError

# --- Cargar modelo y scaler ---
@st.cache_resource
def load_model_and_scaler():
    try:
        model_url = "https://huggingface.co/felipeocampo/no-shows/resolve/main/modelnoshows.joblib"
        with urlopen(model_url) as model_file:
            model = joblib.load(model_file)

        scaler = joblib.load("scaler.joblib")  # Archivo local
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
        st.error("❌ No se encontró el archivo local 'scaler.joblib'.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error inesperado: {str(e)}")
        st.stop()

model, scaler = load_model_and_scaler()

# --- Título de la app ---
st.title("🩺 Predicción de Inasistencia Médica")

# --- Cargar archivo del usuario ---
uploaded_file = st.file_uploader("Sube tu archivo .xlsx", type=["xlsx", "XLSX"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df = df.dropna().reset_index(drop=True)

    # --- Columnas ---
    columnas_id = ['ID', 'Paciente', 'Nº documento']
    columnas_extra = ['Interlocutor', 'Un.org.planificada']
    columnas_a_remover = columnas_id + columnas_extra + ['Tipo de cita']

    try:
        df_ids = df[columnas_id + columnas_extra]
    except KeyError:
        st.error("❌ Faltan columnas identificadoras requeridas en el archivo.")
        st.stop()

    df_modelo = df.drop(columns=columnas_a_remover, errors='ignore')

    # --- Renombrar columnas ---
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

    # --- Orden esperado por el modelo ---
    orden_columnas = [
        'Age', 'Sex', 'Insurance Type', 'Number of Diseases',
        'Recent Hospitalization', 'Number of Medications', 'Hour', 'Day',
        'Month', 'Creation to Assignment Interval',
        'Number of Previous Attendance', 'Number of Previous Non-Attendance'
    ]

    # Verificar columnas requeridas
    missing = [col for col in orden_columnas if col not in df_modelo.columns]
    if missing:
        st.error(f"❌ Faltan las siguientes columnas necesarias: {missing}")
        st.stop()

    df_modelo = df_modelo[orden_columnas]

    # --- Predicción ---
    X_scaled = scaler.transform(df_modelo)
    pred = model.predict(X_scaled)

    df_ids["Predicción"] = pred
    df_ids["Predicción"] = df_ids["Predicción"].replace({0: "Inasistencia", 1: "Asistencia"})

    st.success("✅ Predicción completada.")
    st.dataframe(df_ids)

    # --- Descargar archivo con resultados ---
    output = BytesIO()
    df_ids.to_excel(output, index=False)
    output.seek(0)

    st.download_button(
        label="📥 Descargar archivo con predicciones",
        data=output,
        file_name="predicciones_resultado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

