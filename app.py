import streamlit as st
import pandas as pd
import joblib
from urllib.request import urlopen
from io import BytesIO

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(urlopen("https://huggingface.co/felipeocampo/no-shows/resolve/main/modelnoshows.joblib"))
    scaler = joblib.load("scaler.joblib")
    return model, scaler

model, scaler = load_model_and_scaler()

st.title("Predicción de Inasistencia Médica")
uploaded_file = st.file_uploader("Sube un archivo .xlsx", type=["xlsx", "XLSX"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)

        df = df.dropna()

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

        orden_columnas = [
            'Age', 'Sex', 'Insurance Type', 'Number of Diseases',
            'Recent Hospitalization', 'Number of Medications', 'Hour', 'Day',
            'Month', 'Creation to Assignment Interval',
            'Number of Previous Attendance', 'Number of Previous Non-Attendance'
        ]

        df_modelo = df_modelo[orden_columnas]

        X_scaled = scaler.transform(df_modelo)
        pred = model.predict(X_scaled)

        df_ids["Predicción"] = pred
        df_ids["Predicción"] = df_ids["Predicción"].replace({0: "Inasistencia", 1: "Asistencia"})

        st.success("✅ Predicción completada correctamente.")
        st.dataframe(df_ids)

        output = BytesIO()
        df_ids.to_excel(output, index=False)
        st.download_button(
            label="📥 Descargar resultados en Excel",
            data=output.getvalue(),
            file_name="predicciones_resultado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")

