import streamlit as st
import pandas as pd
import joblib
from urllib.request import urlopen
from io import BytesIO

# Cargar modelo desde Hugging Face y scaler local
@st.cache_resource
def load_model_and_scaler():
    model_url = "https://huggingface.co/felipeocampo/no-shows/resolve/main/modelnoshows.joblib"
    model = joblib.load(urlopen(model_url))
    scaler = joblib.load("scaler.joblib")  # este archivo debe estar en tu repo GitHub
    return model, scaler

model, scaler = load_model_and_scaler()

st.title("Predicción de insistencias médicas")

uploaded_file = st.file_uploader("Sube tu archivo .xlsx", type="xlsx")

if uploaded_file:
    # 1. Leer archivo
    df = pd.read_excel(uploaded_file)

    # 2. Eliminar filas con nulos
    df = df.dropna()

    # 3. Columnas de identificación y no usadas
    columnas_id = ['ID', 'Paciente', 'Nº documento']
    columnas_extra = ['Interlocutor', 'Un.org.planificada']
    columnas_a_remover = columnas_id + columnas_extra + ['Tipo de cita']

    df_ids = df[columnas_id + columnas_extra].copy()

    # 4. Preparar columnas para predicción
    df_modelo = df.drop(columns=columnas_a_remover, errors='ignore')

    # 5. Renombrar columnas
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

    # 6. Reordenar columnas
    orden_columnas = [
        'Age', 'Sex', 'Insurance Type', 'Number of Diseases',
        'Recent Hospitalization', 'Number of Medications', 'Hour', 'Day',
        'Month', 'Creation to Assignment Interval',
        'Number of Previous Attendance', 'Number of Previous Non-Attendance'
    ]
    df_modelo = df_modelo[orden_columnas]

    # 7. Escalar y predecir
    X_scaled = scaler.transform(df_modelo)
    pred = model.predict(X_scaled)

    # 8. Combinar con columnas identificadoras y resultado
    df_ids["Predicción"] = pred
    df_ids["Predicción"] = df_ids["Predicción"].replace({0: "Inasistencia", 1: "Asistencia"})

    st.success("Predicción completada")
    st.dataframe(df_ids)

    # 9. Exportar resultado
    output = BytesIO()
    df_ids.to_excel(output, index=False)
    st.download_button(
        label="Descargar archivo con predicciones",
        data=output.getvalue(),
        file_name="predicciones.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

