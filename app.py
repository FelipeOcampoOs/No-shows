import streamlit as st
import pandas as pd
import joblib
from urllib.request import urlopen
from io import BytesIO

# Cargar modelo desde Hugging Face y scaler desde el repositorio
@st.cache_resource
def load_model_and_scaler():
    model_url = "https://huggingface.co/felipeocampo/no-shows/resolve/main/modelnoshows.joblib"
    model = joblib.load(urlopen(model_url))

    scaler = joblib.load("scaler.joblib")  # este archivo debe estar en tu repo
    return model, scaler

model, scaler = load_model_and_scaler()

st.title("Predicción de insistencias médicas")

uploaded_file = st.file_uploader("Sube tu archivo .xlsx", type="xlsx")

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Escalar los datos
    X_scaled = scaler.transform(df)

    # Predicción
    pred = model.predict(X_scaled)
    df["Predicción"] = pred
    df["Predicción"] = df["Predicción"].replace({0: "Inasistencia", 1: "Asistencia"})

    st.success("Predicción completada")

    # Exportar archivo
    output = BytesIO()
    df.to_excel(output, index=False)
    st.download_button(
        label="Descargar archivo con predicciones",
        data=output.getvalue(),
        file_name="predicciones.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
