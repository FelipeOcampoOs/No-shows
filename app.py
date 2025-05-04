import streamlit as st
import pandas as pd
import joblib
from urllib.request import urlopen
from io import BytesIO

# Cargar modelo desde Hugging Face y scaler desde el repositorio local
@st.cache_resource
def load_model_and_scaler():
    model_url = "https://huggingface.co/felipeocampo/no-shows/resolve/main/modelnoshows.joblib"
    model = joblib.load(urlopen(model_url))
    scaler = joblib.load("scaler.joblib")  # debe estar en tu repositorio
    return model, scaler

model, scaler = load_model_and_scaler()

st.title("Predicción de Inasistencia Médica")
uploaded_file = st.file_uploader("Sube un archivo .xlsx", type="xlsx")


if uploaded_file:
    try:
        # 1. Cargar el archivo Excel
        df = pd.read_excel(uploaded_file)

        # 2. Eliminar filas con valores nulos
        df = df.dropna()

        # 3. Separar columnas para conservar
        columnas_id = ['ID', 'Paciente', 'Nº documento']
        columnas_extra = ['Interlocutor', 'Un.org.planificada']
        columnas_a_remover = columnas_id + columnas_extra + ['Tipo de cita']

        df_ids = df[columnas_id + columnas_extra].copy()

        # 4. Preparar datos para predicción
        df_modelo = df.drop(columns=columnas_a_remover, errors='ignore')

        # 5. Renombrar columnas al formato del modelo
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

        # 6. Reordenar columnas al orden esperado por el modelo
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

        # 8. Agregar predicción en español
        df_ids["Predicción"] = pred
        df_ids["Predicción"] = df_ids["Predicción"].replace({0: "Inasistencia", 1: "Asistencia"})

        # 9. Mostrar resultados
        st.success("✅ Predicción completada correctamente.")
        st.dataframe(df_ids)

        # 10. Descargar archivo
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

