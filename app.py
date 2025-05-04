import streamlit as st
import pandas as pd
import joblib
from urllib.request import urlopen
from io import BytesIO

# --- Cargar modelo desde Hugging Face y scaler desde el repositorio ---
@st.cache_resource
def load_model_and_scaler():
    model_url = "https://huggingface.co/felipeocampo/no-shows/resolve/main/modelnoshows.joblib"
    model = joblib.load(urlopen(model_url))
    scaler = joblib.load("scaler.joblib")  # Este archivo debe estar en tu repo
    return model, scaler

# Cargar modelo y scaler
model, scaler = load_model_and_scaler()

# --- Título de la app ---
st.title("🩺 Predicción de Inasistencia Médica")
st.write("Sube un archivo Excel con los datos de los pacientes para predecir la probabilidad de inasistencia.")

# --- Subida del archivo ---
uploaded_file = st.file_uploader(
    "📎 Sube tu archivo .xlsx",
    type=["xlsx", "XLSX"],
    key="file_upload"
)

if uploaded_file is not None:
    try:
        # 1. Leer Excel
        df = pd.read_excel(uploaded_file)

        # 2. Eliminar filas con valores nulos
        df = df.dropna()

        # 3. Definir columnas clave
        columnas_id = ['ID', 'Paciente', 'Nº documento']
        columnas_extra = ['Interlocutor', 'Un.org.planificada']
        columnas_a_remover = columnas_id + columnas_extra + ['Tipo de cita']

        # 4. Separar identificadores
        df_ids = df[columnas_id + columnas_extra].copy()

        # 5. Preparar columnas para el modelo
        df_modelo = df.drop(columns=columnas_a_remover, errors='ignore')

        # 6. Renombrar columnas del español al inglés
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

        # 7. Reordenar columnas
        orden_columnas = [
            'Age', 'Sex', 'Insurance Type', 'Number of Diseases',
            'Recent Hospitalization', 'Number of Medications', 'Hour', 'Day',
            'Month', 'Creation to Assignment Interval',
            'Number of Previous Attendance', 'Number of Previous Non-Attendance'
        ]
        df_modelo = df_modelo[orden_columnas]

        # 8. Escalar y predecir
        X_scaled = scaler.transform(df_modelo)
        pred = model.predict(X_scaled)

        # 9. Agregar resultados
        df_ids["Predicción"] = pred
        df_ids["Predicción"] = df_ids["Predicción"].replace({0: "Inasistencia", 1: "Asistencia"})

        # 10. Mostrar en pantalla
        st.success("✅ Predicción completada exitosamente.")
        st.dataframe(df_ids)

        # 11. Descargar como Excel
        output = BytesIO()
        df_ids.to_excel(output, index=False)
        st.download_button(
            label="📥 Descargar resultados en Excel",
            data=output.getvalue(),
            file_name="predicciones_resultado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"❌ Ocurrió un error al procesar el archivo: {e}")

else:
    st.info("⬆️ Esperando que subas un archivo Excel (.xlsx).")

