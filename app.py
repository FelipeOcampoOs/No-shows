import streamlit as st
import pandas as pd
import joblib
from io import BytesIO

st.title("🩺 Predicción de Asistencia a Citas Médicas")

uploaded_file = st.file_uploader("Sube tu archivo .xlsx", type="xlsx")

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # 1. Eliminar filas con valores nulos
    df = df.dropna().reset_index(drop=True)

    # 2. Columnas a conservar
    columnas_id = ['ID', 'Paciente', 'Nº documento']
    columnas_extra = ['Interlocutor', 'Un.org.planificada']
    columnas_a_remover = columnas_id + columnas_extra + ['Tipo de cita']
    df_ids = df[columnas_id + columnas_extra]

    # 3. Preparar datos
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

    # Validar columnas
    if all(col in df_modelo.columns for col in orden_columnas):
        df_modelo = df_modelo[orden_columnas]

        # 4. Cargar modelo y scaler
        scaler = joblib.load("scaler.joblib")
        model = joblib.load("modelnoshows.joblib")

        # 5. Escalar y predecir
        X_scaled = scaler.transform(df_modelo)
        pred = model.predict(X_scaled)

        # 6. Agregar predicciones
        df_ids["Predicción"] = pred
        df_ids["Predicción"] = df_ids["Predicción"].replace({0: "Inasistencia", 1: "Asistencia"})

        st.success("✅ Predicción completada.")
        st.dataframe(df_ids)

        # 7. Descargar archivo
        output = BytesIO()
        df_ids.to_excel(output, index=False)
        output.seek(0)

        st.download_button(
            label="📥 Descargar archivo con predicciones",
            data=output,
            file_name="predicciones_resultado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.error("❌ Faltan columnas requeridas para la predicción.")

