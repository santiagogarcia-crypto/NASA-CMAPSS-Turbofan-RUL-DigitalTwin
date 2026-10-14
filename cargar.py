import pandas as pd

# 1. Cargar el dataset procesado
ruta_base = r'F:\Santiago_IA\Digital_Twin_Jav\TURBOFAN'
df = pd.read_csv(f'{ruta_base}\\train_FD001_procesado.csv')

# ------------------------------------------------------------------
# TABLA 1: Dim_Engine (Tabla Dimensión Motor)
# ------------------------------------------------------------------
dim_engine = df.groupby('engine_id').agg(
    max_cycles=('cycle', 'max')
).reset_index()

# ------------------------------------------------------------------
# TABLA 2: Fact_Sensor_Readings (Tabla de Hechos de Sensores)
# ------------------------------------------------------------------
sensor_cols = ['engine_id', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + [col for col in df.columns if col.startswith('s_')]
fact_sensor = df[sensor_cols].copy()
fact_sensor['reading_id'] = fact_sensor['engine_id'].astype(str) + "_" + fact_sensor['cycle'].astype(str)

# ------------------------------------------------------------------
# TABLA 3: Fact_RUL_Predictions (Tabla de Hechos de RUL Real / Predicha)
# ------------------------------------------------------------------
fact_rul = df[['engine_id', 'cycle', 'RUL']].copy()
fact_rul['prediction_id'] = fact_rul['engine_id'].astype(str) + "_" + fact_rul['cycle'].astype(str)

# Categorizar el estado de salud operativo según RUL
fact_rul['health_status'] = fact_rul['RUL'].apply(
    lambda x: 'ROJO' if x <= 20 else ('AMARILLO' if x <= 50 else 'VERDE')
)

# ------------------------------------------------------------------
# EXPORTAR LAS TABLAS PARA POWER BI
# ------------------------------------------------------------------
dim_engine.to_csv(f'{ruta_base}\\Dim_Engine.csv', index=False)
fact_sensor.to_csv(f'{ruta_base}\\Fact_Sensor_Readings.csv', index=False)
fact_rul.to_csv(f'{ruta_base}\\Fact_RUL_Predictions.csv', index=False)

print("\n==================================================")
print("  MODELO ENTIDAD-RELACIÓN (DER) GENERADO CON ÉXITO ")
print("==================================================")
print(f"1. Dim_Engine.csv          -> {dim_engine.shape[0]} motores")
print(f"2. Fact_Sensor_Readings.csv -> {fact_sensor.shape[0]} lecturas temporales")
print(f"3. Fact_RUL_Predictions.csv -> {fact_rul.shape[0]} registros de degradación")
print("\n Archivos listos para ser importados a Power BI.")