import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Definir rutas y columnas
ruta_base = r'F:\Santiago_IA\Digital_Twin_Jav\TURBOFAN'
index_names = ['engine_id', 'cycle']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = [f's_{i}' for i in range(1, 22)]
col_names = index_names + setting_names + sensor_names

# Sensores útiles (removiendo los 7 sensores constantes)
sensores_utiles = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
features = ['setting_1', 'setting_2', 'setting_3'] + sensores_utiles

print("Cargando datos de entrenamiento y prueba...")
# 2. Cargar Dataset de Entrenamiento Procesado
df_train = pd.read_csv(f'{ruta_base}\\train_FD001_procesado.csv')

# 3. Cargar Dataset de Prueba y tomar el último ciclo de cada motor
df_test = pd.read_csv(f'{ruta_base}\\test_FD001.txt', sep=r'\s+', header=None, names=col_names)
df_test_last = df_test.groupby('engine_id').last().reset_index()

# 4. Cargar RUL Real de Prueba
df_rul_real = pd.read_csv(f'{ruta_base}\\RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL_real'])
df_rul_real['RUL_real'] = df_rul_real['RUL_real'].clip(upper=125)

# 5. Entrenar Modelo IA (Random Forest Regressor)
X_train = df_train[features]
y_train = df_train['RUL']

print("Entrenando modelo Random Forest Regressor para RUL...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 6. Evaluacion sobre el conjunto Test
X_test = df_test_last[features]
y_pred = rf_model.predict(X_test)
y_real = df_rul_real['RUL_real'].values

# 7. Métricas de Rendimiento Evaluativas
rmse = np.sqrt(mean_squared_error(y_real, y_pred))
mae = mean_absolute_error(y_real, y_pred)

print("\n==================================================")
print("   RESULTADOS DEL MODELO IA - ESTIMACIÓN DE RUL   ")
print("==================================================")
print(f"-> Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f} ciclos")
print(f"-> Error Absoluto Medio (MAE):            {mae:.2f} ciclos")
print("==================================================")

# 8. Exportar Resultados de Predicción
resultados = pd.DataFrame({
    'engine_id': df_test_last['engine_id'],
    'RUL_Real': y_real,
    'RUL_Predicho': np.round(y_pred, 1)
})
resultados.to_csv(f'{ruta_base}\\Predicciones_RUL_IA.csv', index=False)
print(f"\nPredicciones guardadas exitosamente en:\n{ruta_base}\\Predicciones_RUL_IA.csv")