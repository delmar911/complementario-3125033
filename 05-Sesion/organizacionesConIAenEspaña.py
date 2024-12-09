import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Datos en formato estructurado
datos = {
    "Año": [
        "2021", "2022","2023","2024"
    ],
    "IA(%)": [
        4.5 ,10.1, 15.6,12.4
    ]
}

df = pd.DataFrame(datos)

# Convertir la columna de año en fechas para hacer predicción a través del tiempo
dates = pd.to_datetime(df["Año"])

# Convertir las fechas a valores numéricos para usar en la regresión
df["date_Num"] = (dates - dates.min()).dt.days

# Datos para regresión: usaremos la variable numérica de fecha como X y el porcentaje de IA como y
X = df[["date_Num"]]  # La variable independiente será la fecha en formato numérico
y = df["IA(%)"]  # La variable dependiente será el porcentaje de IA

# Crear y ajustar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X, y)

# Predicción para la línea de regresión
X_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = modelo.predict(X_pred)

# Visualización del gráfico
plt.figure(figsize=(12, 6))
plt.scatter(dates, y, color="blue", label="Datos reales")
plt.plot(pd.to_datetime(df["Año"].min()) + pd.to_timedelta(X_pred.flatten(), unit='d'), y_pred, color="red", label="Regresión lineal")

# Configuración del gráfico
plt.title("Porcentaje de empresas que emplean tecnologías de Inteligencia Artificial (IA) en España", fontsize=14)
plt.xlabel("Año", fontsize=12)
plt.ylabel("IA(%)", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
