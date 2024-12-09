import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Datos en formato estructurado
datos = {
    "Year": [
        "2018", "2019", "2020", "2021", "2022","2023","2024", "2025"
    ],
    "Organizations Using AI": [
        35, 37, 40, 42, 45, 48, 50, 55
    
    ]
}

df = pd.DataFrame(datos)

# Convertir la semana a índice numérico
dates = pd.to_datetime(df["Year"])
df["date_Num"] = (dates - dates.min()).dt.days

# Datos para regresión
X = df[["date_Num"]]
y = df["Organizations Using AI"]  # Ejemplo: análisis de Organizations Using AI

# Crear y ajustar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X, y)

# Predicción para la línea de regresión
X_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = modelo.predict(X_pred)

# Visualización del gráfico
plt.figure(figsize=(12, 6))
plt.plot(dates, y, "o", label="Datos reales (Organizations Using AI)", color="blue")
plt.plot(pd.to_datetime(dates.min()) + pd.to_timedelta(X_pred.flatten(), unit='d'), y_pred, "-", color="red", label="Regresión lineal")

# Configuración del gráfico
plt.title("Regresión lineal para organizaciones que usan IA a través tiempo", fontsize=14)
plt.xlabel("Fecha", fontsize=12)
plt.ylabel("Organizaciones que usan IA en % ", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
