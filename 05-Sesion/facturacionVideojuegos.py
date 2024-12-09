import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Datos en formato estructurado
datos = {
    "Año": [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    "Facturación (millones de euros)": [
        314, 413, 511, 617, 713, 813, 920, 1104, 1303, 1511, 1723
    ]
}

# Crear DataFrame
df = pd.DataFrame(datos)

# Preparar los datos para el modelo de regresión
X = df["Año"].values.reshape(-1, 1)  # Año como variable independiente
y = df["Facturación (millones de euros)"].values  # Facturación como variable dependiente

# Ajustar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X, y)

# Generar valores predichos
predicciones = modelo.predict(X)

# Crear la gráfica
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["Año"], y=df["Facturación (millones de euros)"], color="blue", label="Datos reales")
plt.plot(df["Año"], predicciones, color="red", label="Regresión lineal")

# Configuración del gráfico
plt.title("Regresión Lineal: Año vs. Facturación", fontsize=14)
plt.xlabel("Año", fontsize=12)
plt.ylabel("Facturación (millones de euros)", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Mostrar la gráfica
plt.show()

# Coeficientes del modelo
print(f"Pendiente (coeficiente): {modelo.coef_[0]:.2f}")
print(f"Intersección (término independiente): {modelo.intercept_:.2f}")
