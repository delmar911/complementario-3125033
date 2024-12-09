import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Datos en formato estructurado
datos = {
    "Date": [
        "July 2004", "July 2005", "July 2006", "July 2007"
    ],
    "Java": [
        30.159999999999997,30.29, 30.349999999999998, 30.570000000000004
    
    ],
    "Cobol": [
        0.42, 0.38999999999999996, 0.45999999999999996, 0.32
        
    ],
    "Python": [
        2.5100000000000002,2.97, 3.5700000000000003, 4.34
    ]
}

df = pd.DataFrame(datos)

# Convertir la semana a índice numérico
dates = pd.to_datetime(df["Date"])
df["date_Num"] = (dates - dates.min()).dt.days

# Datos para regresión
X = df[["date_Num"]]
y = df["Java"]  # Ejemplo: análisis de Java

# Crear y ajustar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X, y)

# Predicción para la línea de regresión
X_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = modelo.predict(X_pred)

# Visualización del gráfico
plt.figure(figsize=(12, 6))
plt.plot(dates, y, "o", label="Datos reales (Java)", color="blue")
plt.plot(pd.to_datetime(dates.min()) + pd.to_timedelta(X_pred.flatten(), unit='d'), y_pred, "-", color="red", label="Regresión lineal")

# Configuración del gráfico
plt.title("Regresión lineal popularidad de lenguajes a través tiempo", fontsize=14)
plt.xlabel("Fecha", fontsize=12)
plt.ylabel("Popularidad (Java)", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
