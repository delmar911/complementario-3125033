import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Datos en formato estructurado
datos = {
    "Week": [
        "4/21/2019", "4/28/2019", "5/5/2019", "5/12/2019", "5/19/2019",
        "5/26/2019", "6/2/2019", "6/9/2019", "6/16/2019", "6/23/2019",
        "6/30/2019", "7/7/2019", "7/14/2019", "7/21/2019", "7/28/2019",
        "8/4/2019", "8/11/2019", "8/18/2019", "8/25/2019", "9/1/2019",
        "9/8/2019", "9/15/2019", "9/22/2019", "9/29/2019", "10/6/2019",
        "10/13/2019", "10/20/2019", "10/27/2019", "11/3/2019", "11/10/2019",
        "11/17/2019", "11/24/2019", "12/1/2019", "12/8/2019", "12/15/2019",
        "12/22/2019", "12/29/2019", "1/5/2020", "1/12/2020", "1/19/2020",
        "1/26/2020", "2/2/2020", "2/9/2020", "2/16/2020", "2/23/2020",
        "3/1/2020", "3/8/2020", "3/15/2020", "3/22/2020", "3/29/2020",
        "4/5/2020"
    ],
    "Python": [
        55, 52, 56, 56, 57, 55, 54, 58, 58, 60,
        57, 60, 60, 56, 59, 57, 54, 61, 59, 62,
        64, 67, 65, 67, 64, 68, 63, 61, 64, 66,
        69, 65, 63, 60, 60, 42, 41, 57, 59, 60,
        63, 62, 68, 71, 68, 67, 62, 59, 60, 64,
        65
    ],
    "Java": [
        55, 50, 56, 61, 56, 57, 58, 55, 56, 57,
        54, 58, 57, 55, 53, 54, 51, 58, 60, 57,
        58, 64, 61, 55, 58, 60, 58, 54, 58, 57,
        60, 55, 59, 53, 51, 39, 37, 50, 53, 53,
        56, 55, 55, 59, 57, 58, 52, 48, 47, 49,
        48
    ],
    "C++": [
        18, 16, 17, 18, 17, 17, 15, 16, 16, 16,
        16, 17, 17, 15, 16, 16, 16, 19, 19, 19,
        21, 22, 22, 20, 21, 22, 21, 21, 21, 22,
        22, 20, 20, 19, 18, 13, 12, 17, 18, 18,
        18, 17, 20, 20, 21, 20, 19, 17, 18, 18,
        18
    ]
}

df = pd.DataFrame(datos)

# Convertir la semana a índice numérico
weeks = pd.to_datetime(df["Week"])
df["Week_Num"] = (weeks - weeks.min()).dt.days

# Datos para regresión
X = df[["Week_Num"]]
y = df["Python"]  # Ejemplo: análisis de Python

# Crear y ajustar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X, y)

# Predicción para la línea de regresión
X_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = modelo.predict(X_pred)

# Visualización del gráfico
plt.figure(figsize=(12, 6))
plt.plot(weeks, y, "o", label="Datos reales (Python)", color="blue")
plt.plot(pd.to_datetime(weeks.min()) + pd.to_timedelta(X_pred.flatten(), unit='d'), y_pred, "-", color="red", label="Regresión lineal")

# Configuración del gráfico
plt.title("Regresión lineal para Python a lo largo del tiempo", fontsize=14)
plt.xlabel("Fecha", fontsize=12)
plt.ylabel("Popularidad (Python)", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
