import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Ejemplo de listas ajustadas con la misma longitud
datos = {
    "Week": [
        "2004-01", "2004-02", "2004-03", "2004-04", "2004-05", 
        "2004-06", "2004-07", "2004-08", "2004-09", "2004-10",
        "2004-11", "2004-12", "2005-01", "2005-02", "2005-03", 
        "2005-04", "2005-05", "2005-06", "2005-07", "2005-08",
        "2005-09", "2005-10", "2005-11", "2005-12", "2006-01", 
        "2006-02", "2006-03", "2006-04", "2006-05", "2006-06",
        "2006-07", "2006-08", "2006-09", "2006-10", "2006-11", 
        "2006-12", "2007-01", "2007-02", "2007-03", "2007-04",
        "2007-05", "2007-06", "2007-07", "2007-08", "2007-09", 
        "2007-10", "2007-11", "2007-12", "2008-01", "2008-02",
        
    ],
    "Python": [
        20, 19, 21, 21, 20, 
        22, 20, 21, 20, 21, 
        18, 19, 20, 19, 21, 
        19, 20, 21, 20, 21, 
        18, 25, 18, 18, 18, 
        19, 19, 19,19, 19, 
        18, 17, 19, 18, 17, 
        19, 19, 19, 19, 18, 
        18, 19, 18, 18, 19, 
        18, 17, 17, 19, 19
    ],
    "JavaScript": [
        96, 100, 99, 99, 92, 
        100, 98, 94, 88, 86, 
        81, 80, 75, 82, 81, 
        77, 78, 83, 79, 78, 
        74, 75, 71, 67, 67, 
        69, 74, 70,69, 69, 
        66, 68, 62, 63, 60, 
        60, 58, 57, 61, 61, 
        59, 57, 58, 54, 49,
        52, 54, 56, 57, 53
    ]
}

# Crear el DataFrame
df = pd.DataFrame(datos)


# Convertir las fechas a formato datetime
weeks = pd.to_datetime(df["Week"])

# Convertir la semana a índice numérico (días desde la primera semana)
df["Week_Num"] = (weeks - weeks.min()).dt.days

# Datos para regresión
X = df[["Week_Num"]]  # Semana en formato numérico
y = df["JavaScript"]  # Analizando el uso de JavaScript, puedes cambiarlo a "JavaScript" si lo prefieres

# Crear y ajustar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X, y)

# Predicción para la línea de regresión
X_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = modelo.predict(X_pred)

# Visualización del gráfico
plt.figure(figsize=(12, 6))
plt.plot(weeks, y, "o", label="Datos reales (JavaScript)", color="blue")
plt.plot(pd.to_datetime(weeks.min()) + pd.to_timedelta(X_pred.flatten(), unit='d'), y_pred, "-", color="red", label="Regresión lineal")

# Configuración del gráfico
plt.title("Regresión lineal para JavaScript a lo largo del tiempo", fontsize=14)
plt.xlabel("Fecha", fontsize=12)
plt.ylabel("Popularidad de JavaScript (%)", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)  # Rotar las etiquetas del eje X para que se vean mejor
plt.show()
