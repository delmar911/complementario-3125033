import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Carga de datos en un DataFrame
data = pd.DataFrame({
    "Año": ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023"],
    "Nº de empleados en el sector": [2360, 3376, 4460, 5440, 6337, 6900, 7320, 7686, 7993, 8233, 8480]
})

# Convertir el año a formato numérico
data["Año"] = pd.to_numeric(data["Año"])

# Crear el modelo de regresión lineal
model = LinearRegression()
X = data["Año"].values.reshape(-1, 1)
y = data["Nº de empleados en el sector"].values
model.fit(X, y)

# Generar la gráfica
plt.figure(figsize=(10, 6))
plt.scatter(data["Año"], data["Nº de empleados en el sector"], label="Datos reales")
plt.plot(data["Año"], model.predict(X), color="red", label="Regresión lineal")
plt.xlabel("Año")
plt.ylabel("Nº de empleados en el sector")
plt.title("Evolución del número de empleados en el sector de videojuegos en España")
plt.legend()
plt.grid()
plt.show()