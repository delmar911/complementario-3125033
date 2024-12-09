import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Datos en formato estructurado
datos = {
    "Event ID": [
        "2019969e-ecfa-41c4-b681-9b684bc3b3bf", "1668e954-781f-4731-94dc-24218b983ba1",
        "0ef24a20-1d25-41fa-81b8-e19fb63e9e4c", "073b6225-0998-488c-aa1c-23e49814b6ff",
        "783fd153-6b88-44c1-8db5-d882300088cc"
    ],
    "Timestamp": [
        "2020-02-07 23:46:57", "2021-05-25 19:03:44", "2022-01-04 09:08:07",
        "2022-10-12 19:48:43", "2021-11-24 02:04:33"
    ],
    "Attack Severity": ["Critical", "Critical", "High", "Critical", "Medium"]
}

df = pd.DataFrame(datos)

# Convertir la fecha a formato datetime y extraer el año y mes
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['YearMonth'] = df['Timestamp'].dt.to_period('M')
df['NumericMonth'] = (df['Timestamp'].dt.year - df['Timestamp'].dt.year.min()) * 12 + df['Timestamp'].dt.month

# Mapear severidades a valores numéricos
severity_map = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0}
df['SeverityNumeric'] = df['Attack Severity'].map(severity_map)

# Modelo de regresión lineal
X = df[['NumericMonth']]
y = df['SeverityNumeric']
model = LinearRegression()
model.fit(X, y)

# Predicción
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = model.predict(x_range)

# Visualización del gráfico
plt.figure(figsize=(10, 6))
sns.scatterplot(x='NumericMonth', y='SeverityNumeric', data=df, label='Datos reales')
plt.plot(x_range, y_pred, color='red', label='Regresión lineal')

# Configuración del gráfico
plt.title("Regresión lineal: Severidad de ataques a lo largo del tiempo", fontsize=14)
plt.xlabel("Meses desde el inicio del período", fontsize=12)
plt.ylabel("Severidad (numérica)", fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
