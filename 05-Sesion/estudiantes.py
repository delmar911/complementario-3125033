import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Crear un DataFrame con los datos proporcionados
data = {
    'Name': ['John Smith', 'Alice Johnson', 'Robert Davis', 'Emily Wilson', 'Michael Brown',
             'Laura Lee', 'William Johnson', 'Sarah Miller', 'James Wilson', 'Olivia Clark', 
             'Andrew Hall', 'David Jones', 'Elizabeth Williams', 'Charles Miller', 'Susan Davis', 
             'John Brown', 'Laura Lee', 'William Johnson', 'Sarah Miller', 'James Wilson', 
             'Olivia Clark',  'Andrew Hall', 'David Jones', 'Elizabeth Williams', 'Charles Miller', 
             'Susan Davis', 'John Brown', 'Laura Lee', 'William Johnson', 'Sarah Miller', 
             'James Wilson', 'Olivia Clark', 'Andrew Hall', 'David Jones', 'Elizabeth Williams', 
             'Charles Miller', 'Susan Davis', 'John Brown', 'Laura Lee', 'William Johnson', 
             'Sarah Miller', 'James Wilson', 'Olivia Clark', 'Andrew Hall', 'David Jones', 
             'Elizabeth Williams', 'Charles Miller', 'Susan Davis', 'John Brown', 
             'Laura Lee'],
    'GPA': [3.5, 3.2, 3.8, 3.7, 3.4, 
            3.9, 3.6, 3.7, 3.3, 3.5, 
            3.8, 3.7, 3.3, 3.5, 3.8, 
            3.6, 3.9, 3.6, 3.7, 3.3, 
            3.5, 3.8, 3.7, 3.5, 3.8, 
            3.8, 3.6, 3.7, 3.9, 3.8, 
            3.7, 3.5, 3.7, 3.6, 3.5, 
            3.9, 3.7, 3.8, 3.7, 3.8, 
            3.6, 3.3, 3.6, 3.3, 3.5,
            3.7, 3.7, 3.3, 2.0, 4.0],
    'Python': ['Strong', 'Average', 'Strong', 'Weak', 'Average', 
               'Strong', 'Average', 'Weak', 'Average', 'Weak', 
               'Strong', 'Strong', 'Average', 'Weak', 'Strong', 
               'Average', 'Strong', 'Average', 'Strong', 'Weak', 
               'Strong', 'Strong', 'Strong', 'Weak', 'Strong', 
               'Average', 'Weak', 'Strong', 'Average', 'Weak', 
               'Strong', 'Strong', 'Strong', 'Average', 'Strong',
               'Strong', 'Average', 'Weak', 'Strong', 'Strong', 
               'Weak', 'Strong', 'Average', 'Weak', 'Average', 
               'Strong', 'Average', 'Strong', 'Average', 'Strong']
}

df = pd.DataFrame(data)

# Convertir la columna 'Python' a valores numéricos
python_mapping = {'Weak': 1, 'Average': 2, 'Strong': 3}
df['Python'] = df['Python'].map(python_mapping)

# Realizar el análisis de regresión lineal
X = df[['Python']]  # Variable independiente
y = df['GPA']  # Variable dependiente

# Crear y ajustar el modelo de regresión lineal
model = LinearRegression()
model.fit(X, y)

# Predecir valores para la línea de regresión
y_pred = model.predict(X)

# Crear la gráfica
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Python', y='GPA', data=df, color='blue', label='Datos')
plt.plot(df['Python'], y_pred, color='red', label='Regresión Lineal')

# Etiquetas y título
plt.title('Relación entre GPA y Conocimiento de Python')
plt.xlabel('Nivel de Conocimiento de Python')
plt.ylabel('GPA')
plt.legend()

# Mostrar la gráfica
plt.show()

# Imprimir los coeficientes de la regresión
print(f"Coeficiente de la regresión: {model.coef_[0]}")
print(f"Intercepto de la regresión: {model.intercept_}")
