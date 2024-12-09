import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read the CSV data
df = pd.read_csv('./datos_csv/top10ide.csv', skiprows=0, encoding='utf-8')

# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%B %Y')

# Select some prominent IDEs for regression
ide_columns = ['Eclipse', 'Visual Studio', 'Visual Studio Code', 'NetBeans']

# Prepare the plot
plt.figure(figsize=(12, 8))

# Perform linear regression for each selected IDE
for ide in ide_columns:
    # Create a copy of the data and drop NaN values
    subset = df[['Date', ide]].dropna()
    
    # Prepare X (time) and y (market share)
    X = np.array(range(len(subset))).reshape(-1, 1)
    y = subset[ide].values

    # Perform linear regression
    reg = LinearRegression().fit(X, y)
    
    # Plot original data points
    plt.scatter(X, y, label=f'{ide} - Actual')
    
    # Plot regression line
    plt.plot(X, reg.predict(X), linestyle='--', 
             label=f'{ide} - Trend (R² = {reg.score(X, y):.4f})')

plt.title('IDE Market Share Linear Regression Trend (2004-2008)', fontsize=15)
plt.xlabel('Months since July 2004', fontsize=12)
plt.ylabel('Market Share', fontsize=12)
plt.legend(title='IDE Performance', loc='best')
plt.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()