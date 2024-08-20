import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Generate a synthetic dataset
def generate_synthetic_data(num_days=365*10):
    dates = pd.date_range(start='2014-01-01', periods=num_days, freq='D')
    temperatures = np.random.normal(loc=30, scale=10, size=num_days)  # Normal distribution of temperatures
    humidity = np.random.normal(loc=50, scale=10, size=num_days)
    wind_speed = np.random.normal(loc=10, scale=5, size=num_days)
    is_heatwave = [1 if temp > 35 and random.random() > 0.5 else 0 for temp in temperatures]

    data = {
        'Date': dates,
        'Temperature': temperatures,
        'Humidity': humidity,
        'WindSpeed': wind_speed,
        'Heatwave': is_heatwave
    }

    return pd.DataFrame(data)

df = generate_synthetic_data()
df.to_csv('heatwave_data.csv', index=False)
print("Synthetic dataset generated.")

df = pd.read_csv('heatwave_data.csv')

# Display basic statistics
print(df.describe())

# Plot temperature distribution
sns.histplot(df['Temperature'], bins=50, kde=True)
plt.title('Temperature Distribution')
plt.show()

# Correlation heatmap
corr = df.corr()
sns.heatmap(corr, annot=True)
plt.title('Correlation Matrix')
plt.show()




