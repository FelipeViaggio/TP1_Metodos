import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
import numpy as np

columns_names = ['x1', 'x2']

# Cargar los datos
mediciones = pd.read_csv('mnyo_mediciones.csv', sep=' ', names=columns_names)
groundtruth = pd.read_csv('mnyo_ground_truth.csv', sep=' ', names=columns_names)

# Ver las primeras filas de los datos
print(mediciones.head())
print(groundtruth.head())

# Verificar la cantidad de datos
print('Cantidad de mediciones: ', len(mediciones))
print('Cantidad de groundtruth: ', len(groundtruth))

print(mediciones.columns)

# Supongamos que tienes los tiempos en una lista llamada 'tiempos'
tiempos = list(range(len(mediciones)))

tiempos_finos = np.linspace(min(tiempos), max(tiempos), len(groundtruth))

polinomio_x1 = lagrange(tiempos, mediciones['x1'])
polinomio_x2 = lagrange(tiempos, mediciones['x2'])

posiciones_interpoladas_x1_finos = polinomio_x1(tiempos_finos)
posiciones_interpoladas_x2_finos = polinomio_x2(tiempos_finos)

# Visualizar la trayectoria
plt.figure(figsize=(10, 6))
plt.scatter(mediciones['x1'], mediciones['x2'], label='Mediciones', color = 'y')
plt.plot(posiciones_interpoladas_x1_finos, posiciones_interpoladas_x2_finos, label='Interpolacion', color = 'r')
plt.plot(groundtruth['x1'], groundtruth['x2'], label='Groundtruth')
plt.legend()
plt.show()