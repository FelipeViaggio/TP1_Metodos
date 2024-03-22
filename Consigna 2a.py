import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.interpolate import CubicSpline
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

# Interpolamos con lagrange
polinomio_lagrange_x1 = lagrange(tiempos, mediciones['x1'])
polinomio_lagrange_x2 = lagrange(tiempos, mediciones['x2'])

# Evaluamos el polinomio en los tiempos finos
posiciones_interpoladas_x1_finos = polinomio_lagrange_x1(tiempos_finos)
posiciones_interpoladas_x2_finos = polinomio_lagrange_x2(tiempos_finos)

# Interpolamos con Splines

splines_x1 = CubicSpline(tiempos, mediciones['x1'])
splines_x2 = CubicSpline(tiempos, mediciones['x2'])

# Evaluamos Splines en los tiempos finos
posiciones_interpoladas_x1_finos_splines = splines_x1(tiempos_finos)
posiciones_interpoladas_x2_finos_splines = splines_x2(tiempos_finos)

'''
# Visualizar la trayectoria
plt.figure(figsize=(10, 6))
plt.scatter(mediciones['x1'], mediciones['x2'], label='Mediciones', color = 'y')
plt.plot(posiciones_interpoladas_x1_finos, posiciones_interpoladas_x2_finos, label='Interpolacion', color = 'r')
plt.plot(posiciones_interpoladas_x1_finos_splines, posiciones_interpoladas_x2_finos_splines, label='Interpolacion Splines', color = 'g')
plt.plot(groundtruth['x1'], groundtruth['x2'], label='Groundtruth')
plt.legend()
plt.show() '''

# Calculamos errores, usamos error Absoluto en cada punto

errores_absolutos_lagrange = np.abs(polinomio_lagrange_x1(tiempos_finos) - groundtruth['x1']) + np.abs(polinomio_lagrange_x2(tiempos_finos) - groundtruth['x2'])
errores_absolutos_splines = np.abs(splines_x1(tiempos_finos) - groundtruth['x1']) + np.abs(splines_x2(tiempos_finos) - groundtruth['x2'])


#

fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Esquina superior izquierda, Ground Truth.

axs[0, 0].plot(groundtruth['x1'], groundtruth['x2'], label='Groundtruth')
axs[0, 0].scatter(mediciones['x1'], mediciones['x2'], label='Mediciones', color = 'y')
axs[0,0].legend()
axs[0,0].set_title('Ground Truth')

# Esquina superior derecha, Interpolacion Lagrange vs Ground Truth.

axs[0, 1].plot(groundtruth['x1'], groundtruth['x2'], label='Groundtruth')
axs[0, 1].plot(posiciones_interpoladas_x1_finos, posiciones_interpoladas_x2_finos, label='Interpolacion', color = 'r')
axs[0, 1].scatter(mediciones['x1'], mediciones['x2'], label='Mediciones', color = 'y')
axs[0,1].legend()
axs[0,1].set_title('Interpolacion Lagrange vs Ground Truth')

# Esquina inferior izquierda, Interpolacion Splines vs Ground Truth.

axs[1, 0].plot(groundtruth['x1'], groundtruth['x2'], label='Groundtruth')
axs[1, 0].plot(posiciones_interpoladas_x1_finos_splines, posiciones_interpoladas_x2_finos_splines, label='Interpolacion Splines', color = 'g')
axs[1, 0].scatter(mediciones['x1'], mediciones['x2'], label='Mediciones', color = 'y')
axs[1,0].legend()
axs[1,0].set_title('Interpolacion Splines vs Ground Truth')

# Esquina inferior derecha, Interpolacion Lagrange vs Interpolacion Splines vs Ground Truth.

axs[1, 1].plot(groundtruth['x1'], groundtruth['x2'], label='Groundtruth')
axs[1, 1].plot(posiciones_interpoladas_x1_finos, posiciones_interpoladas_x2_finos, label='Interpolacion', color = 'r')
axs[1, 1].plot(posiciones_interpoladas_x1_finos_splines, posiciones_interpoladas_x2_finos_splines, label='Interpolacion Splines', color = 'g')
axs[1, 1].scatter(mediciones['x1'], mediciones['x2'], label='Mediciones', color = 'y')
axs[1,1].legend()
axs[1,1].set_title('Interpolacion Lagrange vs Interpolacion Splines vs Ground Truth')

plt.tight_layout()
plt.show()

# Grafico de errores

plt.figure(figsize=(10, 6))
plt.plot(tiempos_finos, errores_absolutos_lagrange, label='Errores Lagrange', color = 'r')
plt.plot(tiempos_finos, errores_absolutos_splines, label='Errores Splines', color = 'g')
plt.legend()
plt.title('Errores Absolutos')
plt.show()

# Calculamos el error cuadratico medio