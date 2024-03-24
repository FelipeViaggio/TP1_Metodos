import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.interpolate import CubicSpline
import numpy as np

columns_names = ['x1', 'x2']

# Cargar los datos

mediciones = pd.read_csv('mnyo_mediciones.csv', sep=' ', names=columns_names)
mediciones2 = pd.read_csv('mnyo_mediciones2.csv', sep=' ', names=columns_names)
groundtruth = pd.read_csv('mnyo_ground_truth.csv', sep=' ', names=columns_names)

# Supongamos que tienes los tiempos en una lista llamada 'tiempos'
tiempos_1 = list(range(len(mediciones)))
tiempos_finos = np.linspace(min(tiempos_1), max(tiempos_1), len(groundtruth))

tiempos_2 = list(range(len(mediciones2)))
tiempos_finos2 = np.linspace(min(tiempos_2), max(tiempos_2), len(groundtruth))

# Interpolamos con Splines

splines_x1 = CubicSpline(tiempos_1, mediciones['x1'])
splines_x2 = CubicSpline(tiempos_1, mediciones['x2'])

splines_x1_2 = CubicSpline(tiempos_2, mediciones2['x1'])
splines_x2_2 = CubicSpline(tiempos_2, mediciones2['x2'])

# Evaluamos Splines en los tiempos finos

posiciones_interpoladas_x1_finos_splines = splines_x1(tiempos_finos)
posiciones_interpoladas_x2_finos_splines = splines_x2(tiempos_finos)

posiciones_interpoladas_x1_finos_splines2 = splines_x1_2(tiempos_finos2)
posiciones_interpoladas_x2_finos_splines2 = splines_x2_2(tiempos_finos2)

# Punto de partida Inicial

t1_inicial = tiempos_1[0]
t2_inicial = tiempos_2[0]

# Definimos las funciones que representan las trayectorias

def Tr1(t):
    return np.array([splines_x1(t), splines_x2(t)])

def Tr2(t):
    return np.array([splines_x1_2(t), splines_x2_2(t)])

# Definimos la funcion F

def F(t1, t2):
    return Tr1(t1) - Tr2(t2)

# Definimos la matriz Jacobiana

def J(t1, t2):
    return np.array([[splines_x1.derivative()(t1), -splines_x1_2.derivative()(t2)], [splines_x2.derivative()(t1), -splines_x2_2.derivative()(t2)]])

# Definimos una función para el método de la bisección
def biseccion(f, a, b, tol=1e-6):
    c = a
    while ((b-a) >= tol):
        c = (a+b)/2
        if (f(c) == 0.0):
            break
        if (f(a)*f(c) < 0):
            b = c
        else:
            a = c
    return c

# Definimos una función para el error de la intersección
def error_interseccion(t1, t2):
    return np.linalg.norm(Tr1(t1) - Tr2(t2))

# Usamos el método de la bisección para obtener un mejor punto de partida inicial
t1_inicial = biseccion(lambda t: error_interseccion(t, t2_inicial), min(tiempos_1), max(tiempos_1))
t2_inicial = biseccion(lambda t: error_interseccion(t1_inicial, t), min(tiempos_2), max(tiempos_2))

# Iteraciones del método de Newton
alpha = 0.01  # Parámetro de relajación

tolerancia = 1e-6

for i in range(100):
    t1, t2 = t1_inicial, t2_inicial
    F_t = F(t1, t2)
    J_t = J(t1, t2)

    # Actualizamos t1 y t2 usando el método de Newton con relajación
    delta_t = np.linalg.inv(J_t).dot(F_t)
    t1_inicial -= alpha * delta_t[0]
    t2_inicial -= alpha * delta_t[1]

    # Verificamos si t1 o t2 se salen del rango de los datos
    if t1_inicial < min(tiempos_1) or t1_inicial > max(tiempos_1) or t2_inicial < min(tiempos_2) or t2_inicial > max(tiempos_2):
        print("Los valores de t1 o t2 se salen del rango de los datos en la iteración", i)
        break

    # Verificamos si cumplimos el criterio de parada
    if np.linalg.norm(delta_t) < tolerancia:
        print("El método de Newton ha convergido en la iteración", i)
        break

else:
    print("El método de Newton no ha convergido después de 100 iteraciones")

print(f"El punto de intersección se encuentra en t1 = {t1_inicial} y t2 = {t2_inicial}")

# Graficamos lo que hicimos

plt.figure(figsize=(15, 10))
plt.scatter(mediciones['x1'], mediciones['x2'], label='Mediciones', color = 'y')
plt.plot(posiciones_interpoladas_x1_finos_splines, posiciones_interpoladas_x2_finos_splines, label='Interpolacion 1')
plt.plot(posiciones_interpoladas_x1_finos_splines2, posiciones_interpoladas_x2_finos_splines2, label='Interpolacion 2')

plt.scatter([Tr1(t1_inicial)[0]], [Tr1(t1_inicial)[1]], color='r', label='Punto de intersección')

plt.legend()
plt.title('Interpolaciones con Splines')
plt.show()
