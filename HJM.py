# Importación de bibliotecas necesarias
from mpl_toolkits.mplot3d import Axes3D
import copy as copylib
from progressbar import *
import pandas as pd
import numpy as np
from IPython.display import display
from matplotlib import pylab
from pylab import *
pylab.rcParams['figure.figsize'] = (16, 4.5)
np.random.seed(0)

# Tenores: tiempo que resta entre la fecha de valor de un activo financiero y su
# fecha de vencimiento, expresable en dias, meses o años.

# Cargar y procesar datos
# Leemos un archivo CSV que contiene tasas históricas y las convertimos a porcentajes
dataframe = pd.read_csv('hjm_data.csv').set_index('time') / 100
pd.options.display.max_rows = 10
display(dataframe)

# Variables iniciales
hist_timeline = list(dataframe.index)  # Línea de tiempo histórica
tenors = [eval(x) for x in dataframe.columns]  # Tenores del archivo
hist_rates = matrix(dataframe)  # Tasas históricas en formato matricial

# Graficar evolución histórica de las tasas
plot(hist_rates)
xlabel(r'Tiempo $t$')
ylabel(r'Tasa histórica $f(t,\tau)$')
text(200, 0.065, r'Evolución de la curva de rendimiento histórica con 51 tenores durante 5 años. Cada línea representa un tenor diferente.')
title(r'Tasas históricas $f(t,\tau)$ por $t$')
show()

# Muestra la forma en la que evolucionan las tasas historicas para distintos tenores
# a lo largo del tiempo

# Graficar las tasas por tenores
plot(tenors, hist_rates.transpose())
xlabel(r'Tenor $\tau$')
ylabel(r'Tasa histórica $f(t,\tau)$')
text(3, 0.066, r'Evolución de la curva de rendimiento histórica. Cada línea representa un día en el pasado.')
title(r'Tasas históricas $f(t,\tau)$ por $\tau$')
show()

# Representa la estructura de las tasas historicas observadas para diferentes dias
# en el pasado

# Calcular diferencias de tasas por tiempo
diff_rates = diff(hist_rates, axis=0)
assert hist_rates.shape[1] == diff_rates.shape[1]

plot(diff_rates)
xlabel(r'Tiempo $t$')
title(r'Matriz de tasas diferenciada $df(t,\tau)$ por $t$')
show()

# Cambios en las tasas historicas con respecto al tiempo, estas calculadas como
# derivadas finitas

# Calcular matriz de covarianza
sigma = cov(diff_rates.transpose())
print(f"Dimensiones de Sigma: {sigma.shape}")

# Escalar matriz de covarianza anual
sigma *= 252
eigval, eigvec = linalg.eig(sigma)  # Eigenvalores y eigenvectores
eigvec = matrix(eigvec)

# Verificar tipos
assert type(eigval) == ndarray
assert type(eigvec) == matrix

print("Eigenvalores:")
print(eigval)

# Selección de componentes principales
factors = 3  # Número de factores principales
index_eigvec = list(reversed(eigval.argsort()))[:factors]  # Componentes principales más altos
princ_eigval = array([eigval[i] for i in index_eigvec])  # Eigenvalores principales
princ_comp = hstack([eigvec[:, i] for i in index_eigvec])  # Eigenvectores principales

print("Eigenvalores principales:")
print(princ_eigval)

print("\nEigenvectores principales:")
print(princ_comp)

# Graficar eigenvectores principales
plot(princ_comp, marker='.')
title('Eigenvectores de los componentes principales')
xlabel(r'Tiempo $t$')
show()

# Representa los valores propios asociados con los principales componentes de la
# matriz de covarianza, describen las direcciones principales de variabilidad en
# los datos

# Calcular volatilidades discretizadas
sqrt_eigval = matrix(princ_eigval ** 0.5)
tmp_m = vstack([sqrt_eigval for i in range(princ_comp.shape[0])])
vols = multiply(tmp_m, princ_comp)

print(f'Dimensiones de Vols: {vols.shape}')
plot(vols, marker='.')
title('Volatilidades discretizadas')
xlabel(r'Tiempo $t$')
ylabel(r'Volatilidad $\sigma$')
show()

# Muestra las volatilidades discretizadas, estas se calculan a partir de los valores
# propios y vectores propios principales

# Interpolador polinómico
class PolynomialInterpolator:
    def __init__(self, params):
        assert type(params) == np.ndarray
        self.params = params
    
    def calc(self, x):
        n = len(self.params)
        C = self.params
        X = np.array([x**i for i in reversed(range(n))])
        return sum(np.multiply(X, C))

# Ajustar volatilidades mediante interpoladores polinómicos
fitted_vols = []

def get_matrix_column(mat, i):
    return array(mat[:,i].flatten())[0]

def fit_volatility(i, degree, title):
    vol = get_matrix_column(vols, i)
    fitted_vol = PolynomialInterpolator(polyfit(tenors, vol, degree))
    plot(tenors, vol, marker='.', label='Volatilidad discretizada')
    plot(tenors, [fitted_vol.calc(x) for x in tenors], label='Volatilidad ajustada')
    plt.title(title)
    xlabel(r'Tiempo $t$')
    legend()
    fitted_vols.append(fitted_vol)

# Ajustar las tres principales componentes
subplot(1, 3, 1), fit_volatility(0, 0, 'Primera componente')
subplot(1, 3, 2), fit_volatility(1, 3, 'Segunda componente')
subplot(1, 3, 3), fit_volatility(2, 3, 'Tercera componente')
show()

# Ajuste de las volatilidades discretizadas utilizando interpolación polinómica
# para las tres componentes principales, cada grafico corresponde a una de las
# tres componentes principales, mostrando como se ajusta la volatilidad en funcion
# del tenor
