# %%
import numpy as np 
import matplotlib.pyplot as plt 
import time 
from scipy.stats import uniform

def uniforme(n,seed = None):

    modulo = 2**31-1

    if seed is None:
        x_0 = (time.time()) % modulo # Semillas diferentes
    else:
        x_0 = float(seed)

    iteracion = n    # Número de iteraciones

    # Generador de recurrencia
    x = np.array([x_0,20536,3028,6486,5546]) # Vector inicial
    m = len(x)
    X_t = np.zeros(iteracion)

    for i in range(iteracion):

        x_t = ( 104420*x[0] + 107374182*x[4]) % modulo
        X_t[i] = x_t
        
        for j in range(1,m):
            x[j-1] = x[j]

        x[m-1] = x_t
        
    return X_t / (modulo -1)
    
# Ejercicio 2
n = 100000
U = uniforme(n)
plt.figure(figsize=(7,5))
plt.hist(U,density = True)
plt.xlabel('x')
plt.title('Histograma de los valores simulados')
plt.show()

x = uniform.rvs(0,1,n)
plt.figure(figsize=(7,5))
plt.hist(x,density=True)
plt.xlabel('x')
plt.title('Histograma de los valores simulados por scipy')
plt.show()









