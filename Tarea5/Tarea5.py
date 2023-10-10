# %%
import numpy as np 
import matplotlib.pyplot as plt 
import time 
from scipy.stats import uniform
import scipy.integrate as integrate

# %%
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
n = 100
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

# %%
# Accept-Reject Method
from scipy.stats import cauchy
'''
Simulamos la distribución normal a traves de simulaciones cauchy.
'''

# //////////////////
'''
M = 2/np.sqrt(np.exp(1))   #Cota M

f = lambda x: np.exp(-x**2/2)
g = lambda x: M/(1+ x**2)

x = np.linspace(-5,5,100)

plt.plot(x,f(x))
plt.plot(x,g(x))
plt.show()


# Método aceptar y rechazar
size = 10
Y = np.zeros(size+1)

count = 0
while count <= size:

    X = cauchy.rvs(0,1)
    u = uniform.rvs(0,1)

    if u <= f(X)/g(X):
        Y[count] = X
        count = count + 1

# plt.hist(Y,density= True)
'''

# %%
# Ejercicio 5     Adaptive Rejection Sampling
x = np.linspace(0.01,9,100)
f = lambda x: x*np.exp(-x)

# Rejilla inicial
X = [0.4,1,3,8]
X.sort()

Y = []
n = len(X)
for i in range(n):
    Y.append(np.log(f(X[i])))


def recta(x,x_i,x_j):
    '''
    Función que dado dos puntos de la rejilla, calcula sus correspondientes valores en la función log-densidad para con ellos crear una recta entre este par de puntos. Se ingresa tambien un x arbitrario para calcular el valor de la recta para ese dominio.
    '''

    y_i = np.log(f(x_i))
    y_j = np.log(f(x_j))

    y = y_i + (y_i-y_j)/(x_i - x_j)*(x- x_i)
    return y

def techo(x):
    '''
    Esta función recive un x, devuelve el valor de la envolvente del método ARS en dicho valor de x.
    '''
    if x <= 0:
        g = 0
    elif (x >= 0 and x < X[0]):
        g = recta(x,X[0],X[1])
    elif (x >= X[0] and x < X[1]):
        g = recta(x,X[1],X[2])
    elif (x >= X[n-2] and x < X[n-1]):
        g = recta(x,X[n-3],X[n-2])
    elif (x >= X[n-1]):
        g = recta(x,X[n-2],X[n-1])
    else:
        for i in range(1,n-2):
            if ( x >= X[i] and x < X[i+1]):
                g = min(recta(x,X[i-1],X[i]),recta(x,X[i+1],X[i+2]))
    return g

def ftecho(x):
    return np.exp(techo(x))


# Vectorización (ya que la función no puede recibir vectores)
x_techo = np.linspace(0.1,9,400) # Dominio para la función techo
envolvente = np.zeros(len(x_techo))
for i in range(len(x_techo)):
    envolvente[i] = techo(x_techo[i])


### Gráficas de interes
# Grafica densidad
plt.plot(x,f(x),label = 'Gamma(2,1)')
plt.title('Densidad de la distribución Gamma(2,1)')
plt.xlabel('x')
plt.legend()
plt.show()

# Grafica log-densidad
plt.plot(x,np.log(f(x)), label = 'log f(x)')
plt.plot(X,Y,"o")
plt.title('Log densidad y su envolvente')
plt.ylabel('log f(x)')
plt.xlabel('x')

# Grafica de la envolvente
plt.plot(x_techo,envolvente, label = 'h(x)')
plt.legend()
plt.show()

# Graficar la función de densidad y su acotamiento
plt.plot(x,f(x), label = 'Gamma(2,1)')
plt.plot(np.array(X),f(np.array(X)),'o')
plt.plot(x_techo, np.exp(envolvente), label = 'exp h(x)')
plt.title('Densidad y su envolvente')
plt.xlabel('x')
plt.legend()
plt.show()

# %%

omega = integrate.quad(ftecho,0,np.inf)[0]  # Constante de normalidad

def g(x):
    return ftecho(x)/omega

def cdf_g(x):
    return integrate.quad(g,0,x)[0]

acumulado = np.zeros(len(x_techo))
for j in range(len(x_techo)):
    acumulado[j] = cdf_g(x_techo[j])


#Plot de cdf_g la función acumulada
plt.plot(x_techo,acumulado)  
plt.show()

# %%
def quantil_g(x):
    t = 0 
    while cdf_g(t) < x:    
        t = t + 0.1
    return t

m= 10
U = uniform.rvs(0,1,m)
Y_muestra = np.zeros(m)
for i in range(m):
    Y_muestra[i] = quantil_g(U[i])



# # Gráfica de la simulación de la nueva variable
# plt.hist(Y_muestra,density=True)
# plt.plot(x,f(x))





# print("quantil",quantil_g(0.25))
# print(cdf_g(quantil_g(0.456)))
    













