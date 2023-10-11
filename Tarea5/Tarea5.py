import numpy as np 
import matplotlib.pyplot as plt 
import time 
from scipy.stats import uniform
import scipy.integrate as integrate

# Ejercicio 2
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


# Ejercicio 5     Adaptive Rejection Sampling

def recta(x,x_i,x_j):
    '''
    Función que dado dos puntos de la rejilla, calcula sus correspondientes valores en la función log-densidad para con ellos crear una recta entre este par de puntos. Se ingresa tambien un x arbitrario para calcular el valor de la recta para ese dominio.
    '''

    y_i = np.log(f(x_i))
    y_j = np.log(f(x_j))

    y = y_i + (y_i-y_j)/(x_i - x_j)*(x- x_i)
    return y

def quantil_g(x):

    x_techo = np.linspace(0.05,9,400) 

    indice = 0
    indexNotFound = True
    while indexNotFound:
        if (acumulado[indice] < x  and indice < len(x_techo) -1):
            indice = indice + 1

        else: 
            indexNotFound = False

    return x_techo[indice]

def valor_x(z):
    '''
    Busca el indice en el linspace para que x_techo(indice) es el número mayor z más pequeño, que es la entrada de la función.
    
    '''
    x_techo = np.linspace(0.05,9,400) 

    indice = 0
    indexNotFound = True

    while indexNotFound:
        if (x_techo[indice] < z  and indice < len(x_techo) -1):
            indice = indice + 1

        else: 
            indexNotFound = False

    return indice

def constructor(X):
    '''
    Construye la función de distribución acumulada para la función de densidad techo.
    '''

    X.sort()
    Y = []
    n = len(X)
    for i in range(n):
        Y.append(np.log(f(X[i])))

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

    #Simular de la densidad techo 
    omega = integrate.quad(ftecho,0,np.inf)[0]  # Constante de normalidad
    
    def g(x):
        return ftecho(x)/omega

    # Vectoriazación de g
    x_techo = np.linspace(0.05,9,400) # Dominio para la función techo
    densidad_g = np.zeros(len(x_techo))
    for j in range(len(x_techo)):
        densidad_g[j] = g(x_techo[j])

    # Vectorización (ya que la función no puede recibir vectores)
    envolvente = np.zeros(len(x_techo))
    for i in range(len(x_techo)):
        envolvente[i] = techo(x_techo[i])

    # Función acumulada de probabilidad
    def cdf_g(x):
        return integrate.quad(g,0,x)[0]

    acumulado = np.zeros(len(x_techo))
    for j in range(len(x_techo)):
        acumulado[j] = cdf_g(x_techo[j])

    ### Gráficas de interés
    # Descomentar para ver. Con cuidado de no iterar graficas al actualizar la rejilla, por eso mejor dejar comentadas
    
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

    # # Graficar la función de densidad y su acotamiento
    # plt.plot(x,f(x), label = 'Gamma(2,1)')
    # plt.plot(np.array(X),f(np.array(X)),'o')
    # plt.plot(x_techo, np.exp(envolvente), label = 'exp h(x)')
    # plt.title('Densidad y su envolvente')
    # plt.xlabel('x')
    # plt.legend()
    # plt.show()

    #Plot de cdf_g la función acumulada
    # plt.plot(x_techo,acumulado)  
    # plt.show()

    return acumulado, densidad_g


# INICIALIZACION  /////////////
x = np.linspace(0.01,9,100)
f = lambda x: x*np.exp(-x)

# Rejilla 
X = [0.4,1,3,8,6,0.25,2.2,1.2] #X = [0.4,1,3,8] malla inicial
X.sort()

acumulado, densidad_g = constructor(X)

tamaño = 100000    #Modifica tamaño de muestra objetivo
Muestra = np.zeros(tamaño)


for k in range(tamaño):

    # Transformada inversa
    u = uniform.rvs(0,1)
    z = quantil_g(u)

    # Actualizar la rejilla con más puntos según accept-reject
    # Las actualizaciones a la rejilla se hacen manualmente, ya que al actualizar un número arbitrario de veces el proceso tarda mucho. Para hacer las actualizaciones modiquese la rejilla inicial según la condición de aceptación. 
    actualizador = 1
    if actualizador <=0:   #Cantidad de actualizaciones
        actualizador += 1
        if u*f(z) > densidad_g[valor_x(z)]:  #Condición de rechazo
            X.append(z)
            actualizador = actualizador + 1
            acumulado = constructor(X)[0]
    else:
        
        Muestra[k] = z


# print(Muestra)
plt.hist(Muestra,density=True,bins= 100) #int(np.sqrt(len(Muestra))))   
plt.title('Simulación ARS para gamma(2,1)')
plt.plot(x,f(x),label = 'Gamma(2,1)')
plt.xlabel('x')
plt.legend()
plt.show()

