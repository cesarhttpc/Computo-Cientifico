
import numpy as np
import time
from funciones2 import backward, GramSchmidtModified
from scipy.stats import norm
import matplotlib.pyplot as plt


# Ejercicio 1
A = np.array([[1,2,2],[2,0,2],[0,1,2]])   

print("De la matriz A: \n", A)
Q, R = GramSchmidtModified(A)

print("La factorización QR por Gram-Schmidt modificado es: ")
print("Q: \n", Q)
print("R: \n", R ,"\n")
print("Verificando la ortogonalidad  Q*Q: \n", Q.T @ Q)
print("Comprobamos que en efecto es factorización") 
print("A = \n", A)
print("QR = \n", Q@R)

# Ejercicio 2
'''
We have a linear regresion problem. That is, we want to have x that satisfies the relation b = Ax and minimize the error.

With the notation we want x (mean square estimator) to be the minimum error of

                b = Ax

with the design matrix A and response vector b.

'''
print("\n \n \n Regresión Lineal")

A = np.array([[1,2,0],[1,7,2],[1,7,9],[1,4,5]])
print("La matriz de diseño X es: \n", A)

b = np.array([[1],[2],[3],[1]])
print("Los valores de la variable respuesta es: \n", b)

Q, R = GramSchmidtModified(A)

b_hat = Q.T @ b
x = backward(R,b_hat)
print("los coeficientes son: \n", x)

# Ejercicio 3

def vandermonde(x,p):
    import numpy as np

    n = len(x)
    X = np.ones((n,p))

    for i in range(p):
        X[:,i] = x**i

    return X

def regresion(X,Y):
    '''
    This function solve for a least square estimator via QR factorization described in Algorithm 11.2 of the book (Threfeten)
    '''
    Q, R = GramSchmidtModified(X)

    beta = backward(R,Q.T @ Y)
    return beta

def RegresionPolinomial(n,p):
    '''
    This is a very particular function, its mission is to create an experiment. Imagine we have a regresion problem. We create the regresor with a partition of the interval of [0,4\pi]. Then we imagine that the observed variable is a transformation of such partition plus an error term that distribuyes standar normal. With the observations constucted, now its time to solve for the least square estimator. In this case we permit a polinomial fit, this is made by the vondermont matrix of the regresor. Finally the results are shown in a plot.
    
    Input: 
            It receives two parameters, n stands fot the number of "simulated" observation, or the data we can disposes. The second parameter p stand for a p-1 order polinomial regresion.

    Output: 
            The plot of the experiment.
    '''
    # Crea el regresor x como se indica
    x = np.zeros(n)

    for i in range(n):
        x[i] = 4*np.pi * (i+1) / n

    # Crear la matriz de vandermonde para regresión polinomial. (matriz de diseño)
    X = vandermonde(x,p)

    # Crear el vector de variables respuesta
    mu , sigma = 0 , 0.11
    error = norm.rvs(mu,sigma,n)
    Y = np.sin(x) + error

    # Regresión 
    beta = regresion(X,Y)

    # Graficar
    a,b = 0, 4*np.pi
    dominio = np.linspace(a,b,100)   #Dominio a graficar

    # Observando detalladamente en esta única línea se genera el polinomio
    y = lambda z: sum([coeff*z**j for (coeff, j) in zip(beta,range(p))])

    plt.scatter(x, Y, c ="blue")
    plt.plot(dominio, y(dominio),color = 'black', label = 'Ajuste con p = %u. \n' %p)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(" Ajuste polimonial con n =%u. \n" % n)
    plt.legend()
    plt.show()


# Grafica de la regresión polinomial del experimento con parametros
n = 100       # observaciones
p = 3       # polimonio de grado p-1
RegresionPolinomial(n,p)

# Grafica de la regresión polinomial del experimento con parametros
n = 1000       # observaciones
p = 3       # polimonio de grado p-1
RegresionPolinomial(n,p)

# Grafica de la regresión polinomial del experimento con parametros
n = 10000       # observaciones
p = 3       # polimonio de grado p-1
RegresionPolinomial(n,p)

# Grafica de la regresión polinomial del experimento con parametros
n = 100       # observaciones
p = 4       # polimonio de grado p-1
RegresionPolinomial(n,p)

# Grafica de la regresión polinomial del experimento con parametros
n = 1000       # observaciones
p = 4       # polimonio de grado p-1
RegresionPolinomial(n,p)

# Grafica de la regresión polinomial del experimento con parametros
n = 10000      # observaciones
p = 4       # polimonio de grado p-1
RegresionPolinomial(n,p)

# Grafica de la regresión polinomial del experimento con parametros
n = 100       # observaciones
p = 6       # polimonio de grado p-1
RegresionPolinomial(n,p)

# Grafica de la regresión polinomial del experimento con parametros
n = 1000       # observaciones
p = 6       # polimonio de grado p-1
RegresionPolinomial(n,p)

# Grafica de la regresión polinomial del experimento con parametros
n = 10000       # observaciones
p = 6       # polimonio de grado p-1
RegresionPolinomial(n,p)

# Grafica de la regresión polinomial del experimento con parametros
n = 100       # observaciones
p = 100       # polimonio de grado p-1
RegresionPolinomial(n,p)

# Grafica de la regresión polinomial del experimento con parametros
n = 1000       # observaciones
p = 100       # polimonio de grado p-1
RegresionPolinomial(n,p)

# Grafica de la regresión polinomial del experimento con parametros
n = 10000       # observaciones
p = 100       # polimonio de grado p-1
RegresionPolinomial(n,p)



# Ejercicio 4  No correr o explota la máquina

# n = 100000
# p = int(0.1*n)
# RegresionPolinomial(n,p)









