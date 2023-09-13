
import numpy as np
import time
from funciones2 import backward, GramSchmidtModified
from scipy.stats import norm


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

# print("\n\n\n\n")

# n = 17

# mu = 0
# sigma = 0.11
# error = norm.rvs(mu,sigma,n)

# x = np.zeros(n)

# for i in range(n):
#     x[i] = 4*np.pi * (i+1) / n

# senx = np.sin(x)

# y = senx + error





# print(y)
# print(error)




