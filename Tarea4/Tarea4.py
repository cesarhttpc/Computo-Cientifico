# %%
import numpy as np
from funciones4 import GramSchmidtModified

# Ejercicio 2
# Buscamos eigenvalores de la matriz A

def diagonal(A,n):
    '''
    This functions receives a square numpy array (square matrix). 
    This makes an n-iterative QR factorization with shift. 
    
    
    '''

    m = len(A)
    sigma = A[m-1,m-1]
    
    T = A.copy()

    # n = 100
    for i in range(n):

        T = A - sigma*np.identity(m)
        Q, R = GramSchmidtModified(T)

        T = R@Q + sigma*np.identity(m)
        sigma =T[m-1,m-1]
    return T




epsilons = [1,3,4,5]

for epsilon in epsilons:

    print("Iteración QR con shift para: " )
    A = np.array([[8,1,0],[1,4,1/10**(epsilon)],[0,1/10**(epsilon),1]])

    T = diagonal(A,100)

    print("A =\n",A,"\nPor iteración QR\n",T,"\n")


# %%
# Ejercicio 5
# QR without shift 
def IterSinShift(A):
    n = 20  #Cantidad de iteraciones 
    T = A.copy()

    for k in range(n):
        Q,R = GramSchmidtModified(T)    
        T = R@Q 
    return T
print(T,"\n")


# Definimos la matriz A
A = np.array([[1/3,2/3,-2/3],[-2/3,2/3,1/3],[2/3,1/3,2/3]]) #Ortogonal
print(A)
print(IterSinShift(A),"\n")

A = np.array([[1,0,0],[0,-1,0],[0,0,1]]) #Ortogonal
print(A)
print(IterSinShift(A),"\n")

A = np.array([[3,-1,0],[-1,2,-1],[0,-1,3]]) # Arbitraria
A = GramSchmidtModified(A)[0]     #Ortogonal
print(A)
print(IterSinShift(A),"\n")

















