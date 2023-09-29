import numpy as np
from funciones4 import GramSchmidtModified

# Buscamos eigenvalores de la matriz A

# QR without shift 

# Definimos la matriz A
# A = np.array([[3,-1,0],[-1,2,-1],[0,-1,3]])
# print(A)

# n = 50
# T = A.copy()

# for k in range(n):
#     Q,R = GramSchmidtModified(T)    
#     T = R@Q 

# print(T,"\n")


# Iteración QR para valores propios (eigenvalores) 
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


# Ejercicio 2

epsilons = [1,3,4,5]

for epsilon in epsilons:

    print("Iteración QR con shift para: " )
    A = np.array([[8,1,0],[1,4,1/10**(epsilon)],[0,1/10**(epsilon),1]])

    T = diagonal(A,100)

    print("A =\n",A,"\nPor iteración QR\n",T,"\n")
















