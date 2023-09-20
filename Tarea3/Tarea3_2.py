import numpy as np
from funciones3 import GramSchmidtModified, backward
from scipy.stats import norm
from scipy.linalg import inv


def regresion(X,Y):
    '''
    This function solve for a least square estimator via QR factorization described in Algorithm 11.2 of the book (Threfeten)

    Input: 
            X is a np.array([],[],[])
            Y is a np.array([, , ,])    don't confuse with a np.array([],[],[])
    '''
    Q, R = GramSchmidtModified(X)

    beta = backward(R,Q.T @ Y)
    return beta

# Parámetros
n = 6
d = 5

X = norm.rvs(0,1,(n,d))
beta = np.array([5,4,3,2,1])
epsilon = norm.rvs( 0 , 0.13, n)

y = X @ beta + epsilon

# Estimador por regresión lineal
beta_hat = regresion(X, y)

print("Estimador de mínimos cuadrados beta: ", beta_hat)
print("\n")

DeltaX = norm.rvs(0,0.01, (n,d))

beta_p = regresion(X+DeltaX, y)
print(beta_p)


beta_c = inv((X + DeltaX).T @ (X + DeltaX)) @ ( (X + DeltaX).T @y)




