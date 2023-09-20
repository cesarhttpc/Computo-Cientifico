#%%
import numpy as np
from scipy.stats import  norm
from funciones3 import GramSchmidtModified, cholesky
from scipy.linalg import norm as norma #norma de una matriz
from scipy.linalg import cholesky as scipycholesky
import time


#%%
# Generación de los valores propios
def eigenvalue(alpha):
    '''
    Esta función nos da un arreglo de valores decrecientes según la formula 
    
        \lambda_i = \frac{\alpha^{20}}{\alpha^i}
         
    '''
    n = 20
    lamda = np.zeros(n)
    for i in range(n):
        lamda[i] = alpha **20 / alpha**(i+1)
    return lamda


#%%
# Generar la matriz aleatoria
n = 20  

A = norm.rvs(0,1,(n,n), random_state=12)
Q, R = GramSchmidtModified(A)


#%%
# Cholesky con matriz bien condicionada
print("Descomposición Cholesky para matriz bien condicionada")

alpha = 1.1
D = np.diag(eigenvalue(alpha))

# Simular errores epsilon
epsilon = norm.rvs(0, 0.02, n, random_state = 3)
lamda_eps = eigenvalue(alpha) + epsilon

D_eps = np.diag(lamda_eps)

# Construcción de las matrices B
B     = Q.T @   D   @ Q
B_eps = Q.T @ D_eps @ Q 

# Descomposición Cholesky
R1 = cholesky(B)
R1_eps = cholesky(B_eps)


DeltaR1 = R1 - R1_eps

print("Descomentar para ver la matriz R (no impresa debido a su tamaño) \n")
# print("Diferencia bien condicionada :\n", DeltaR1)   # Impresión de pantalla

#%%

# Cholesky con matriz mal condicionada
print("Descomposición Cholesky para matriz mal condicionada")

inicio = time.time()

alpha = 5
D = np.diag(eigenvalue(alpha))

# Simular errores epsilon
epsilon = norm.rvs(0, 0.02, n, random_state = 3)
lamda_eps = eigenvalue(alpha) + epsilon

D_eps = np.diag(lamda_eps)

# Construcción de las matrices B
B     = Q.T @   D   @ Q
B_eps = Q.T @ D_eps @ Q 

# Descomposición Cholesky
R2 = cholesky(B)
R2_eps = cholesky(B_eps)

DeltaR2 = R2 - R2_eps
# print("Diferencia mal condicionada :\n", DeltaR2)  #Impresión de pantalla
print("Descomentar para ver la matriz R (no impresa debido a su tamaño) ")

fin = time.time()
print("Tiempo de ejecución para mal condicionado:",fin - inicio, "\n")


#%%

print("Norma en L2 para B bien condicionada: ", norma(DeltaR1))
print("Norma en L2 para B mal  condicionada: ", norma(DeltaR2),"\n")



# %%

print("Descomposición Cholesky dada por scipy")
inicio = time.time()
R_sci = scipycholesky(B)
R_sci_eps = scipycholesky(B_eps)

# print("R de Cholesky dada por scicpy: \n",R_sci)   # Impresión de pantalla
print("Norma de R dada por scipy: ",norma(R_sci- R_sci_eps))

fin = time.time()

print("Tiempo Scipy: ", fin - inicio)

