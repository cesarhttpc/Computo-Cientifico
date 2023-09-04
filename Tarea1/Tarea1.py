## %%
import numpy as np
from scipy.stats import uniform

##%%
def MM(A,B):
    '''
    Matrix multiplication function.

    This functión multiplies the matrix A with B if this have sense.
    '''
    n = A.shape[1]
    m = B.shape[0]

    if n == m:    #To check the multiplication has the correct dimention
        M = np.zeros([A.shape[0],B.shape[1]])
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                aux = 0
                for k in range(n):
                    aux +=  A[i,k] * B[k,j]
                M[i,j] = aux
        return M
    else:
        print("It's not possible to do this matrix multiplication")

def LU(A):
    '''
    Factorization LU without pivoting.

    The input has to be a square matrix, and the diagonal elements must be non-zero values.
    '''
    m = A.shape[0]

    L = np.identity(m)
    U = A.copy()
    for k in range(m-1):
        for j in range(k+1,m):
            L[j,k] = U[j,k]/U[k,k]
            U[j,k:m] = U[j,k:m] - L[j,k] * U[k,k:m] 
            
    return L,U

def forward(L,b):
    '''
    Forward substitution

    This funtion solves a system of linear equations where the matrix of the coefficient form a lower triangular matrix.

    '''
    y = np.zeros(len(b))

    for i in range(len(b)):
        aux = 0    
        for j in range(i):
            aux += L[i,j]*y[j]   #Auxiliar variable that count the sum of the terms for the multiplication of L_i and y.
        y[i] = (b[i] - aux)/L[i,i]   # Form obtained by 'despeje' of the variable of interest.
    return y

def backward1(U,b):
    '''
    Backward substitution
    
    This funtion solves a system of linear equations where the matrix of the coefficient form a upper triangular matrix.
    
    '''
    #Trasform the matrix of coeffient to a lower triangular matrix an we call the forward substitution.
    dimention = U.shape
    L = np.zeros(dimention) 

    m = dimention[0]
    b_inv = np.zeros([m])

    for i in range(m):  
        L[i,:] = U[m-i-1]
        b_inv[i] = b[m-i-1]

    print(b_inv)
    return forward(L,b_inv)

def LUP(A):
    '''
    Factorization LU with partial pivoting

    This function recieves a square matrix A and the output are the matrix L, U and P which satisfy the relation

    PA = LU
    
    '''
    m = A.shape[0]

    U = A.copy()
    L = np.identity(m,dtype= float)
    P = np.identity(m,dtype= float)

    for k in range(m-1):
        Index_max = k
        maxi = U[k,k]
        for i in range(k, m):
            if abs(U[i,k]) > abs(maxi):
                maxi = U[i,k]
                Index_max = i

        AuxRowU = U[k].copy()
        U[k] = U[Index_max]
        U[Index_max] = AuxRowU

        AuxRowL = L[k,: k].copy()
        L[k,: k] = L[Index_max,: k]
        L[Index_max,: k] = AuxRowL

        AuxRowP = P[k].copy()
        P[k] = P[Index_max]
        P[Index_max] = AuxRowP

        for j in range(k+1, m):
            L[j,k] = U[j,k]/U[k,k]
            U[j,k:m] = U[j,k:m] - L[j,k]*U[k,k:m] 
    
    return L,U,P



U = np.array([[1,1,1],[0,1,1],[0,0,1]], dtype=float)
b = np.array([1,1,1],dtype=float)

x = np.zeros(len(b))
m = len(b)

for i in range(m):
    aux = 0
    for j in range(m-i-1):
        aux += U[i,j]



#%%

m = 6
for i in range(m-1,-1,-1):
    print(i)




#%%











#Main

# If --name-- == "__main__":

# Ejercicio 3

# A = np.array([[1,0,0,0,1],[-1,1,0,0,1],[-1,-1,1,0,1],[-1,-1,-1,1,1],[-1,-1,-1,-1,1]],dtype = float)
# print("La factoriazación nos da que \n L: \n",LUP(A)[0],"\n U: \n",LUP(A)[1],"\n P:\n",LUP(A)[2])


# b = uniform.rvs(size=5)

# L,U,P = LUP(A)

# print(P@b)
# y = forward(L,P@b)
# print(y)

# x = backward(U,y)
# print(x)






# Ejemplo 1 
# A = np.array([[2,1,1,0],[4,3,3,1],[8,7,9,5],[6,7,9,8]])
# b = np.array([1,2,3,4])
# L = LU(A)[0]

# Ejemplo 2
# L = np.array([[1,0,0],[5,1,0],[8,3,1]])
# b = np.array([1,2,3])

# # Ejemplo 3
# L = np.array([[2,0,0,0],[5,5,0,0],[8,3,2,0],[4,3,2,1]])
# b = np.array([2,15,3,20])
# print(L, "\n valor de la matriz y vector \n", b)
# print(forward(L,b))

# Ejemplo 4
# U = np.array([[4,3,2,1],[8,3,2,0],[5,5,0,0],[2,0,0,0]])
# b = np.array([20,3,15,2])
# print(backward(U,b))

## Ejemplo 4.5
# U = np.array([[1,1,1],[0,1,1],[0,0,1]], dtype=float)
# b = np.array([1,1,1],dtype=float)
# print(backward(U,b))


# Ejemplo 5
# A = np.array([[2,1,1,0],[4,3,3,1],[8,7,9,5],[6,7,9,8]], dtype = float)
# print(LUP(A))


# A = np.array([[2,3,1,5],[6,13,5,19],[2,19,10,23],[4,10,11,31]],dtype = float)
        









## %%
