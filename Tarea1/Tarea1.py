## %%
import numpy as np

##%%
def MM(A,B):

    n = A.shape[1]
    m = B.shape[0]

    if n == m:
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

    m = A.shape[0]
    
    L = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    U = A.copy()
    for k in range(m-1):
        for j in range(k+1,m):
            L[j,k] = U[j,k]/U[k,k]
            U[j,k:m] = U[j,k:m] - L[j,k] * U[k,k:m] 
            
    return L,U

def forward(L,b):

    y = np.zeros(len(b))

    for i in range(len(b)):
        aux = 0
        for j in range(i):
            aux += L[i,j]*y[j]
        y[i] = (b[i] - aux)/L[i,i]
    return y

def backward(U,b):
    
    dimention = U.shape
    L = np.zeros(dimention) 

    m = dimention[0]
    b_inv = np.zeros([m])
    # b_inv = b.copy()

    for i in range(m):
        L[i,:] = U[m-i-1]
        b_inv[i] = b[m-i-1]

    print(b_inv)
    return forward(L,b_inv)

##%%
#Main

# Ejemplo 1 
# A = np.array([[2,1,1,0],[4,3,3,1],[8,7,9,5],[6,7,9,8]])
# b = np.array([1,2,3,4])
# L = LU(A)[0]

# Ejemplo 2
# L = np.array([[1,0,0],[5,1,0],[8,3,1]])
# b = np.array([1,2,3])

# Ejemplo 3
# L = np.array([[2,0,0,0],[5,5,0,0],[8,3,2,0],[4,3,2,1]])
# b = np.array([2,15,3,20])

# print(L, "\n valor de la matriz y vector \n", b)

# print(forward(L,b))

# Ejemplo 4
U = np.array([[4,3,2,1],[8,3,2,0],[5,5,0,0],[2,0,0,0]])
b = np.array([20,3,15,2])

print(backward(U,b))

        


## %%
