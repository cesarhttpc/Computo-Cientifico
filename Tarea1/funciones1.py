def MM(A,B):
    '''
    Matrix multiplication function.

    This functión multiplies the matrix A with B if this have sense.
    '''
    import numpy as np

    n = A.shape[1]
    m = B.shape[0]

    try: 
        mB = B.shape[1]
    except:
        mB = 1

    if n == m:    #To check the multiplication has the correct dimention
        M = np.zeros([A.shape[0],mB])
        for i in range(A.shape[0]):
            for j in range(mB):
                aux = 0
                for k in range(n):
                    aux +=  A[i,k] * B[k,j]
                M[i,j] = aux
        return M
    else:
        print("It's not possible to do this matrix multiplication")



    # #  Examples

    # #  Main
    # A = np.array([[1,0,0,0],[-2,1,0,0],[-4,0,1,0],[-3,0,0,1]])
    # B = np.array([[2,1,1,0],[4,3,3,1],[8,7,9,5],[6,7,9,8]])
    # print(MM(A,B))

def forward(L,b):
    '''
    Forward substitution

    This funtion solves a system of linear equations where the matrix of the coefficient form a lower triangular matrix.

    Input:


    Output:


    Example:
            # Main
            L = np.array([[2,0,0,0],[5,5,0,0],[8,3,2,0],[4,3,2,1]])
            b = np.array([2,15,3,20])
            print(forward(L,b))


    '''
    import numpy as np

    try:
        y = np.zeros(len(b))

        for i in range(len(b)):
            aux = 0    
            for j in range(i):
                aux += L[i,j]*y[j]   #Auxiliar variable that count the sum of the terms for the multiplication of L_i and y.
            y[i] = (b[i] - aux)/L[i,i]   # Form obtained by 'despeje' of the variable of interest.
        return y
    except:
        print("The system of equation can't be solved \n There is a cero in the diagonal")

def backward(U,b):
    '''
    Backward substitution
    
    This funtion solves a system of linear equations where the matrix of the coefficient form a upper triangular matrix.

    Input: 


    Output:


    Example:

            # Main
            U = np.array([[2,1,-3,1],[0,1,-4,-1],[0,0,4,5],[0,0,0,3]], dtype=float)
            b = np.array([16,33,16,9], dtype=float)
            print(backward(U,b))

    '''
    import numpy as np

    try:
        x = np.zeros(len(b))
        m = len(b)

        for i in range(m):
            aux = 0
            for j in range(i):
                aux += U[m-i-1,m-j-1] * x[m-j-1]      #Auxiliar variable that count the sum of the terms for the multiplication of U and x.
            x[m-i-1] = (b[m-i-1] - aux)/U[m-i-1,m-i-1]    # Form obtained by 'despeje' of the variable of interest.
        return x
    except:
        print("The system of equations can't be solved")

def LU(A):
    '''
    Factorization LU without pivoting.

    The input has to be a square matrix, and the diagonal elements must be non-zero values.

    Input:     
            Square matrix A with non-zero values in the diagonal.
    
    Output:
            Two square matrix L,U lower and upper trinagular, respetively.

    Example:
            # Main
            A = np.array([[2,1,1,0],[4,3,3,1],[8,7,9,5],[6,7,9,8]])
            L,U = LU(A)


    '''
    import numpy as np

    m = A.shape[0]

    L = np.identity(m)
    U = A.copy()
    for k in range(m-1):
        for j in range(k+1,m):
            L[j,k] = U[j,k]/U[k,k]
            U[j,k:m] = U[j,k:m] - L[j,k] * U[k,k:m] 
            
    return L,U

def LUP(A):
    '''
    Factorization LU with partial pivoting

    This function recieves a square matrix A and the output are the matrix L, U and P which satisfy the relation

    PA = LU



    Example:
            # Main
            A = np.array([[2,1,1,0],[4,3,3,1],[8,7,9,5],[6,7,9,8]], dtype = float)
            print(LUP(A))
    
    '''
    import numpy as np

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

def LinearSystem(A,b):
    '''
    This function solve a system of linear equations if it has a unique solution.

    Input:
            It receives a numpy matrix A and a numpy vector b, such that Ax = b.
    
    Output: 
            It returns the vector x that is the solution of the system.
    
    Example:

    
    
    '''
    try:   
        L,U,P = LUP(A)

        y = forward(L,P@b)
        x = backward(U,y)
    except:
        # If there are an error with the resolution with de LUP function then it's not possible to solve.
        print("The system of equations do not have a unique solution")

    return x

def cholesky(D):
    '''
    This function factorize a hermitian (simetric) positive define matrix D in the following way

                    D = R*R

    Input:
            D is a numpy array that form a positive define simetric matrix.

    Output:
            The matrix R.

    Example:
            A = np.array([[1,0,0,0,1],[-1,1,0,0,1],[-1,-1,1,0,1],[-1,-1,-1,1,1],[-1,-1,-1,-1,1]],dtype = float)
            D = np.transpose(A) @ A
            R = cholesky(D)
    '''
    import numpy as np
    R = D.copy()
    m = np.shape(D)[0]

    for k in range(m):
        for j in range(k+1,m):
            
            R[j,j:] = R[j,j:] - R[k,j:] * (R[k,j]/R[k,k])

        R[k,k:] = R[k,k:]/np.sqrt(R[k,k])

    # Since the former algoritm overwrite the input matrix, its necessesary to fill the below diagonal with zeros.
    for i in range(m):
        for j in range(i):
            R[i,j] = 0

    return R
