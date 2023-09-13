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

def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    import numpy as np

    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)

def GramSchmidt(A):
    '''
    Ortogonalization algoritm 

    Given a  basis of the space in \mathbb{R}^n, this algoritm produce an ortonormal basis for the same space. We represent the input matrix A where each column is the a_j initial basis. 

        |   .     .         .  |    
        |   .     .         .  |
    A = |  a_1   a_2  ...  a_n |
        |   .     .         .  |
        |   .     .         .  |

    Input: 
            This function receives n numpy vectors.

    Output:
            This function produce the numpy matrix Q, where each column is a vector from the new basis.

    Example:

                                | 2|
            The basis a_1 =     | 1|
                                | 1|

                                | 1|
            The basis a_2 =     | 0|
                                |10|

                                | 2|
            The basis a_3 =     |-3|
                                |11|

            is transform into an ortonormal basis

                        [ 0.81649658 -0.34188173  0.46524211]
                  Q =   [ 0.40824829 -0.22792115 -0.88396   ]
                        [ 0.40824829  0.91168461 -0.04652421]

            By the comand:

            A = np.array([[2,1,2],[1,0,-3],[1,10,11]])
            GramSchmidt(A)[0]

    '''
    
    import numpy as np
    n = len(A)

    R = np.zeros([n,n])
    Q = np.zeros([n,n]) 
    V = np.zeros([n,n])

    
    for j in range(n):

        V[:,j] = A[:,j]
        for i in range(j):

            R[i,j] = np.dot(Q[:,i], A[:,j])

            V[:,j] = V[:,j] - R[i,j] * Q[:,i]   

        R[j,j] = np.sqrt(np.dot(V[:,j],V[:,j]))
        Q[:,j] = V[:,j] / R[j,j]        

    return Q,R

def GramSchmidtModified(A):
    '''
    Ortogonormalization algoritm via modified Gram-Schmidt or QR decomposition (reduced QR)
    
                                    A = QR
    where A is a n x m matrix, Q is a  n x m matrix and R is a m x m matrix.

    Given a  basis of the space in \mathbb{R}^n, this algoritm produce an ortonormal basis for the same space. We represent the input matrix A where each column is the a_j initial basis. 

        |   .     .         .  |    
        |   .     .         .  |
    A = |  a_1   a_2  ...  a_n |
        |   .     .         .  |
        |   .     .         .  |

    Input: 
            This function receives n numpy vectors.

    Output:
            This function produce the numpy matrix Q, where each column is a vector from the new basis.

    Example:

                                | 2|
            The basis a_1 =     | 1|
                                | 1|

                                | 1|
            The basis a_2 =     | 0|
                                |10|

                                | 2|
            The basis a_3 =     |-3|
                                |11|

            is transform into an ortonormal basis

                        [ 0.81649658 -0.34188173  0.46524211]
                  Q =   [ 0.40824829 -0.22792115 -0.88396   ]
                        [ 0.40824829  0.91168461 -0.04652421]

            By the comand:

            A = np.array([[2,1,2],[1,0,-3],[1,10,11]])
            GramSchmidtModified(A)[0]
    
    '''
    import numpy as np
    m,n = np.shape(A)
    V = np.zeros([m,n])
    R = np.zeros([n,n])
    Q = np.zeros([m,n])

    for i in range(n):
        
        V[:,i] = A[:,i] 

    for i in range(n):
        
        R[i,i] = np.sqrt(np.dot(V[:,i],V[:,i]))
        Q[:,i] = V[:,i] / R[i,i]

        for j in range(i+1,n):

            R[i,j] = np.dot(Q[:,i], V[:,j]) 
            V[:,j] = V[:,j] - R[i,j] * Q[:,i]


    return Q, R

def vandermonde(x,p):
    import numpy as np

    n = len(x)
    X = np.ones((n,p))

    for i in range(p):
        X[:,i] = x**i

    return X

