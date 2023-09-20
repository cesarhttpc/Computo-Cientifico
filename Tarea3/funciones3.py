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



