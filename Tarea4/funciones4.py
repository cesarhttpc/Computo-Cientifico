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

