# %%
import numpy as np
import random
from scipy.stats import uniform, beta 
import matplotlib.pyplot as plt
from scipy.special import gammaln


# def objetivo(p, N, x):

#     m = len(x) 
#     suma = 0
#     for i in range(m):

#         # termino = - np.log( np.math.factorial(x[i])) - np.log( np.math.factorial(N - x[i]))
#         termino = - np.log(float(np.math.factorial(x[i]))) - np.log(float(np.math.factorial(N - x[i])))

        
#         suma = termino + suma

#     f = m*np.log( np.math.factorial(N)) + suma + np.sum(x) *np.log(p) + (m* N - np.sum(x) ) *np.log(1-p) + 19*np.log(1-p)

#     return f


def objetivo(p, N, x):
    suma = 0
    m = len(x)

    for i in range(m):
        termino = - gammaln(x[i] + 1) - gammaln(N - x[i] + 1)
        suma += termino

    f = m * gammaln(N + 1) + suma + np.sum(x) * np.log(p) + (m * N - np.sum(x)) * np.log(1 - p) + 19 * np.log(1 - p)

    return f

def betadist(p):

    return 19*np.log( 1- p )


# objetivo(0.5,7,np.array([3,4,5]))






# %%


def MetropolisHastingsHibrido(x,tamañoMuestra = 10000):

    # Punto inicial
    x_0 = np.array([0.5,10])

    Cadena_p = np.zeros(tamañoMuestra)  # p
    Cadena_N = np.zeros(tamañoMuestra)  # N
    Cadena_p[0] = x_0[0]
    Cadena_N[0] = x_0[1] 


    for k in range(tamañoMuestra-1):


        N_max = 1000


        # Kernel híbrido
        # indices = [1,2,3,4,5]
        indices = [2]
        j = random.choice(indices)

        # Simulación de propuesta
        if j == 1:


            pass
            # alpha = CadenaAlpha[k]

            # CadenaAlpha[k+1] = CadenaAlpha[k]
            # CadenaLambda[k+1] = gamma.rvs(alpha + 20 , 1/(1 +  sum(t**alpha)))


        elif j == 2:

            p = beta.rvs(1, 20) 
            N = random.choice(np.arange(N_max))

            f_y = objetivo(p, N, x)
            f_x = objetivo(Cadena_p[k],Cadena_N[k],x)
            q_y = betadist(p)
            q_x = betadist(Cadena_p[k])

            if np.exp(f_y - f_x + q_x - q_y) > uniform.rvs(0,1):
                Cadena_p[k+1] = p
                Cadena_N[k+1] = N 

            else:
                Cadena_p[k+1] = Cadena_p[k]
                Cadena_N[k+1] = Cadena_N[k]






        # elif j == 3:

        #     alpha_p = expon.rvs(0) 
        #     lamda_p = gamma.rvs(alpha_p, scale = 1)


        #     f_y = LogObjetivo(alpha_p,lamda_p,t)
        #     f_x = LogObjetivo(CadenaAlpha[k], CadenaLambda[k],t)
        #     q_x = LogPropuesta3(CadenaAlpha[k],CadenaLambda[k])
        #     q_y = LogPropuesta3(alpha_p,lamda_p)

        #     if np.exp(f_y + q_x - f_x - q_y) > uniform.rvs(0,1):

        #         CadenaAlpha[k+1] = alpha_p
        #         CadenaLambda[k+1] = lamda_p
        #     else:
                
        #         CadenaAlpha[k+1] = CadenaAlpha[k]   
        #         CadenaLambda[k+1] = CadenaLambda[k]

        # else:

        #     CadenaLambda[k+1] = CadenaLambda[k]
            
        #     sigma = 0.5
        #     e = norm.rvs(0,sigma)
        #     alpha_p = CadenaAlpha[k] + e

        #     f_y = LogObjetivo(alpha_p,CadenaLambda[k+1],t)
        #     f_x = LogObjetivo(CadenaAlpha[k], CadenaLambda[k],t)

        #     if np.exp(f_y-f_x) > uniform.rvs(0,1):

        #         CadenaAlpha[k+1] = alpha_p
        #     else:
                
        #         CadenaAlpha[k+1] = CadenaAlpha[k]   
            


    return Cadena_p, Cadena_N



x = np.array([7,7,8,8,9,4,7,5,5,6,9,8,11,7,5,5,7,3,10,3])

Cadena_p, Cadena_N = MetropolisHastingsHibrido(x)

plt.plot(Cadena_p,Cadena_N)












# %%
n = 24
z = np.arange(n)
print(z)
lst = []
for i in range(100000):

    lst.append(random.choice(z))

plt.hist(lst, bins = n)


print(lst)



# %%

# np.math.factorial(4)


x = np.array([1,3,2,5])

np.math.factorial(x)