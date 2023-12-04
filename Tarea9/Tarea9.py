# %%
import numpy as np
import random
from scipy.stats import uniform, beta ,bernoulli, poisson, hypergeom, gamma
import matplotlib.pyplot as plt
from scipy.special import gammaln


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


def MetropolisHastingsHibrido(x,tamañoMuestra = 10000):

    # Punto inicial
    N_max = 1000
    w = np.arange(max(x)+ 1, N_max)

    p_0 = uniform.rvs(0,1)
    N_0 = random.choice(w)

    Cadena_p = np.zeros(tamañoMuestra)  # p
    Cadena_N = np.zeros(tamañoMuestra)  # N
    Cadena_p[0] = p_0
    Cadena_N[0] = N_0

    for k in range(tamañoMuestra-1):

        # Kernel híbrido
        indices = [1,2,3,4,5]
        j = random.choice(indices)

        # Simulación de propuesta
        if j == 1:

            Cadena_N[k+1] = Cadena_N[k]
            a = np.sum(x) +1
            l = len(x)*Cadena_N[k]- np.sum(x) + 20
            p = gamma.rvs(a, scale = 1/l)

            f_y = objetivo(p, Cadena_N[k], x)
            f_x = objetivo(Cadena_p[k],Cadena_N[k],x)
            q_y = gamma.logpdf(p,a,scale = 1/l)
            q_x = gamma.logpdf(Cadena_p[k], a,scale = 1/l)

            if np.exp(f_y - f_x + q_x - q_y) > uniform.rvs(0,1):
                Cadena_p[k+1] = p

            else:
                Cadena_p[k+1] = Cadena_p[k]


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


        elif j == 3:

            Cadena_p[k+1] = Cadena_p[k] 
            N = Cadena_N[k] + hypergeom.rvs(50,9,5)

            f_y = objetivo(Cadena_p[k], N, x)
            f_x = objetivo(Cadena_p[k],Cadena_N[k],x)
            q_x = hypergeom.logpmf(Cadena_N[k],50,9,5)
            q_y = hypergeom.logpmf(N,50,9,5)
            if np.exp(f_y - f_x + q_x - q_y) > uniform.rvs(0,1):
                Cadena_N[k+1] = N 

            else:
                Cadena_N[k+1] = Cadena_N[k]


        elif j == 4 : 

            Cadena_p[k+1] = Cadena_p[k] 

            N = Cadena_N[k] + poisson.rvs(1)

            if N <= N_max:


                f_y = objetivo(Cadena_p[k], N, x)
                f_x = objetivo(Cadena_p[k],Cadena_N[k],x)

                if np.exp(f_y - f_x ) > uniform.rvs(0,1):
                    Cadena_N[k+1] = N 

                else:
                    Cadena_N[k+1] = Cadena_N[k]
            else: 

                Cadena_N[k+1] = Cadena_N[k]


        elif j == 5 :

            Cadena_p[k+1] = Cadena_p[k]
            N = Cadena_N[k] + 2*bernoulli.rvs(1/2)-1

            f_y = objetivo(Cadena_p[k], N, x)
            f_x = objetivo(Cadena_p[k],Cadena_N[k],x)

            if np.exp(f_y - f_x ) > uniform.rvs(0,1):
                Cadena_N[k+1] = N 

            else:
                Cadena_N[k+1] = Cadena_N[k]

    return Cadena_p, Cadena_N


#Datos dados
x = np.array([7,7,8,8,9,4,7,5,5,6,9,8,11,7,5,5,7,3,10,3])

Cadena_p, Cadena_N = MetropolisHastingsHibrido(x)

plt.plot(Cadena_p,Cadena_N)














# %%


# hypergeom.rvs(10,5)
hypergeom.logpmf(154,10000,2000, 800)
hypergeom.rvs(10000,2000,800)