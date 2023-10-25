# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm, uniform, expon
from scipy.special import gamma as Fgamma

np.random.seed(2)

# %%
def posterior(a,b,x):

    if (a >= 1 and a <= 4 and  b >= 0):
        n = int(len(x))
        r1 = np.prod(x)
        r2 = sum(x)
        f_post = (b**(n*a)/Fgamma(a)**n)*r1**(a-1)*np.exp(-b*(r2+1))
    else:
        f_post = 0
    return f_post


def posteriorGrafica(a,b,x):

    n = len(x)
    r1 = np.prod(x)
    r2 = sum(x)
    f_post = (b**(n*a)/Fgamma(a))*r1**(a-1)*np.exp(-b*(r2+1))

    return f_post


def MetropolisHastingsRW(x_i ,tamañoMuestra = 100000, propuesta = 'Normal'):

    # Punto inicial
    x = np.array([3,40])


    Muestra1 = np.zeros(tamañoMuestra)  # alpha
    Muestra2 = np.zeros(tamañoMuestra)  # beta
    Muestra1[0] = x[0]
    Muestra2[0] = x[1] 


    for k in range(tamañoMuestra-1):

        # Simulación de propuesta
        if propuesta == 'Normal':

            sigma1 = 0.05
            sigma2 = 0.5
            e1 = norm.rvs(0,sigma1)
            e2 = norm.rvs(0,sigma2)
            e = np.array([e1,e2])
            y = x + e 

        if propuesta == 'Uniforme':
            epsilon1 = 0.1
            epsilon2 = 2
            e1 = (1-2*uniform.rvs(0,1))*epsilon1
            e2 = (1-2*uniform.rvs(0,1))*epsilon2

            e = np.array([e1,e2])
            y = x + e

        # Cadena de Markov
        cociente = posterior(y[0], y[1], x_i)/posterior(x[0], x[1], x_i)
        p_min = min(1,cociente)

        # Transición de la cadena
        if uniform.rvs(0,1) < p_min :    #Ensayo Bernoulli
            Muestra1[k+1] = y[0]
            Muestra2[k+1] = y[1]
            x = y
        else:
            Muestra1[k+1] = x[0]
            Muestra2[k+1] = x[1]
        
    return Muestra1,Muestra2


# Simulación de muestra
x1_i = gamma.rvs(3, scale = 1/100, size = 4)
x2_i = gamma.rvs(3, scale = 1/100, size = 30)


# Grafica contorno inicial
print("Graficas de contorno")
# n = 4
a_dominio, b_dominio = np.meshgrid(np.linspace(1, 4, 200),
                                   np.linspace(0, 30, 200))

Z1 = posteriorGrafica(a_dominio,b_dominio,x1_i)
Z2 = posteriorGrafica(a_dominio,b_dominio,x2_i)

# n = 4
plt.contour(a_dominio,b_dominio,Z1, levels = 300,linewidths = .5 ,cmap = 'inferno')
plt.title('Contorno densidad a posterio (n = 4)')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.show()

# n = 30
plt.contour(a_dominio,b_dominio,Z2, levels = 300,linewidths = .5 ,cmap = 'inferno')
plt.title('Contorno densidad a posterior (n = 30)')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.show()

print('Graficas trayectoria RWMH')

# Grafica trayectoria n = 4
Muestra1_1, Muestra2_1 = MetropolisHastingsRW(x_i= x1_i)

plt.plot(Muestra1_1,Muestra2_1, linewidth = .5, color = 'gray')

# Grafica de contorno
a_dominio, b_dominio = np.meshgrid(np.linspace(1, 4, 200),
                                   np.linspace(0, 30, 200))
Z = posteriorGrafica(a_dominio,b_dominio,x1_i)

plt.contour(a_dominio,b_dominio,Z, levels = 300,linewidths = .5 ,cmap = 'inferno')
plt.plot(np.linspace(0.8,4,10),np.zeros(10),linestyle = '--', color = 'black') 
plt.plot(np.ones(10),np.linspace(-0.5, 40, 10),linestyle = '--', color = 'black')
plt.title('Metropolis-Hasting con caminata aleatoria (n = 4)')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.show()

# Grafica trayectoria n = 30
Muestra1_2, Muestra2_2 = MetropolisHastingsRW(x_i= x2_i)

plt.plot(Muestra1_2,Muestra2_2, linewidth = .5, color = 'gray')

# Grafica de contorno
a_dominio, b_dominio = np.meshgrid(np.linspace(1, 4, 200),
                                   np.linspace(0, 30, 200))
Z = posteriorGrafica(a_dominio,b_dominio,x2_i)

plt.contour(a_dominio,b_dominio,Z, levels = 300,linewidths = .5 ,cmap = 'inferno')
plt.plot(np.linspace(0.8,4,10),np.zeros(10)  , linestyle = '--', color = 'black') 
plt.plot(np.ones(10),np.linspace(-0.5, 40, 10),linestyle = '--', color = 'black')
plt.title('Metropolis-Hasting con caminata aleatoria (n=30)')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.show()


# Graficar la cadena de Markov

fig, (ax1, ax2) = plt.subplots(2, sharex=False)
fig.suptitle(r'Cadena de Markov de M-H (n = 4) para $\alpha$ y $\beta$ ')
# ax1.ylabel(r'$\alpha$')
ax1.plot(Muestra1_1)
plt.xlabel('t')
ax2.plot(Muestra2_1)
# plt.ylabel(r'$\beta$')
plt.show()

fig, (ax1, ax2) = plt.subplots(2, sharex=False)
fig.suptitle(r'Cadena de Markov de M-H (n = 30) para $\alpha$ y $\beta$')
ax1.plot(Muestra1_2)
ax2.plot(Muestra2_2)
plt.xlabel('t')
plt.show()

    
# Burn-in

plt.plot(np.log(posteriorGrafica(Muestra1_1[:10000],Muestra1_2[:10000],x1_i)))
plt.ylabel(r'log $f(a,\beta, x)$')
plt.xlabel('t')
plt.title('Log de distribución objetivo (n=4)')
plt.show()

plt.plot(np.log(posteriorGrafica(Muestra2_1[:10000],Muestra2_2[:10000],x2_i)))
plt.ylabel(r'log $f(a,\beta, x)$')
plt.xlabel('t')
plt.title('Log de distribución objetivo (n=30)')
plt.show()


burn_in = 6000

# Histogramas
fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (12,6))
fig.suptitle(r'Histograma para las marginales de la posterior (n=4) $\alpha$ y $\beta$ ')
ax1.hist(Muestra1_1[burn_in:],density=True,bins=40)
ax2.hist(Muestra2_1[burn_in:],density= True,bins=40)
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (12,6))
fig.suptitle(r'Histograma para las marginales de la posterior (n=30) $\alpha$ y $\beta$ ')
ax1.hist(Muestra1_2[burn_in:],density=True,bins=40)
ax2.hist(Muestra2_2[burn_in:],density= True,bins=40)
plt.show()

# Propuesta alternativa (uniforme)

Muestra1_1, Muestra2_1 = MetropolisHastingsRW(x_i= x1_i,tamañoMuestra=50000,propuesta='Uniforme')

plt.plot(Muestra1_1,Muestra2_1, linewidth = .5, color = 'gray')
plt.contour(a_dominio,b_dominio,Z1, levels = 300,linewidths = .5 ,cmap = 'inferno')
plt.title('Metropolis-Hasting con caminata aleatoria (n = 4) propuesta uniforme')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.show()


Muestra1_2, Muestra2_2 = MetropolisHastingsRW(x_i= x2_i,tamañoMuestra=50000,propuesta='Uniforme')

plt.plot(Muestra1_2,Muestra2_2, linewidth = .5, color = 'gray')
plt.contour(a_dominio,b_dominio,Z2, levels = 300,linewidths = .5 ,cmap = 'inferno')
plt.title('Metropolis-Hasting con caminata aleatoria (n = 4) propuesta uniforme')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.show()

# Graficar la cadena de Markov

fig, (ax1, ax2) = plt.subplots(2, sharex=False)
fig.suptitle(r'Cadena de Markov de M-H (n = 4) para $\alpha$ y $\beta$ (prop. unif)')
# ax1.ylabel(r'$\alpha$')
ax1.plot(Muestra1_1)
plt.xlabel('t')
ax2.plot(Muestra2_1)
# plt.ylabel(r'$\beta$')
plt.show()

fig, (ax1, ax2) = plt.subplots(2, sharex=False)
fig.suptitle(r'Cadena de Markov de M-H (n = 30) para $\alpha$ y $\beta$ (prop. unif)')
ax1.plot(Muestra1_2)
ax2.plot(Muestra2_2)
plt.xlabel('t')
plt.show()

burn_in = 6000

# Histogramas
fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (12,6))
fig.suptitle(r'Histograma para las marginales de la posterior (n=4) $\alpha$ y $\beta$ ')
ax1.plot(Muestra1_1[burn_in:6200])
ax2.plot(Muestra2_1[burn_in:6200])
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (12,6))
fig.suptitle(r'Histograma para las marginales de la posterior (n=30) $\alpha$ y $\beta$ ')
ax1.plot(Muestra1_2[burn_in:6200])
ax2.plot(Muestra2_2[burn_in:6200])
plt.show()



# Ejercicio 2

def MetropolisHastings(a ,tamañoMuestra = 30000, propuesta = 'Gamma'):
    
    
    # Punto inicial (distribución inicial)
    x = 900

    Muestra = np.zeros(tamañoMuestra)
    Muestra[0] = x

    for k in range(tamañoMuestra-1):

        if propuesta == 'Gamma':
            # Simulación de q
            y = gamma.rvs(int(a))

            # Cadena de Markov
            cociente = (y/x)**(a-int(a))
            p_min = min(1,cociente)

        elif propuesta == 'Uniforme':

            y = uniform.rvs(0,30)

            cociente = (y/x)**(a-1)*np.exp(x-y)
            p_min = min(1,cociente)

        elif propuesta == 'Exponencial': 
            # Simulación de q
            y = expon.rvs(scale = 1)
            # Cadena de Markov
            # cociente = (y/x)**(a-1)
            # cociente = (expon.pdf(y)/expon.pdf(x))* (np.exp((y-x)/100))
            beta = 10
            cociente = (y/x)**(a-1) *np.exp( x-y + y/beta -x/beta)
            p_min = min(1,cociente)
            # print("cociente: ",cociente)
        else:
            print('Error de elección de propuesta')

        # Transición de la cadena
        if uniform.rvs(0,1) < p_min :    #Ensayo Bernoulli
            Muestra[k+1] = y
            x = y
        else:
            Muestra[k+1] = x

    return Muestra

np.random.seed(12)   # Semilla: 12,42 

# Simular Gamma(a)
a = 7.73
Muestra = MetropolisHastings(a)


plt.plot(Muestra)
plt.title('Cadena de Markov de Metropolis-Hastings')
plt.ylabel('Gamma(a)')
plt.xlabel('t')
plt.show()


plt.plot(Muestra[:50])
plt.title('Primeros pasos en cadena de Markov de Metropolis-Hastings')
plt.ylabel('Gamma(a)')
plt.xlabel('t')
plt.show()


burn_in = 50
plt.hist(Muestra[50:],density= True, bins = 50, label= 'M-H')
dominio = np.linspace(0,30,100)
plt.plot(dominio,gamma.pdf(dominio,a), label = 'Gamma(a)')
plt.legend()
plt.title('Histograma Metropolis-Hasting prop Gamma([a])')
plt.ylabel('Gamma(a)')
plt.xlabel('t')
plt.show()

# Propuesta uniforme
Muestra = MetropolisHastings(a,propuesta = 'Uniforme', tamañoMuestra= 50000)

plt.plot(Muestra)
plt.title('Cadena de Markov de Metropolis-Hastings')
plt.ylabel('Gamma(a)')
plt.xlabel('t')
plt.show()


plt.plot(Muestra[:50])
plt.title('Primeros pasos en cadena de Markov de Metropolis-Hastings')
plt.ylabel('Gamma(a)')
plt.xlabel('t')
plt.show()


burn_in = 50
plt.hist(Muestra[50:],density= True, bins = 50, label= 'M-H')
dominio = np.linspace(0,30,100)
plt.plot(dominio,gamma.pdf(dominio,a), label = 'Gamma(a)')
plt.legend()
plt.title('Histograma Metropolis-Hasting prop Unif')
plt.ylabel('Gamma(a)')
plt.xlabel('t')
plt.show()


# plt.plot(Muestra[50:200])




# %%
z = expon.rvs(scale = 1, size = 30)
print(z)

