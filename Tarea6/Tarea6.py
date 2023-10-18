import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import uniform
from scipy.stats import beta

def ber(p,n, random_state = 2):
    '''
    Generador de v.a. Bernoulli de parámetro p
    '''
    uniforme = uniform.rvs(0,1,n, random_state = random_state)   

    bernoulli = np.zeros(n)
    for i in range(n):
        if uniforme[i] <= 1/3:
            bernoulli[i] = 1
            
    return bernoulli

def posterior(p,r,n):
    '''
    Función arbitraria a la cual se desea simular v.a. por medio de Metropolis-Hastings.
    '''
    if (p >= 0 and p <= 1/2): 
        f_x = p**r*(1-p)**(n-r) * np.cos(np.pi* p)
    else: 
        f_x = 0
    return f_x

def MetropolisHastings(n,r,tamañoMuestra = 20000, objetivo = posterior, propuesta = 'Beta'):
    '''
    Método de Metropolis-Hastings para simulación de variables aleatorias. Dicho método requiere de la previa simulación de una v.a. (simple) en este caso tomamos la distribución beta, como propuesta para generar simulaciones de una distribución descrita en la función objetivo (llamada posterior por aplicaciones a inferencia bayesiana). Dicho método tiene una distribución del punto inicial en una uniforme (0,1/2). La cadena de Markov generada recursivamente según el experimento Bernoulli donde la probabilidad de trancisión al estado propuesto es dada por la expresión con el mínimo y el cociente entre densidades.

    Input: 
        n:
            Es parámetro de la distribución objetivo (cantidad de simulaciones bernoulli)

        r:
            Es la cantidad de éxitos de la previa simulación Bernoulli, tambien es parámetro de la distribución posterior (ver pdf)

        tamañoMuestra: 
            Cantidad de pasos en la cadena de Markov.

        objetivo:
            Distribución objetivo a muestrear v.a.

        propuesta:
            Distribución propuesta de donde se simulan v.a. para generar de la objetivo según MH.
    
    '''
    q = beta(r+1,n-r+1)


    # Punto inicial (distribución inicial)
    x = uniform.rvs(0,1/2)  

    Muestra = np.zeros(tamañoMuestra)
    Muestra[0] = x

    for k in range(tamañoMuestra-1):

        if propuesta == 'Beta':
            # Simulación de q
            y = beta.rvs(r+1,n-r+1)
            # Cadena de Markov
            cociente = (objetivo(y,r,n)/objetivo(x,r,n))*(q.pdf(x)/q.pdf(y))
            p_min = min(1,cociente)

        elif propuesta == 'Uniforme':
            # Simulación de q
            y = uniform.rvs(0,1/2)
            # Cadena de Markov
            cociente = (objetivo(y,r,n)/objetivo(x,r,n))
            p_min = min(1,cociente)
        else:
            print('Error de elección de propuesta')

        # Transición de la cadena
        if uniform.rvs(0,1) < p_min :    #Ensayo Bernoulli
            Muestra[k+1] = y
            x = y
        else:
            Muestra[k+1] = x

    return Muestra

# Semilla   
rnd = 2  # 2 y 4

# Ejercicio 1
print("Ejercicio 1: ")
p = 1/3

n1 = 5
simulacion = ber(p,n1, random_state= rnd)
r1 = sum(simulacion)
print("Simulación para n = %u: "%n1, simulacion)
print("r: ",r1,"\n")

n2 = 40
simulacion = ber(p,n2, random_state= rnd)
r2 = sum(simulacion)
print("Simulación para n = %u: "%n2, simulacion)
print("r: ",r2,"\n")

# ----------------------------------------------
# Ejercicio 2 

# Graficar la función objetivo
soporte = np.linspace(0,1/2,100)

posterior1_vec = np.zeros(len(soporte))
for i in range(len(soporte)):
    posterior1_vec[i] = posterior(soporte[i],r1,n1)

posterior2_vec = np.zeros(len(soporte))
for i in range(len(soporte)):
    posterior2_vec[i] = posterior(soporte[i],r2,n2)

muestra1 = MetropolisHastings(n1,r1)
muestra2 = MetropolisHastings(n2,r2)

## Graficas Individiales
'''
plt.plot(soporte, posterior1_vec, label = 'n = %u, r = %u' %(n1,r1))
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Distribución posterior (objetivo)")
plt.legend()
plt.show()

plt.plot(soporte, posterior2_vec, label = 'n = %u, r = %u' %(n2,r2))
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Distribución posterior (objetivo)")
plt.legend()
plt.show()

plt.hist(muestra1,density=True,bins = 80)
plt.show()

plt.hist(muestra2,density=True,bins = 80)
plt.show()
'''

# Compilación gráficas
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('Metropolis-Hastings (n = %u,r =%u) y (n = %u, r= %u)'%(n1,r1,n2,r2))
ax1.plot(soporte, posterior1_vec, label = 'n = %u, r = %u' %(n1,r1))
ax2.hist(muestra1,density=True,bins = 60)
ax3.plot(soporte, posterior2_vec, color = 'orange',label = 'n = %u, r = %u' %(n2,r2))
ax4.hist(muestra2,color = 'orange',density=True,bins = 60)

for ax in fig.get_axes():
    ax.label_outer()

plt.show()

# Graficas de la cadena de Markov
chequeo1 = MetropolisHastings(n1,r1,tamañoMuestra=80)
chequeo2 = MetropolisHastings(n1,r1,tamañoMuestra=80)

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Cadena de Markov de M-H')
ax1.plot(chequeo1)
ax2.plot(chequeo2,color = 'orange')
plt.show()

# --------------------------------------
# Ejercicio 4

# Metropolis-Hastings para una propuesta uniforme en (0,1/2)
Muestra1Propuesta = MetropolisHastings(n1,r1,propuesta='Uniforme')
Muestra2Propuesta = MetropolisHastings(n2,r2,propuesta='Uniforme')

# Graficas de la simulación
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('Metropolis-Hastings (n = %u,r =%u) y (n = %u, r= %u)'%(n1,r1,n2,r2))
ax1.plot(soporte, posterior1_vec, label = 'n = %u, r = %u' %(n1,r1))
ax2.hist(Muestra1Propuesta,density=True,bins = 60)
ax3.plot(soporte, posterior2_vec, color = 'orange',label = 'n = %u, r = %u' %(n2,r2))
ax4.hist(Muestra2Propuesta,color = 'orange',density=True,bins = 60)

for ax in fig.get_axes():
    ax.label_outer()

plt.show()

# Graficas de la cadena de Markov
chequeo1Prop = MetropolisHastings(n1,r1,tamañoMuestra=80, propuesta= 'Uniforme')
chequeo2Prop = MetropolisHastings(n1,r1,tamañoMuestra=80, propuesta= 'Uniforme')

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Cadena de Markov de M-H propuesta uniforme')
ax1.plot(chequeo1Prop)
ax2.plot(chequeo2Prop,color = 'orange')
plt.show()
