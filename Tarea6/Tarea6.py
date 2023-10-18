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

def MetropolisHastings(n,r,tamañoMuestra = 50000, objetivo = posterior):

    q = beta(r+1,n-r+1)


    # Punto inicial (distribución inicial)
    x = uniform.rvs(0,1/2)  

    Muestra = np.zeros(tamañoMuestra)
    Muestra[0] = x

    for k in range(tamañoMuestra-1):

        # Simulación de q
        y = beta.rvs(r+1,n-r+1)

        # Cadena de Markov
        cociente = (objetivo(y,r,n)/objetivo(x,r,n))*(q.pdf(x)/q.pdf(y))
        p_min = min(1,cociente)

        # if ber(p_min,1) == 1:
        if uniform.rvs(0,1) < p_min :    # Forma alterna

            Muestra[k+1] = y
            x = y
        else:
            Muestra[k+1] = x

    return Muestra

# Semilla   
rnd = 4  # 2 y 4 predeterminado, 35 es mala

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
# Ejercicio 2 y 3

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
ax2.hist(muestra1,density=True,bins = 70)
ax3.plot(soporte, posterior2_vec, label = 'n = %u, r = %u' %(n2,r2))
ax4.hist(muestra2,density=True,bins = 70)

for ax in fig.get_axes():
    ax.label_outer()

plt.show()

# Graficas de la cadena de Markov
chequeo1 = MetropolisHastings(n1,r1,tamañoMuestra=80)
plt.plot(chequeo1)
plt.show()

chequeo2 = MetropolisHastings(n1,r1,tamañoMuestra=80)
plt.plot(chequeo2)
plt.show()



















# # %%
# # ----------------------------------------------
# # Ejercicio 2 

# #Distribución objetivo:
# posterior = lambda p: p**r*(1-p)**(n-r) * np.cos(np.pi* p) 
# soporte = np.linspace(0,1/2,100)

# plt.plot(soporte, posterior(soporte), label = 'f(x)')
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.title("Distribución posterior")
# plt.show()
# # plt.legend()
# # %%

# # Metropolis-Hastings

# # Punto inicial 
# x = 1/4

# tamañoMuestra = 100000  #Muestra
# Muestra = np.zeros(tamañoMuestra)
# Muestra[0] = x
# for k in range(tamañoMuestra-1):

#     # Simulación de q
#     y = uniform.rvs(0,1/2)

#     # Cadena de Markov
#     cociente = posterior(y)/posterior(x)
#     p_min = min(1,cociente)

#     # if ber(p_min,1) == 1:
#     if uniform.rvs(0,1) < p_min :    # Forma alterna

#         Muestra[k+1] = y
#         x = y
#     else:
#         Muestra[k+1] = x


# # print(Muestra)


# # %%
# #Constante de normalización
# omega = integrate.quad(posterior,0,0.5)[0]
# print(omega)


# # %%
# plt.hist(Muestra,density=True,bins = 80)

# plt.plot(soporte, posterior(soporte)/omega, label = 'f(x)')
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.title("Distribución posterior")

# # %%

# plt.plot(Muestra)



