## %%
import numpy as np
from scipy.stats import uniform
from funciones import forward, backward, LUP, LinearSystem, cholesky
import time
import matplotlib.pyplot as plt

# if __name__ == "__main__":

# Ejercicio 3
# Dar la descomposición LUP para la matriz A y para una matriz aleatoria U(0,1).

A = np.array([[1,0,0,0,1],[-1,1,0,0,1],[-1,-1,1,0,1],[-1,-1,-1,1,1],[-1,-1,-1,-1,1]],dtype = float)
L,U,P = LUP(A)

print("Ejercicio 3")
print("Descomposición PA = LU \n L:", L ,"\n U:\n ", U ,"\n P:\n ", P)

D = uniform.rvs(0,1,(5,5), random_state=12)
print("Matriz A: \n",D)

L1,U1,P1 = LUP(D)

print("Descomposición PA = LU \n L:", L1 ,"\n U:\n ", U1 ,"\n P:\n ", P1)

# Ejercicio 4
# Resolver el sistema Ax = b y Dx = b con D y b aleatoria.
print("Ejercicio 4")
random_seed = [12,34,63,22,19]

for i in random_seed:

    #Generamos un vector b aleatorio 
    b = uniform.rvs(0,1,5, random_state = i)

    #Para la matriz de interes y la aleatoria resolvemos el sistema
    x = LinearSystem(A,b)
    x1 = LinearSystem(D,b)
    
    #Imprimimos en pantalla el vector solución x y vemos que dicha vector es solución
    print("Para el sistema Ax = b\n La solución es x =", x)
    print("Comprobamos que es solución. \nb  =",b,"\nAx =",A@x,"\n\n")

    print("Para el sistema Dx = b\n La solución es x =", x1)
    print("Comprobamos que es solución. \nb  =",b,"\nDx =",D@x1,"\n\n")


#Ejercicio 5

#Construimos una matriz simétrica y definida positiva
D = np.transpose(A) @ A

print("La matriz D es: \n ",D)

R = cholesky(D)
print("La matriz R es : \n",R)

print("Comprobamos que en efecto es factorizacion")
print("El producto de R*R: \n" ,np.transpose(R) @ R)

n = 60   #600
tiempoLU = np.zeros(n)
tiempoChl = np.zeros(n)

for i in range(1,n):

    A = uniform.rvs(0,1,(i,i))

    #Tiempo de ejecución para factorización LUP
    inicio = time.time()
    LUP(A)
    fin = time.time()

    tiempoLU[i] = fin - inicio

    #Tiempo de ejecución para Cholesky
    inicio = time.time()
    cholesky(A.T@A)
    fin = time.time()

    tiempoChl[i] = fin - inicio


plt.plot(tiempoLU,label = "L U")
plt.plot(tiempoChl,label = "Cholesky")
plt.xlabel("Tamaño (n) ")
plt.ylabel("Tiempo (ns)")
plt.title("Orden de los algoritmos")
plt.legend()





