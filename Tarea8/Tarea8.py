
# %% 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, gamma
import random 

np.random.seed(17)  # 21

def GaussDist(x, mu = 0, sigma = 1):

    f_x = 1/(2*np.pi*sigma**2) * np.exp( -(x-mu)**2 /(2*sigma)   )
    return f_x

def objetivo(x,y,rho,mu_x = 0,mu_y = 0,sigma_x = 1 ,sigma_y =1):

    f_x = 1 / (np.pi * sigma_x * sigma_y * np.sqrt(1-rho**2))* np.exp(-(((x-mu_x)/sigma_x)**2 + ((y-mu_y)/sigma_y)**2 - 2*rho*(x-mu_x)*(y-mu_y)/(sigma_x * sigma_y))/(2*(1-rho**2)))  

    return f_x

def MetropolisHastingsGIBBS(rho ,mu_x = 0, mu_y = 0, sigma_x = 1, sigma_y = 1, tamañoMuestra = 100000):

    # Punto inicial
    x_0 = np.array([0,0])

    MuestraX = np.zeros(tamañoMuestra)  
    MuestraY = np.zeros(tamañoMuestra)  
    MuestraX[0] = x_0[0]
    MuestraY[0] = x_0[1] 


    for k in range(tamañoMuestra-1):

        # Kernel híbrido
        indices = [1,2]
        j = random.choice(indices)

        # Simulación de propuesta
        if uniform.rvs(0,1) < 10**(-5):

            MuestraX[k+1] = MuestraX[k]
            MuestraY[k+1] = MuestraY[k]

        elif j == 1:

            y_previo = MuestraY[k]

            mu = mu_x + rho * sigma_x * (y_previo - mu_y) / sigma_y 
            sigma = sigma_x**2 * (1-rho**2)

            x_nuevo = norm.rvs(mu,sigma)

            MuestraX[k+1] = x_nuevo
            MuestraY[k+1] = y_previo

        elif j == 2:

            x_previo = MuestraX[k]

            mu = mu_y + rho* sigma_y * (x_previo -  mu_x)/ sigma_x 
            sigma = sigma_y * (1- rho**2)

            y_nuevo = norm.rvs(mu,sigma)

            MuestraX[k+1] = x_previo
            MuestraY[k+1] = y_nuevo

    return MuestraX,MuestraY



# Ejercicio 1

rho = 0.80

# Ejecutar cadena
MuestraX, MuestraY  = MetropolisHastingsGIBBS(rho)
plt.plot(MuestraX, MuestraY, linewidth = .5, color = 'teal')

# Grafica contornos
a_dominio, b_dominio = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))

Z1 = objetivo(a_dominio,b_dominio,rho)

plt.contour(a_dominio,b_dominio,Z1, levels = 3, linewidths = .5 ,cmap = 'inferno')
plt.title('MCMC')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.show()


plt.hist(MuestraX,bins=50, alpha = 0.8,density= True, label= 'x')
plt.hist(MuestraY,bins=50,alpha = 0.8, density= True, label= 'y')
plt.title("Histograma para x,y con rho = 0.80")
plt.show()

rho = 0.95

# Ejecutar cadena
MuestraX, MuestraY  = MetropolisHastingsGIBBS(rho)
plt.plot(MuestraX, MuestraY, linewidth = .5, color = 'teal')

# Grafica contornos
a_dominio, b_dominio = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))

Z1 = objetivo(a_dominio,b_dominio,rho)

plt.contour(a_dominio,b_dominio,Z1, levels = 3, linewidths = .5 ,cmap = 'inferno')
plt.title('MCMC')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.show()

plt.hist(MuestraX,bins=50, alpha = 0.8 , density= True, label= 'x')
plt.hist(MuestraY,bins=50, alpha = 0.8 , density= True, label= 'y')
plt.title("Histograma para x,y con rho = 0.95")
plt.show()


# Ejercicio 2

from scipy.special import gamma as Fgamma
from scipy.stats import weibull_min, expon


def LogObjetivo(alpha, lamda, t):

    n = len(t)
    p = np.prod(t)
    s = np.sum(t**alpha)


    f = (n + alpha -1)*np.log(lamda) - lamda *(s + 1) + alpha *np.log(p) - np.log(Fgamma(alpha)) + n*np.log(alpha) - alpha 
    
    return f


def LogPropuesta3(alpha, lamda):

    return -alpha -np.log(Fgamma(alpha)) + (alpha-1)*np.log(lamda)- lamda

def MetropolisHastingsHibrido(t,tamañoMuestra = 100000):

    # Punto inicial
    x_0 = np.array([1,1])

    CadenaAlpha  = np.zeros(tamañoMuestra)  # alpha
    CadenaLambda = np.zeros(tamañoMuestra)  # lambda
    CadenaAlpha[0]  = x_0[0]
    CadenaLambda[0] = x_0[1] 


    for k in range(tamañoMuestra-1):

        # Kernel híbrido

        j = uniform.rvs(0,1)

        # Simulación de propuesta
        if j <= 0.000:

            alpha = CadenaAlpha[k]

            CadenaAlpha[k+1] = CadenaAlpha[k]
            CadenaLambda[k+1] = gamma.rvs(alpha + 20 , 1/(1 +  np.sum(t**alpha)))


        elif (j > 0.000 and   j <=0.02):

            CadenaLambda[k+1] = CadenaLambda[k]

            beta = -sum (np.log(t)) + 1  ##
            if beta > 0 :

                CadenaAlpha[k+1] = gamma.rvs(21, scale = 1/beta)
            else:
                
                CadenaAlpha[k+1] = CadenaAlpha[k]




        elif j <= 0.5:

            alpha_p = expon.rvs(0) 
            lamda_p = gamma.rvs(alpha_p, scale = 1)


            f_y = LogObjetivo(alpha_p,lamda_p,t)
            f_x = LogObjetivo(CadenaAlpha[k], CadenaLambda[k],t)
            q_x = LogPropuesta3(CadenaAlpha[k],CadenaLambda[k])
            q_y = LogPropuesta3(alpha_p,lamda_p)

            if np.exp(f_y + q_x - f_x - q_y) > uniform.rvs(0,1):

                CadenaAlpha[k+1] = alpha_p
                CadenaLambda[k+1] = lamda_p
            else:
                
                CadenaAlpha[k+1] = CadenaAlpha[k]   
                CadenaLambda[k+1] = CadenaLambda[k]

        else:

            CadenaLambda[k+1] = CadenaLambda[k]
            
            sigma = 0.5
            e = norm.rvs(0,sigma)
            alpha_p = CadenaAlpha[k] + e

            f_y = LogObjetivo(alpha_p,CadenaLambda[k+1],t)
            f_x = LogObjetivo(CadenaAlpha[k], CadenaLambda[k],t)

            if np.exp(f_y-f_x) > uniform.rvs(0,1):

                CadenaAlpha[k+1] = alpha_p
            else:
                
                CadenaAlpha[k+1] = CadenaAlpha[k]   
            


    return CadenaAlpha,CadenaLambda

# t = weibull_min.rvs(1,1,size = 20)
t = expon.rvs(0,size = 20)

CadenaAlpha, CadenaLambda  = MetropolisHastingsHibrido(t)



plt.plot(CadenaAlpha,CadenaLambda,linewidth = .5, color = 'teal')
plt.title('MCMC')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\lambda$')
plt.show()

plt.hist(CadenaAlpha, bins= 50)
plt.title(r'$\alpha$')
plt.show()
        
plt.hist(CadenaLambda, bins=50)
plt.title(r'$\lambda$')
plt.show()

# plt.plot(CadenaAlpha)
# plt.plot(CadenaLambda)

print("Media muestral para alpha: " ,np.mean(CadenaAlpha))
print("Media muestral para lambda:", np.mean(CadenaLambda))


# Ejercicio 3

def MetropolisHastingsGibs(t, p, tamañoMuestra = 100000):

    # Punto inicial
    x = np.array([1.0,1,1,1,1,1,1,1,1,1,1])

    cadena = []
    cadena.append(x.copy())

    for k in range(tamañoMuestra -1):


        # kernel híbrido
        indices = [0,1,2,3,4,5,6,7,8,9,10]
        j = random.choice(indices)

        # Simulación de propuesta
        if j != 0:

            a = p[j-1] + 1.8 
            b = x[0] + t[j-1]

            x[j] = gamma.rvs(a, scale = 1/b)

            cadena.append(x.copy())

        else:
            
            a = 10 * 1.8 + 0.01
            b = 1 + sum(x[1:])

            x[0] = gamma.rvs(a, scale = 1/b)

            cadena.append(x.copy())

    return cadena


t = [94.32, 15.72, 62.88, 125.76, 5.24, 31.44, 1.05, 1.05, 2.1, 10.48]
p = [5,1,5,14,3,17,1,1,4,22]


cadena = np.array(MetropolisHastingsGibs(t,p))


plt.hist(cadena[:,0],bins = 50, density= True)
plt.title(r'$\beta$')
plt.show()
plt.hist(cadena[:,1], bins = 50, density= True)
plt.title(r'$\lambda_1$')
plt.show()
plt.hist(cadena[:,2], bins = 50, density= True)
plt.title(r'$\lambda_2$')
plt.show()
plt.hist(cadena[:,3], bins = 50, density= True)
plt.title(r'$\lambda_3$')
plt.show()
plt.hist(cadena[:,4], bins = 50, density= True)
plt.title(r'$\lambda_4$')
plt.show()
plt.hist(cadena[:,5], bins = 50, density= True)
plt.title(r'$\lambda_5$')
plt.show()
plt.hist(cadena[:,6], bins = 50, density= True)
plt.title(r'$\lambda_6$')
plt.show()
plt.hist(cadena[:,7], bins = 50, density= True)
plt.title(r'$\lambda_7$')
plt.show()
plt.hist(cadena[:,8], bins = 50, density= True)
plt.title(r'$\lambda_8$')
plt.show()
plt.hist(cadena[:,9], bins = 50, density= True)
plt.title(r'$\lambda_9$')
plt.show()
plt.hist(cadena[:,10], bins = 50, density= True)
plt.title(r'$\lambda_{10}$')
plt.show()
plt.plot(cadena, alpha = 0.9)


























