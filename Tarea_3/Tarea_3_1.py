import numpy as np
import matplotlib.pyplot as plt

# función -log(pi)
def U(x1,x2):
    return 10*(x1**2 - x2)**2 + (x2 - 0.25)**4

# malla del plano
x1 = np.linspace(-2,2,200)
x2 = np.linspace(-1,3,200)

X1, X2 = np.meshgrid(x1,x2)

Z = U(X1,X2)

plt.figure(figsize=(6,6))

# curvas de nivel
plt.contour(X1,X2,Z,levels=30)

# caminata MCMC
plt.plot(samples[:,0],samples[:,1],linewidth=0.5)

# puntos
plt.scatter(samples[:,0],samples[:,1],s=5)

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Metropolis-Hastings Walk")
plt.show()