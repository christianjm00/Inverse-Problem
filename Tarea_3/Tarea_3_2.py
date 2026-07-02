import numpy as np
import matplotlib.pyplot as plt
import corner
from pytwalk import pytwalk


def log_pi(x):
    return -10*(x[0]**2-x[1])**2-(x[1]-0.25)**4


def Metropolis_Hastings(x,n,gamma):
    x_i=np.copy(x)
    v_x=[]
    for _ in range(n):
        xp=x_i+gamma*np.random.normal(0,1,2)
        logalpha=min(0,log_pi(xp)-log_pi(x_i)+np.log(prop_q(xp,x_i,gamma))-np.log(prop_q(x_i,xp,gamma)))
        if np.log(np.random.rand())<=logalpha:
            x_i = xp
        v_x.append(x_i.copy())
    return np.array(v_x)


if __name__ == "__main__":
    v_gamma=[0.001,0.05,0.1,1.0]
    x_ini=np.random.uniform(-1,1,2)

    for gamma in v_gamma:
        print("gamma=",gamma)
        # generar muestras
        da_x=Metropolis_Hastings(x_ini,10000,gamma)
        da_x_5000=da_x[5000:]

        # inciso (c)
        n=200
        x1=np.linspace(-2,2,n)
        x2=np.linspace(-2,2,n)

        X1,X2=np.meshgrid(x1,x2)
        Z = np.zeros_like(X1)
        for i in range(n):
            for j in range(n):
                Z[i,j]=-log_pi(np.array([X1[i,j],X2[i,j]]))

        plt.figure(figsize=(6,6))

        plt.contour(X1,X2,Z,levels=30)

        plt.plot(da_x_5000[:,0],da_x_5000[:,1],linewidth=0.5)
        plt.scatter(da_x_5000[:,0],da_x_5000[:,1],s=5)

        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.title(f"Metropolis-Hastings caminata (gamma={gamma})")
        plt.savefig(f"T3_gamma={gamma}_ga_c.pdf")
        plt.show()
        # inciso (d)

        corner.corner(
            da_x_5000,
            labels=[r"$x_1$", r"$x_2$"],
            show_titles=True
        )
        plt.savefig(f"T3_gamma={gamma}_ga_d.pdf")
        plt.show()

    # inciso (e)
  
    def U(x):
        return 10*(x[0]**2-x[1])**2+(x[1]-0.25)**4
    Pyt=pytwalk(n=2, U=U)
    x0 = np.array([0.0,0.0])
    xp0 = np.array([1.0,1.0])
    Pyt.Run(10000,x0,xp0)
    Pyt.Ana(start=5000)