import numpy as np
import matplotlib.pyplot as plt
import corner
from scipy.stats import multivariate_normal
import pytwalk 
def prop_pi(x):
    ppi=np.exp(-10*(x[0]**2-x[1])**2-(x[1]-0.25)**4)
    return ppi

def prop_q(x,xp,gamma):
    palpha=np.exp(-0.5*np.linalg.norm(x-xp)**2/gamma**2)
    return palpha
# a

def Metropolis_Hastings(x,n,gamma,prop_pi,prop_q):
    x_i=np.copy(x)
    v_x=[]
    for _ in range(n):
        xp=x_i+gamma*np.random.normal(0,1,2)
        ec=np.log(prop_pi(xp))-np.log(prop_pi(x_i))+np.log(prop_q(xp,x_i,gamma))-np.log(prop_q(x_i,xp,gamma))
        logalpha=min(0,ec)
        u=np.random.uniform(0,1)
        if np.log(u)<=logalpha:
            x_i=xp
        v_x.append(x_i.copy())
        
    return np.array(v_x)
if __name__=="__main__":
    #b
    v_gamma=[0.001,0.05,0.1,1.0]
    gamma=1
    x_ini=np.random.uniform(-1,1,2)
    da_x=Metropolis_Hastings(x_ini,10000,gamma,prop_pi,prop_q)
    print(da_x.shape)
    da_x_5000=da_x[5000:]
    print(da_x_5000.shape)
    #c
    n=200
    x1=np.linspace(-2,2,n)
    x2=np.linspace(-2,2,n)

    X1,X2=np.meshgrid(x1,x2)
    #print(X1,X2)
    X=np.column_stack([X1.reshape(-1), X2.reshape(-1)])
    #X=X.T
    print(X.shape)
    Z=prop_pi(X.T)
    print(Z.shape)
    Z=Z.reshape(n,n)
    print(Z.shape)
    plt.figure(figsize=(6,6))
    plt.contour(X1,X2,-np.log(Z),levels=30)
    plt.show()
    plt.figure()
    plt.plot(da_x_5000[:,0],da_x_5000[:,1],linewidth=0.5)
    plt.scatter(da_x_5000[:,0],da_x_5000[:,1],s=5)

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Metropolis-Hastings caminata")
    plt.show()
    #d
    corner.corner(
        da_x_5000,
        labels=[r"$x_1$", r"$x_2$"],
        show_titles=True
    )
    plt.show()
    #e
    Pyt0=pytwalk.pytwalk(n=2,U=lambda x :-np.log(prop_pi(x)))
    x0=np.array([0.0, 0.0])
    xp0=np.array([1.0, 1.0])
    Pyt0.Run(10000, x0, xp0)
    Pyt0.Ana(start=5000)