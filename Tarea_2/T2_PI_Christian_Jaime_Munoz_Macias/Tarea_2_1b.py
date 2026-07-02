import numpy as np
import matplotlib.pyplot as plt

def k(C,gamma,x):
    return C*np.exp(-x**2/(2*gamma**2))

def g(k,f_e,x_d):
    return lambda x:np.trapezoid(k(x-x_d)*f_e,x_d)

def matriz(x,x_d):
    d=np.zeros((len(x),len(x)))
    d=d+x_d
    d=d-x[:,np.newaxis]
    return d

def f_true(x):
    f=np.zeros_like(x)
    f[(x>0.1)&(x<0.25)]=0.75
    f[(x>0.3)&(x<0.32)]=0.32
    f[(x>0.5)&(x<1)]=np.sin(2*np.pi*x[(x>0.5)&(x<1)])**4
    return f

def b_ruido(A,f,SNR):
    b_exact=A@f
    noise_std=np.max(np.abs(b_exact))/SNR
    np.random.seed(0)
    noise=noise_std*np.random.normal(0,1,size=len(f))
    b_noisy=b_exact+noise
    noise_level=np.linalg.norm(noise)
    return b_noisy,noise_level

def iteracion_landweber(K,b,x0,beta,tau,tol=1e-6,max_iter=1000):
    x=np.copy(x0)
    for i in range(max_iter):
        residual=b-K@x
        oper=K.T@residual
        x+=beta*oper
        
        if np.linalg.norm(residual)<tau*tol:
            print(f"iteracion de Landweber convergió en {i} iteraciones.")
            break

    return x    
def SVD(A):
    U,s,V=np.linalg.svd(A,full_matrices=False)
    return U,s,V
if __name__=="__main__":
    
    gamma=0.025
    C=1/np.sqrt(2*np.pi*gamma**2)

    n=[10,50,100]
    SNR=[10,20,100]
    for n_i in n:
        for SNR_i in SNR:
            h=1/n_i
            x=np.linspace(0,1,n_i)
            x_d=np.linspace(0,1,n_i)
            f_ev=f_true(x)
            ma_dif=matriz(x,x_d)
            K=h*k(C,gamma,ma_dif)
            b_rui,noise_level=b_ruido(K,f_ev,SNR_i)
            print("noise level:",noise_level)
            U,s,V=SVD(K)
            s_ord=np.sort(s)[::-1]
            #print(s_ord)
            beta=1/s_ord[0]**2
            print("beta:",beta)
            f_ob_land=iteracion_landweber(K,b_rui,np.zeros_like(b_rui),beta,1.01,tol=noise_level,max_iter=10000)
       
            plt.figure()

            plt.title(rf"$n={n_i},\,SNR={SNR_i}$.")
            plt.plot(x,f_ev,label=r"$f_{true}$")
            plt.plot(x,f_ob_land,label=r"$f_{Landweber}$")
            plt.grid(True,alpha=0.3)
            plt.legend()

            plt.tight_layout()
            plt.savefig(f"Tarea_2_1b_n{n_i}_SNR{SNR_i}.pdf")
            plt.show()
