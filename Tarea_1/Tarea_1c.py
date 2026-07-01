import numpy as np
import matplotlib.pyplot as plt

def k(C,gamma,x):
    k_e=C*np.exp(-x**2/(2*gamma**2))
    return k_e

def g(k,f_e,x_d):
    return lambda x: np.trapezoid(k(x-x_d)*f_e,x_d)

def matriz(x,x_d):
    """
    x_d: dato de x medido \n
    x : dato de x que se quiere obtener \n
    """

    d=np.zeros((len(x),len(x)))
    d=d+x_d
    d=d-x[:,np.newaxis]
    return d

def filtro_TSVD(s,alpha):
    ver=s**2
    w=np.zeros_like(s)
    w[ver>alpha]=1
    return w

def SVD(A):
    U,s,V=np.linalg.svd(A, full_matrices=False)
    return U,s,V

def sol_TSVD(A,b,alpha):
    U,s,V=SVD(A)
    w=filtro_TSVD(s,alpha)
    x=V.T@np.diag(w/s)@U.T@b
    return x

def sol_Tikhonov(A,b,alpha):
    M=A.T@A+alpha*np.eye(A.shape[1])
    x=np.linalg.solve(M,A.T@b)
    #print(x)
    return x

def f_true(x):
    f=np.zeros_like(x)
    f[(x>0.1)&(x<0.25)]=0.75
    f[(x>0.3)&(x<0.32)]=0.32
    f[(x>0.5)&(x<1)]=np.sin(2*np.pi*x[(x>0.5)&(x<1)])**4
    return f

def b_ruido(A,f,SNR):
    b_exact=np.dot(A,f)
    noise_std=np.max(b_exact)/SNR
    np.random.seed(0)  
    b_noisy=b_exact+noise_std*np.random.normal(len(f))
    noise_level=np.linalg.norm(b_noisy-b_exact)
    return b_noisy,noise_level

def residual_function(A,b,noise_level,alpha,method='tikhonov'):
    if method=='tikhonov':
        f_reg=sol_Tikhonov(A,b,alpha)
    else:
        f_reg=sol_TSVD(A,b,alpha)
    data_residual = np.linalg.norm(A@f_reg-b)
    return data_residual-noise_level

def find_alpha_discrepancy(A,b,noise_level,alpha_min=1e-10,alpha_max=1.0,tol=1e-6,method='tikhonov'):
    def same_sign(a,b):
        return a*b>0
    def f(alpha):
        return residual_function(A,b,noise_level,alpha,method)
    f_min=f(alpha_min)
    f_max=f(alpha_max)
    if same_sign(f_min,f_max):
        return alpha_max
    low,high=alpha_min,alpha_max
    for i in range(50):
        mid=(low+high)/2.0
        if abs(high-low)<tol:
            break
        f_mid=f(mid)
        if same_sign(f(low),f_mid):
            low=mid
        else:
            high=mid
    return (low+high)/2.0

if __name__=="__main__":
    n=2**np.arange(1,10)
    SNR=10000
    
    gamma=0.025
    C=1/np.sqrt(2*np.pi*gamma**2)
    #for
    n_i=100
    x=np.linspace(0,1,n_i)
    x_d=np.linspace(0,1,n_i)
    f_ev=f_true(x)
    ma_dif=matriz(x,x_d)
    K=k(C,gamma,ma_dif)
    
    b_rui,noise_level=b_ruido(K,f_ev,SNR)
    alpha_tsvd=find_alpha_discrepancy(K,b_rui,noise_level,method='tsvd')
    alpha_tik=find_alpha_discrepancy(K,b_rui,noise_level,method='tikhonov')
    f_ob_TSVD=sol_TSVD(K,b_rui,alpha_tsvd)
    f_ob_Tik=sol_Tikhonov(K,b_rui,alpha_tik)
    #sol_TSVD
    plt.figure()
    plt.title("comparacion de los resultados")

    plt.title(f'n={n_i}, SNR={SNR}, '+r'\alpha_{Tik.} :'+f'{round(alpha_tik,2)}'+r'\alpha_{TSVD} :'+f'{round(alpha_tsvd,2)}')
    plt.plot(x,f_ev,label=r"f_{true}")
    plt.plot(x,f_ob_Tik,label=r"f_{Tikhonov}")
    plt.plot(x,f_ob_TSVD,label=r"f_{TSVD}")
    plt.grid(True,alpha=0.3)
    plt.legend()

    plt.show()
