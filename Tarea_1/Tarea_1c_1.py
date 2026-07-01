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

def filtro_TSVD(s,alpha):
    ver=s**2
    w=np.zeros_like(s)
    w[ver>alpha]=1
    return w

def SVD(A):
    U,s,V=np.linalg.svd(A,full_matrices=False)
    return U,s,V

def sol_TSVD(A,b,alpha):
    U,s,V=SVD(A)
    w=filtro_TSVD(s,alpha)
    x=V.T@np.diag(w/s)@U.T@b
    return x

def sol_Tikhonov(A,b,alpha):
    M=A.T@A+alpha*np.eye(A.shape[1])
    x=np.linalg.solve(M,A.T@b)
    return x

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

def residual_function(A,b,noise_level,alpha,method='tikhonov'):
    if method=='tikhonov':
        f_reg=sol_Tikhonov(A,b,alpha)
    else:
        f_reg=sol_TSVD(A,b,alpha)
    data_residual=np.linalg.norm(A@f_reg-b)
    return data_residual-noise_level


def find_alpha_discrepancy(A,b,noise_level,alpha_min=1e-10,alpha_max=1.0,tol=1e-6,method='tikhonov'):
    def same_sign(a,b):
        return a*b>0
    def f(alpha):
        return residual_function(np.copy(A),np.copy(b),noise_level,alpha,method)
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
            alpha_tsvd=find_alpha_discrepancy(K,b_rui,noise_level,method='tsvd')
            alpha_tik=find_alpha_discrepancy(K,b_rui,noise_level,method='tikhonov')
            f_ob_TSVD=sol_TSVD(K,b_rui,alpha_tsvd)
            f_ob_Tik=sol_Tikhonov(K,b_rui,alpha_tik)
            U,s,V=SVD(K)
            w=filtro_TSVD(s,alpha_tsvd)
            ATA_reg_tik=K.T@K+alpha_tik*np.eye(n_i)
            ATA_reg_tsvd=K.T@K+alpha_tsvd*np.eye(n_i)

            s_tik=np.linalg.svd(ATA_reg_tik,compute_uv=False)
            s_tsvd=np.linalg.svd(ATA_reg_tsvd,compute_uv=False)
            fig,ax=plt.subplots(1,2,figsize=(15,4))

            ax[0].set_title(rf"$n={n_i},\,SNR={SNR_i},\,\alpha_{{Tik}}={alpha_tik:.2e},\,\alpha_{{TSVD}}={alpha_tsvd:.2e}$")
            ax[0].plot(x,f_ev,label=r"$f_{true}$")
            ax[0].plot(x,f_ob_Tik,label=r"$f_{Tikhonov}$")
            ax[0].plot(x,f_ob_TSVD,label=r"$f_{TSVD}$")
            ax[0].grid(True,alpha=0.3)
            ax[0].legend()

            ax[1].loglog(s_tik,'o',label='Tikhonov')
            ax[1].loglog(s_tsvd,'x',label='TSVD')
            ax[1].axhline(np.max(np.abs(b_rui))/SNR_i,linestyle='--')
            ax[1].legend()

            plt.tight_layout()
            plt.savefig(f"Tarea_1c_n{n_i}_SNR{SNR_i}.pdf")
            plt.show()
