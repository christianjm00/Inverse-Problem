import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from pytwalk import pytwalk

def f(xi):
    return xi

def u_med(x):
    return x*(1-x)*(1+x)/6

def A(x,xi,h):
    n,m=x.shape[0],xi.shape[0]
    A_mat=np.zeros((n,m))
    for j in range(n):
        x_j=x[j]
        for i in range(m):
            xi_i=xi[i]
            if x_j<=xi_i:
                A_mat[j,i]=xi_i*(1-x_j)*h
            else:
                A_mat[j,i]=x_j*(1-xi_i)*h
    return A_mat

def b_ruido(b_exact,SNR):
    noise_std=np.std(b_exact)/SNR
    np.random.seed(42)
    noise=noise_std*np.random.normal(0,1,size=len(b_exact))
    b_noisy=b_exact+noise
    noise_level=np.linalg.norm(noise)
    return b_noisy,noise_level,noise_std

def Q(n_f,delta,h):
    Q_mat=np.zeros((n_f,n_f))
    for i in range(n_f):
        Q_mat[i,i]=2.0
        if i>0:
            Q_mat[i,i-1]=-1.0
        if i<n_f-1:
            Q_mat[i,i+1]=-1.0
    Q_mat[0,0]=1.0
    Q_mat[n_f-1,n_f-1]=1.0
    Q_mat[0,1]=0.0
    Q_mat[n_f-1,n_f-2]=0.0
    Q_mat=Q_mat*(delta)/(h**2)
    return Q_mat

def log_posterior(x,A_mat,b_data,lamb,Q_mat,mu):
    residual=b_data-A_mat@x
    log_likelihood=-0.5*lamb*(residual@residual)
    diff=x-mu
    log_prior=-0.5*(diff@(Q_mat@diff))
    return log_likelihood+log_prior

def Metropolis_Hastings(x,n_iter,A_mat,b_data,lamb,Q_mat,mu,gamma):
    x_i=x.copy()
    n_f=len(x_i)
    samples=np.zeros((n_iter,n_f))
    acceptance=0
    log_post_current=log_posterior(x_i,A_mat,b_data,lamb,Q_mat,mu)
    for it in range(n_iter):
        xp=x_i+gamma*np.random.randn(n_f)
        log_post_proposal=log_posterior(xp,A_mat,b_data,lamb,Q_mat,mu)
        log_alpha=log_post_proposal-log_post_current
        if np.log(np.random.rand())<log_alpha:
            x_i=xp.copy()
            log_post_current=log_post_proposal
            acceptance+=1
        samples[it]=x_i.copy()
    return samples

def discrepancy_principle(delta,A_mat,b_data,lamb,mu,h,n_f,noise_level):
    Q_mat=Q(n_f,delta,h)
    ATA=A_mat.T@A_mat
    ATb=A_mat.T@b_data
    posterior_cov=np.linalg.inv(lamb*ATA+Q_mat)
    posterior_mean=posterior_cov@(lamb*ATb+Q_mat@mu)
    residual=np.linalg.norm(A_mat@posterior_mean-b_data)
    return residual-noise_level

def find_delta(A_mat,b_data,lamb,mu,h,n_f,noise_level,delta_min=1e-6,delta_max=1e3,tol=1e-4):
    for _ in range(50):
        delta_mid=(delta_min+delta_max)/2
        res_mid=discrepancy_principle(delta_mid,A_mat,b_data,lamb,mu,h,n_f,noise_level)
        if abs(res_mid)<tol:
            return delta_mid
        res_min=discrepancy_principle(delta_min,A_mat,b_data,lamb,mu,h,n_f,noise_level)
        if res_min*res_mid<0:
            delta_max=delta_mid
        else:
            delta_min=delta_mid
    return (delta_min+delta_max)/2

if __name__=="__main__":
    #inciso 3
    n=40
    a=0.0
    b=1.0
    gamma=0.3
    h=(b-a)/n
    xi=np.array([a+(i+0.5)*h for i in range(n)])
    x_obs=np.linspace(a,b,n+1)[1:-1]
    n_f=len(xi)
    
    A_ev=A(x_obs,xi,h)
    b_ev=u_med(x_obs)
    b_ev_n,noise_level,noise_std=b_ruido(b_ev,100)
    
    lamb=1/(noise_std**2)
    mu=np.zeros(n_f)
    delta_opt=find_delta(A_ev,b_ev_n,lamb,mu,h,n_f,noise_level)
    Q_ev=Q(n_f,delta_opt,h)
    #inciso 4
    ATA=A_ev.T@A_ev
    ATb=A_ev.T@b_ev_n
    posterior_cov=np.linalg.inv(lamb*ATA+Q_ev)
    posterior_mean=posterior_cov@(lamb*ATb+Q_ev@mu)
    
    x_ini=np.random.uniform(-3,3,n_f)
    samples=Metropolis_Hastings(x_ini,30000,A_ev,b_ev_n,lamb,Q_ev,mu,gamma)
    burn_in=15000
    samples_post_burn=samples[burn_in:]
    f_mean_mcmc=np.mean(samples_post_burn,axis=0)
    f_std_mcmc=np.std(samples_post_burn,axis=0)
    
    
    
    plt.figure(figsize=(6,18))
    plt.subplot(1,3,1)
    plt.plot(xi,f(xi),'b-',label='f real')
    plt.plot(xi,f_mean_mcmc,'r-',label='media MCMC')
    plt.plot(xi,posterior_mean,'g-',label='media exacta')
    plt.fill_between(xi,f_mean_mcmc-2*f_std_mcmc,f_mean_mcmc+2*f_std_mcmc,alpha=0.3,color='red')
    #plt.fill_between(xi,posterior_mean-2*posterior_cov,posterior_mean+2*posterior_cov,alpha=0.3,color='green')
    plt.legend()
    plt.title('Distribucion posterior')
    
    plt.subplot(1,3,2)
    plt.plot(samples[:1000,0],label='f[0]')
    plt.plot(samples[:1000,n_f//2],label=f'f[{n_f//2}]')
    plt.plot(samples[:1000,-1],label='f[-1]')
    plt.legend()
    plt.title('Primeras 1000 iteraciones')
    U=lambda x_i: log_posterior(x_i,A_ev,b_ev_n,lamb,Q_ev,mu)
    Pyt=pytwalk(n=len(xi), U=U)
    x0 =np.zeros(len(xi))
    xp0 = np.ones(len(xi))
    Pyt.Run(10000,x0,xp0)
    Pyt.Ana(start=5000)
    # 6
    mu_mcmc=np.mean(samples_post_burn,axis=0)
    cov_mcmc=np.cov(samples_post_burn.T)
    
    G=multivariate_normal(mean=mu_mcmc,cov=cov_mcmc+1e-6*np.eye(n_f))
    N=5000
    f_samples=G.rvs(N)
    log_G=G.logpdf(f_samples)
    
    log_pi=np.zeros(N)
    for i in range(N):
        log_pi[i]=log_posterior(f_samples[i],A_ev,b_ev_n,lamb,Q_ev,mu)
    
    log_weights=log_pi-log_G
    M=np.max(log_weights)
    log_z0_num=M+np.log(np.mean(np.exp(log_weights-M)))
    
    n_obs=len(b_ev_n)
    Q_inv=np.linalg.inv(Q_ev)
    Lambda=lamb*np.eye(n_obs)+A_ev@Q_inv@A_ev.T
    sign,logdet=np.linalg.slogdet(Lambda)
    log_z0_ana=-0.5*n_obs*np.log(2*np.pi)-0.5*logdet-0.5*(b_ev_n@np.linalg.solve(Lambda,b_ev_n)-mu@Q_ev@mu)
    
    plt.subplot(1,3,3)
    plt.bar(['Analitico','Numerico'],[log_z0_ana,log_z0_num])
    plt.title('Constante de normalizacion')
    plt.tight_layout()
    plt.show()
    
    print(f'log(z0) analitico: {log_z0_ana:.6f}')
    print(f'log(z0) numerico: {log_z0_num:.6f}')
    print(f'diferencia: {abs(log_z0_ana-log_z0_num):.6f}')