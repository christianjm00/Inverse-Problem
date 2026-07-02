import numpy as np
import matplotlib.pyplot as plt

def f(xi):
    return xi

def u_med(x):
    return x*(1-x)*(1+x)/6
    
def A(x,xi,h):
    n,m=x.shape[0],xi.shape[0]
    A=np.zeros((n,m))
    for j in range(n):
        for i in range(m):
            xi_i=xi[i]
            x_j=x[j]
            if x_j<=xi_i:
                A[j,i]=xi_i*(1-x_j)*h
            elif x_j>=xi_i:
                A[j,i]=x_j*(1-xi_i)*h
    return A

def b_ruido(b_exact,SNR):
    noise_std=np.max(np.abs(b_exact))/SNR
    np.random.seed(0)
    noise=noise_std*np.random.normal(0,1,size=len(b_exact))
    b_noisy=b_exact+noise
    noise_level=np.linalg.norm(noise)
    return b_noisy,noise_level,noise_std

def Q(n,delta,h):
    A=np.zeros((n-1,n-1))
    i=np.arange(n-1)
    A[i,i]=2.0
    A[i[1:],i[:-1]]=-1.0
    A[i[:-1],i[1:]]=-1.0
    A*=(delta)/(h**2)
    return A

def log_Q_1d2(Q):
    eigvals,eigvec=np.linalg.eigh(Q)   
    return 0.5*np.sum(np.log(eigvals))

def log_p_x(n,x,Q,log_Q_1d2,mu):
    ev_p=-0.5*n*np.log(2*np.pi)+0.5*log_Q_1d2-0.5*np.dot((x-mu),np.dot(Q,(x-mu)))
    return ev_p

def log_p_bx(A,x,b, lamb):
    ev_p_bx=-0.5*lamb*np.linalg.norm(np.dot(A,x)-b)**2
    return ev_p_bx

def log_p_xb(log_p_bx,log_p_x):
    return log_p_bx+log_p_x

def prop_q(x,xp,gamma):
    palpha=np.exp(-0.5*np.linalg.norm(x-xp)**2/gamma**2)
    return palpha

class log_p_xb_f():
    def __init__(self,log_p_xb,log_p_bx,log_p_x,log_Q_1d2,Q,A,b,lamb,n,mu):
        self.log_p_xb=log_p_xb
        self.log_p_bx=log_p_bx
        self.log_p_x=log_p_x
        self.log_Q_1d2=log_Q_1d2
        self.Q=Q
        self.A=A
        self.b=b
        self.lamb=lamb
        self.n=n
        self.mu=mu
        self.log_Q_1d2_ev=self.log_Q_1d2(self.Q)
    def fun(self,x):
        log_p_bx_ev=self.log_p_bx(self.A,x,self.b,self.lamb)
        log_p_x_ev=self.log_p_x(self.n,x,self.Q,self.log_Q_1d2_ev,self.mu)
        res=self.log_p_xb(log_p_bx_ev,log_p_x_ev)        
        return res

def Metropolis_Hastings(x,n,log_p_xb_f,gamma):
    x_i=np.copy(x)
    v_x=[]
    for _ in range(n):
        xp=x_i+gamma*np.random.normal(0,1,len(x_i))
        logalpha=min(0,log_p_xb_f(xp)-log_p_xb_f(x_i)+np.log(prop_q(xp,x_i,gamma))-np.log(prop_q(x_i,xp,gamma)))
        if np.log(np.random.rand())<=logalpha:
            x_i = xp
        v_x.append(x_i.copy())
    return np.array(v_x)
if __name__ == "__main__":
    n=100
    a=0.0
    b=1.0
    delta=0.1 # se tiene encontrar delta mejor
    gamma=0.2
    h_t=(b-a)/n
    
    x=np.linspace(a,b,n)
    xi=x[:-1]+0.5*h_t
    n_f=len(xi)
    A_ev=A(x,xi,h_t)
    b_ev=u_med(x)
    b_ev_n,noise_level,noise_std=b_ruido(b_ev,100)
    Q_ev=Q(n,delta,h_t)
    lamb = 1/(noise_std**2)
    mu=0.1*np.ones(len(xi))
    log_p_xb_fu=log_p_xb_f(log_p_xb,log_p_bx,log_p_x,log_Q_1d2,Q_ev,A_ev,b_ev_n,lamb,n,mu)
    
    x_ini=np.random.uniform(-1,1,len(xi))
    Metropolis_Hastings(x_ini,n,log_p_xb_fu.fun,gamma)