import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": False,
        "grid.alpha": 0.3,
        "lines.linewidth": 2,
        "figure.dpi": 120
    })

class rk4():
    def __init__(self,f,t0,tf,x_def,h):
        self.f=f
        self.t0=t0
        self.tf=tf
        self.x_def=x_def
        self.h=h
    def run(self):
        self.t=np.arange(self.t0,self.tf+self.h,np.abs(self.h))
        self.x=np.zeros(len(self.t))
        self.x[0]=self.x_def
        co_n=np.arange(len(self.t)-1)
        for n in co_n:
            var0=n
            var1=n+1
            k1=self.f(self.t[var0],self.x[var0])
            k2=self.f(self.t[var0]+self.h/2,self.x[var0]+self.h/2*k1)
            k3=self.f(self.t[var0]+self.h/2,self.x[var0]+self.h/2*k2)
            k4=self.f(self.t[var0]+self.h,self.x[var0]+self.h*k3)
            self.x[var1]=self.x[var0]+self.h/6*(k1+2*k2+2*k3+k4)
        return self.t,self.x

class ec_log():
    def __init__(self,r=1.0,K=100.0):
        self.r=r
        self.K=K
    def eva(self,t,x):
        return self.r*x*(1-x/self.K)

class ec_lamb():
    def __init__(self,x,t,r=1.0,K=100.0):
        self.t=t
        self.x=x
        self.r=r
        self.K=K
    def eva(self,t,lamb):
        x_t=self.x[np.argmin(np.abs(self.t-t))]
        return -self.r*(1-(2*x_t/self.K))*lamb

def ruido(x,t,t_obs):
    sigma=0.05*np.max(x)
    idx=[np.argmin(np.abs(t-ti)) for ti in t_obs]
    y=x[idx]+sigma*np.random.randn(len(t_obs))
    return y,sigma

def integ_lamb(lamb,t,t_obs,y_obs,sigma,h):
    v_lambda=np.zeros_like(t)
    idx_obs=[np.argmin(np.abs(t-ti)) for ti in t_obs]
    for n in range(len(t)-2,-1,-1):
        if (n+1) in idx_obs:
            pos=idx_obs.index(n+1)
            v_lambda[n+1]+=(1/sigma**2)*(lamb.x[n+1]-y_obs[pos])
        k1=lamb.eva(t[n+1],v_lambda[n+1])
        k2=lamb.eva(t[n+1]-h/2,v_lambda[n+1]-h/2*k1)
        k3=lamb.eva(t[n+1]-h/2,v_lambda[n+1]-h/2*k2)
        k4=lamb.eva(t[n+1]-h,v_lambda[n+1]-h*k3)
        v_lambda[n]=v_lambda[n+1]-h/6*(k1+2*k2+2*k3+k4)
    return v_lambda

def J(B,X0,X0b,sigma,t,t_obs,x,y_obs):
    idx=[np.argmin(np.abs(t-ti)) for ti in t_obs]
    return ((X0-X0b)**2)/(2*B)+np.sum((x[idx]-y_obs)**2)/(2*sigma**2)

def Grad_J(B,X0,X0b,lamb0):
    return (X0-X0b)/B+lamb0

def forward(ec,t0,T,X0,h):
    rk4_=rk4(ec.eva,t0,T,X0,h)
    return rk4_.run()

def adjunto(x_ev,v_t,t_obs,y_obs,sigma,h):
    ec_lamb0=ec_lamb(x_ev,v_t)
    return integ_lamb(ec_lamb0,v_t,t_obs,y_obs,sigma,h)

def eva_J(X0,ec,t0,T,h,B,X0b,sigma,t_obs,y_obs):
    t,x=forward(ec,t0,T,X0,h)
    return J(B,X0,X0b,sigma,t,t_obs,x,y_obs)

def eva_gJ(X0,ec,t0,T,h,B,X0b,sigma,t_obs,y_obs):
    t,x=forward(ec,t0,T,X0,h)
    lamb=adjunto(x,t,t_obs,y_obs,sigma,h)
    return Grad_J(B,X0,X0b,lamb[0])

def met_opt(eva_J,eva_gJ,X0,tau,n,cr_p):
    x_i=np.copy(X0)
    v_x=[x_i]
    v_gJ=[]
    v_J=[eva_J(x_i)]
    for i in range(n):
        g_ev=eva_gJ(x_i)
        v_gJ.append(g_ev)
        if np.abs(g_ev)<=cr_p:
            break
        x_i=x_i-tau*g_ev
        v_x.append(x_i)
        v_J.append(eva_J(x_i))
    return x_i,v_x,v_gJ,v_J

if __name__=="__main__":
    t0=0.0
    T=5.0
    h=(T-t0)/10000
    X0_true=10.0
    X0b=5.0
    X0=5.0
    B=10.0
    deltaX0=1.0
    ec=ec_log()
    t_obs=0.5*np.arange(1,11)

    t,x_true=forward(ec,t0,T,X0_true,h)
    y_obs,sigma=ruido(x_true,t,t_obs)

    t,x=forward(ec,t0,T,X0,h)
    lamb=adjunto(x,t,t_obs,y_obs,sigma,h)
    J0=J(B,X0,X0b,sigma,t,t_obs,x,y_obs)
    gJ=Grad_J(B,X0,X0b,lamb[0])
    print("gradiente adjunto:",gJ)

    eps=1e-6
    _,x_eps=forward(ec,t0,T,X0+eps,h)
    J_eps=J(B,X0+eps,X0b,sigma,t,t_obs,x_eps,y_obs)
    gJ_fd=(J_eps-J0)/eps
    print("gradiente FD:",gJ_fd)
    print("error relativo:",np.abs(gJ-gJ_fd)/np.abs(gJ_fd)*100,"%")

    V_E1,V_E2=[],[]
    var_eps=1/(10**np.arange(1,9,dtype=float))
    for e in var_eps:
        _,x_e=forward(ec,t0,T,X0+e*deltaX0,h)
        Je=J(B,X0+e*deltaX0,X0b,sigma,t,t_obs,x_e,y_obs)
        V_E1.append(np.abs(Je-J0))
        V_E2.append(np.abs(Je-(J0+e*gJ*deltaX0)))

    logE1=np.log(V_E1)
    logE2=np.log(V_E2)
    loge=np.log(var_eps)
    for i in range(1,len(var_eps)-1):
        print("i:",i)
        print("pendiente E1:",(logE1[i]-logE1[i-1])/(loge[i]-loge[i-1]))
        print("pendiente E2:",(logE2[i]-logE2[i-1])/(loge[i]-loge[i-1]))

    fJ=lambda X0_:eva_J(X0_,ec,t0,T,h,B,X0b,sigma,t_obs,y_obs)
    fgJ=lambda X0_:eva_gJ(X0_,ec,t0,T,h,B,X0b,sigma,t_obs,y_obs)
    x_op,v_x,v_gJ,v_J=met_opt(fJ,fgJ,X0,1e-3,10000,1e-6)
    print("x_op:",x_op)

    v_x=np.array(v_x)
    v_gJ=np.array(v_gJ)
    v_J=np.array(v_J)

    plt.figure()
    plt.plot(t,x_true,label='verdadera')
    plt.plot(t,x,label=f'forward X0={X0}')
    plt.plot(t,lamb,label='lambda')
    plt.legend()
    plt.grid()
    plt.savefig(f"gra_x_lamb_B_{B}.pdf")

    plt.figure()
    plt.loglog(var_eps,V_E1,label='E1')
    plt.loglog(var_eps,V_E2,label='E2')
    plt.legend()
    plt.grid()
    plt.savefig(f"gra_E1_E2_B_{B}.pdf")

    plt.figure()
    plt.plot(v_J)
    plt.title("J")
    plt.legend()
    plt.grid()
    plt.savefig(f"gra_J_{B}.pdf")

    plt.figure()
    plt.plot(np.abs(v_gJ))
    plt.title(r"$\left| \nabla J \right|$")
    plt.legend()
    plt.grid()
    plt.savefig(f"gra_nablaJ_B_{B}.pdf")

    plt.figure()
    plt.plot(np.abs(v_x-X0_true))
    plt.title(r"$\left| X_0^{(k)}-X_0 \right|$")
    plt.grid()
    plt.savefig(f"gra_X0_B_{B}.pdf")
    
    plt.show()