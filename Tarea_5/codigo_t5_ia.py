import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import corner

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'figure.dpi': 120,
    'savefig.dpi': 150,
    'axes.linewidth': 1,
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linestyle': '--'
})

class SEIR():
    def __init__(self,sigma,gamma,N,t,dis):
        self.sigma=sigma
        self.gamma=gamma
        self.N=N
        self.t=t
        self.dis=dis
        
    def cam_t(self,t,dis):
        self.t=t
        self.dis=dis

    def f(self,i):
        S=self.y[i][0]
        E=self.y[i][1]
        I=self.y[i][2]
        return np.array([
            -self.beta*I*S/self.N,
            self.beta*I*S/self.N-self.sigma*E,
            self.sigma*E-self.gamma*I,
            self.gamma*I
        ])

    def J(self,i):
        S=self.y[i][0]
        E=self.y[i][1]
        I=self.y[i][2]
        return np.array([
            [-self.beta*I/self.N,0,-self.beta*S/self.N,0],
            [self.beta*I/self.N,-self.sigma,self.beta*S/self.N,0],
            [0,self.sigma,-self.gamma,0],
            [0,0,self.gamma,0]
        ])

    def ajuste_y(self,i):
        self.y[i]=np.maximum(self.y[i],0.0)
        self.y[i]*=self.N/np.sum(self.y[i])

    def sim(self,theta):
        u=theta[1:]
        den=1+np.sum(u)
        S0=self.N/den
        E0=self.N*u[0]/den
        I0=self.N*u[1]/den
        R0=self.N*u[2]/den
        self.y_0=np.array([S0,E0,I0,R0])
        self.beta=theta[0]
        self.y=np.zeros((self.dis,4))
        self.y[0]=self.y_0
        h=self.t/(self.dis-1)
        for i in range(1,self.dis):
            f_ev=self.f(i-1)
            J_ev=self.J(i-1)
            self.y[i]=self.y[i-1]+h*f_ev+(h**2)*np.dot(J_ev,f_ev)/2
            self.ajuste_y(i)
        return self.y

class c_i_mod():
    def __init__(self,q,sigma,dt):
        self.q=q
        self.sigma=sigma
        self.dt=dt
    def obt(self,E):
        E_mid=np.zeros_like(E)
        E_mid[0]=E[0]
        E_mid[1:] = 0.5*(E[1:] + E[:-1])
        return self.q*self.sigma*E_mid*self.dt

class verosimilitud():
    def __init__(self,eta,c):
        self.c=c
        self.eta=eta
    def eva(self,c_mod):
        return -np.sum((self.c-c_mod)**2)/(2*self.eta)

class priori():
    def __init__(self,mu,Sigma):
        self.mu=mu
        self.Sigma=Sigma
    def eva(self,til_theta):
        x_d=til_theta-self.mu
        b=np.linalg.solve(self.Sigma,x_d)
        return -0.5*np.dot(x_d,b)-np.sum(til_theta)

class posterior():
    def __init__(self,veros,prio,model,cmod):
        self.veros=veros
        self.prio=prio
        self.model=model
        self.cmod=cmod
    def eva(self,theta):
        if np.any(theta<=0):
            return -np.inf
        y=self.model.sim(theta)
        E=y[:,1]
        c_mod=self.cmod.obt(E)
        til_theta=np.log(theta)
        return self.veros.eva(c_mod)+self.prio.eva(til_theta)

class posterior_opt():
    def __init__(self,veros,prio,model,cmod):
        self.veros=veros
        self.prio=prio
        self.model=model
        self.cmod=cmod
    def eva(self,theta):
        if np.any(theta<=0):
            return 1e20
        y=self.model.sim(theta)
        E=y[:,1]
        c_mod=self.cmod.obt(E)
        til_theta=np.log(theta)
        return -1*(self.veros.eva(c_mod)+self.prio.eva(til_theta))

class Metropolis_Hastings():
    def __init__(self,gamma_p,x0,n_iter,post):
        self.gamma_p=gamma_p
        self.x0=x0
        self.n_iter=n_iter
        self.post=post
    def run(self):
        x_i=self.x0.copy()
        til_x=np.log(x_i)
        n=len(x_i)
        samples=np.zeros((self.n_iter,n))
        log_post_current=self.post.eva(x_i)
        for i in range(self.n_iter):
            til_xp=til_x+self.gamma_p*np.random.randn(n)
            xp=np.exp(til_xp)
            log_post_proposal=self.post.eva(xp)
            log_alpha=log_post_proposal-log_post_current
            if np.log(np.random.rand())<log_alpha:
                til_x=til_xp
                x_i=xp
                log_post_current=log_post_proposal
            samples[i]=x_i
        return samples

if __name__=="__main__":
    dir_path = r"C:\Users\chris\OneDrive\Escritorio\python\CIMAT\problemas inversos\Tarea_5\pandemic_infuenza_SF_1918.xls"
    df = pd.read_excel(dir_path, header=None)
    
    d_cases = np.array(df.iloc[6:,2].values)
    c = d_cases
    eta = np.var(c)
    N = 550000.0
    sigma = 1/3
    gamma = 1/5
    q = 0.5
    t = len(c)
    dis = t
    
    mod_SEIR = SEIR(sigma, gamma, N, t, dis)
    cimod = c_i_mod(q, sigma, t/(dis))
    
    x_0 = np.array([0.1, 10.0, 1.0, 0.1])
    mu = np.copy(x_0)
    Sigma = 0.01 * np.eye(4)
    
    prior = priori(mu, Sigma)
    vero = verosimilitud(eta, c)
    post = posterior(vero, prior, mod_SEIR, cimod)
    post_opt = posterior_opt(vero, prior, mod_SEIR, cimod)
    
    res = minimize(post_opt.eva, x_0, method='Nelder-Mead',
                  options={'maxiter': 10000, 'xatol': 1e-4, 'fatol': 1/eta, 'disp': False})
    x_opt_ne = res.x
    
    mu_actualizado = np.copy(x_opt_ne)
    prior_actualizado = priori(np.log(mu_actualizado), Sigma)
    post_actualizado = posterior(vero, prior_actualizado, mod_SEIR, cimod)
    
    met_p = Metropolis_Hastings(0.01, x_opt_ne, 20000, post_actualizado)
    samples = met_p.run()
    
    theta_mean = np.mean(samples[-5000:], axis=0)
    theta_std = np.std(samples[-5000:], axis=0)
    
    # Grafica 1: Cadenas MCMC por separado
    params_names = [r'\beta', r'u_E', r'u_I', r'u_R']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i in range(4):
        plt.figure(figsize=(8, 4))
        plt.plot(samples[:, i], color=colors[i], alpha=0.7, linewidth=0.6)
        plt.ylabel(f'${params_names[i]}$', fontsize=12)
        plt.xlabel('Iteracion', fontsize=12)
        plt.title(f'Cadena MCMC - ${params_names[i]}$', fontsize=12)
        mean_val = np.mean(samples[-5000:, i])
        std_val = np.std(samples[-5000:, i])
        plt.axhline(y=mean_val, color='k', linestyle='--', linewidth=1.5, label=f'Media = {mean_val:.3f}')
        plt.axhline(y=mean_val + std_val, color='gray', linestyle=':', linewidth=1)
        plt.axhline(y=mean_val - std_val, color='gray', linestyle=':', linewidth=1)
        plt.legend(loc='best', fontsize=9)
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()
    
    # Grafica 2: Distribuciones posteriores con corner
    samples_burn = samples[-5000:]
    labels = [r'$\beta$', r'$u_E$', r'$u_I$', r'$u_R$']
    fig = corner.corner(samples_burn, labels=labels, show_titles=True, 
                        title_kwargs={"fontsize": 10}, label_kwargs={"fontsize": 10},
                        color='#1f77b4', hist_kwargs={'density': True, 'alpha': 0.6})
    plt.suptitle('Distribuciones Posteriores', fontsize=12, y=0.98)
    plt.show()
    
    # Grafica 3: SEIR compartimentos con desviación
    sim_Seir_mh = mod_SEIR.sim(theta_mean)
    tiempos = np.arange(len(c))
    
    n_sims = 300
    idx_sims = np.random.choice(len(samples_burn), n_sims, replace=False)
    S_sims, E_sims, I_sims, R_sims = [], [], [], []
    
    for idx in idx_sims:
        theta_sim = samples_burn[idx]
        sim_temp = mod_SEIR.sim(theta_sim)
        S_sims.append(sim_temp[:, 0])
        E_sims.append(sim_temp[:, 1])
        I_sims.append(sim_temp[:, 2])
        R_sims.append(sim_temp[:, 3])
    
    S_sims = np.array(S_sims)
    E_sims = np.array(E_sims)
    I_sims = np.array(I_sims)
    R_sims = np.array(R_sims)
    
    comps = [(S_sims, 'Susceptibles', r'S(t)', '#1f77b4'),
             (E_sims, 'Expuestos', r'E(t)', '#ff7f0e'),
             (I_sims, 'Infectados', r'I(t)', '#2ca02c'),
             (R_sims, 'Recuperados', r'R(t)', '#d62728')]
    
    for sims, title, ylabel, color in comps:
        plt.figure(figsize=(8, 4))
        mean_sim = np.mean(sims, axis=0)
        std_sim = np.std(sims, axis=0)
        plt.fill_between(tiempos, mean_sim - 2*std_sim, mean_sim + 2*std_sim, 
                         alpha=0.3, color=color, label=r'$\pm 2\sigma$')
        plt.plot(tiempos, mean_sim, '-', color=color, linewidth=2, label='Media')
        plt.xlabel('Tiempo (dias)', fontsize=12)
        plt.ylabel(f'${ylabel}$', fontsize=12)
        plt.title(title, fontsize=12)
        plt.legend(loc='best', fontsize=9)
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()
    
    # Grafica 4: Incidencia con incertidumbre
    c_mod_sims = []
    for idx in idx_sims:
        theta_sim = samples_burn[idx]
        sim_temp = mod_SEIR.sim(theta_sim)
        E_temp = sim_temp[:, 1]
        c_temp = cimod.obt(E_temp)
        c_mod_sims.append(c_temp)
    
    c_mod_sims = np.array(c_mod_sims)
    c_mod_mean = np.mean(c_mod_sims, axis=0)
    c_mod_std = np.std(c_mod_sims, axis=0)
    
    plt.figure(figsize=(9, 5))
    plt.fill_between(tiempos, c_mod_mean - 2*c_mod_std, c_mod_mean + 2*c_mod_std, 
                     alpha=0.3, color='steelblue', label=r'$\pm 2\sigma$ MCMC')
    plt.plot(tiempos, c_mod_mean, '-', color='royalblue', linewidth=2, label='Media MCMC')
    plt.plot(tiempos, c, 'o', color='crimson', markersize=3, alpha=0.6, label='Datos')
    plt.xlabel('Tiempo (dias)', fontsize=12)
    plt.ylabel('Casos reportados', fontsize=12)
    plt.title('Incidencia - Modelo SEIR con incertidumbre', fontsize=12)
    plt.legend(loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()
    
    # Grafica 5: Predicción secuencial cada 5 días (como pediste)
    plt.figure(figsize=(10, 6))
    plt.plot(tiempos, c, 'o', color='crimson', markersize=3, alpha=0.5, label='Datos reales', zorder=10)
    
    colores_proy = plt.cm.viridis(np.linspace(0.2, 0.9, 15))
    idx_color = 0
    
    for dias_entreno in range(5, min(60, len(c)), 5):
        if dias_entreno + 5 > len(c):
            break
            
        datos_entreno = c[:dias_entreno]
        mod_temp = SEIR(sigma, gamma, N, dias_entreno, dias_entreno)
        cimod_temp = c_i_mod(q, sigma, dias_entreno/(dias_entreno))
        prior_temp = priori(np.log(x_opt_ne), Sigma)
        vero_temp = verosimilitud(eta, datos_entreno)
        post_temp = posterior(vero_temp, prior_temp, mod_temp, cimod_temp)
        
        res_temp = minimize(posterior_opt(vero_temp, prior_temp, mod_temp, cimod_temp).eva, 
                           x_opt_ne, method='Nelder-Mead',
                           options={'maxiter': 2000, 'disp': False})
        theta_temp = res_temp.x
        
        mod_proy = SEIR(sigma, gamma, N, dias_entreno + 5, dias_entreno + 5)
        sim_proy = mod_proy.sim(theta_temp)
        E_proy = sim_proy[:, 1]
        c_proy = cimod_temp.obt(E_proy)
        
        n_sims_proy = 100
        c_proy_sims = []
        for _ in range(n_sims_proy):
            theta_pert = theta_temp + np.random.randn(4) * theta_temp * 0.1
            theta_pert = np.maximum(theta_pert, 0.01)
            mod_pert = SEIR(sigma, gamma, N, dias_entreno + 5, dias_entreno + 5)
            sim_pert = mod_pert.sim(theta_pert)
            c_pert = cimod_temp.obt(sim_pert[:, 1])
            c_proy_sims.append(c_pert[dias_entreno:dias_entreno+5])
        
        c_proy_sims = np.array(c_proy_sims)
        c_proy_mean = np.mean(c_proy_sims, axis=0)
        c_proy_std = np.std(c_proy_sims, axis=0)
        
        t_pred = np.arange(dias_entreno, dias_entreno + 5)
        color = colores_proy[idx_color % len(colores_proy)]
        
        plt.plot(t_pred, c_proy_mean, '-', color=color, linewidth=1.5, alpha=0.8)
        plt.fill_between(t_pred, c_proy_mean - 2*c_proy_std, c_proy_mean + 2*c_proy_std, 
                         alpha=0.2, color=color)
        plt.axvline(x=dias_entreno - 0.5, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        idx_color += 1
    
    plt.xlabel('Tiempo (dias)', fontsize=12)
    plt.ylabel('Casos reportados', fontsize=12)
    plt.title('Predicci{\\o}n secuencial: entrenamiento hasta $T$ y proyeccion a $T+5$', fontsize=11)
    plt.legend(['Datos reales'], loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()