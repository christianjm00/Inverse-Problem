# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 23:52:23 2025

@author: yaded
"""

from numpy.random import uniform, normal
from numpy import ones, zeros, cumsum, shape, cov, mean, ceil, sqrt
from numpy import floor, exp, log, sum, pi, savetxt, loadtxt, array

import numpy as np
mat = np.asmatrix

from time import time, localtime, strftime
try:
    from tqdm.auto import trange as _twalk_trange
except Exception:
    _twalk_trange = None


try:
    from pylab import plot, hist, xlabel, ylabel, title
except:
    print("pytwalk: WARNING: pylab module not available, Ana, TS and Hist methods will fail.")


#### Funciones auxiliares y constantes
# norma al cuadrado (suma de cuadrados de los componentes)
def SqrNorm(x):
    return sum(x*x)

log2pi = log(2*pi)
log3 = log(3.0)


def Remain( Tr, it, sec1, sec2):
    """Información estimada de tiempo restante.
    Tr: total de iteraciones
    it: iteración actual
    sec1: tiempo de inicio (time())
    sec2: tiempo actual (time())
    Devuelve un string amigable con el tiempo restante o la hora estimada de finalización.
    """

    # segundos restantes estimados
    ax = int( (Tr - it) *  ((sec2 - sec1)/it) )

    if (ax < 1):
        return " "

    if (ax < 60):
        return "Finish in approx. %d sec." % (ax,)

    if (ax <= 360):
        return "Finish in approx. %d min and %d sec." % ( ax // 60, ax % 60)

    if (ax > 360):
        ax += sec2  # tiempo final estimado
        return "Finish by " + strftime("%a, %d %b %Y, %H:%M.", localtime(ax))



class pytwalk:
    """
    Clase principal que implementa el algoritmo t-walk.

    Parámetros importantes del constructor:
      n    : dimensión del espacio de parámetros.
      U    : función objetivo U(x) = -log(f(x)) en forma de energía (por defecto: 0.5*x^2).
      Supp : función que devuelve True si x está en el soporte (por defecto: todo el R^n).
      t,u,w: (opcional) para uso con penalizaciones/termalización; implementado para compatibilidad.

    El muestreador mantiene internamente el historial en self.Output y tasas de aceptación en self.Acc.
    """

    def __init__( self, n, U=(lambda x: sum(0.5*x**2)), Supp=(lambda x: True),
        t=-1, u=(lambda x: sum(0.5*x**2)), w=(lambda x: 0.0),
        ww=[0.0000, 0.4918, 0.4918, 0.0082 + 0.0082, 0.0], aw=1.5, at=6.0, n1phi=4.0):

        # dimensión
        self.n = n
        self.t = t

        # soporte para el caso en que se usan t, u, w (para estimación termalizada)
        if self.t >= 0: ### Penalized likelihood / temperatura
            self.LikelihoodEnergy = u
            self.PriorEnergy = w
            self.Output_u = array([0.0])
        else:  ### Caso normal: U es la energía total
            self.PriorEnergy = (lambda x: 0.0) 
            self.LikelihoodEnergy = U
            self.t = 1.0

        # función U interna (combina likelihood y prior según el caso)
        self.U = (lambda x: self.Energy(x))
        self.Supp = Supp

        # arreglos para guardar la cadena; se inicializan vacíos
        self.Output = zeros((1, n+1))
        self.Output_u = array([0.0])
        self.T = 1
        self.Acc = zeros(6)  # acceptance rates por kernel y global

        #### Probabilidades acumuladas para escoger kernel
        self.Fw = cumsum(ww)
        
        #### Parámetros para las propuestas
        self.aw = aw  # para Walk
        self.at = at  # para Traverse

        # probabilidad de mover cada parámetro
        self.pphi = min( n, n1phi)/(1.0*n)
        self.WAIT = 30

    def Energy( self, x):
        """Combina las partes de likelihood y prior si aplica."""
        self.ll_e = self.LikelihoodEnergy(x)
        self.prior_e = self.PriorEnergy(x)
        return self.t*self.ll_e + self.prior_e

    def _SetUpInitialValues( self, x0, xp0):
        """Verifica valores iniciales x0 y xp0 estén en el soporte y sean distintos por coordenada.
        Devuelve [bool_ok, U(x0), U(xp0)].
        """

        if any(abs(x0 -xp0) <= 0):
            print("pytwalk: ERROR, not all entries of initial values different.")
            return [ False, 0.0, 0.0]

        if not(self.Supp(x0)):
            print("pytwalk: ERROR, initial point x0 out of support.")
            return [ False, 0.0, 0.0]
        u = self.U(x0)

        if not(self.Supp(xp0)):
            print("pytwalk: ERROR, initial point xp0 out of support.")
            return [ False, u, 0.0]
        up = self.U(xp0)
        
        return [ True, u, up]



    def Run( self, T, x0, xp0, t=1):
        """
        Ejecuta el t-walk por T iteraciones.
          - x0, xp0: dos puntos iniciales dentro del soporte y cada coordenada distinta entre sí.
          - t: factor de temperatura (opcional, por compatibilidad).

        La función llena self.Output con las muestras (fila: parámetros..., última columna: U).
        """
        
        self.t = t

        sec = time()
        print("pytwalk: Running the twalk with %d iterations"\
            % (T,), end=' ')
        if self.t == 1:
            print(". ",  strftime("%a, %d %b %Y, %H:%M:%S.", localtime(sec)))
        else:
            print(" (%f). " % (self.t,), strftime("%a, %d %b %Y, %H:%M:%S.", localtime(sec)))

        ### Validar puntos iniciales
        [ rt, u, up] = self._SetUpInitialValues( x0, xp0)

        if (not(rt)):
            return 0
        
        sec2 = time() # último tiempo usado para estimar tiempo restante
        print("       " + Remain( T, 2, sec, sec2))

        x = x0
        xp = xp0

        # Reservar espacio para la salida
        self.Output = zeros((T+1, self.n+1))
        self.Output_u = zeros(T+1)
        self.T = T+1
        self.Acc = zeros(6)
        kercall = zeros(6) # cantidad de veces que se llamó cada kernel

        # guardar estado inicial
        self.Output[ 0, 0:self.n] = x.copy()
        self.Output[ 0, self.n] = u
        self.Output_u[0] = self.ll_e

        j1=1
        j=0

        # loop principal
        iterable = _twalk_trange(T, desc='pytwalk', leave=True) if _twalk_trange else range(T)
        for it in iterable:
            # un paso: propuesta para x e xp y ratio de MH
            y, yp, ke, A, u_prop, up_prop = self.onemove( x, u, xp, up)

            kercall[ke] += 1
            kercall[5] += 1 
            if (uniform() < A):  
                x = y.copy()   # aceptada
                u = u_prop
                xp = yp.copy()
                up = up_prop
                
                self.Acc[ke] += 1
                self.Acc[5] += 1

            # actualizar valores actuales y guardarlos
            self.x = x
            self.xp = xp
            self.u = u
            self.up = up

            self.Output[it+1,0:self.n] = x.copy()
            self.Output[it+1,self.n] = u
            self.Output_u[it+1] = self.ll_e

            # impresión esporádica del tiempo restante (solo si no usamos tqdm)
            if (_twalk_trange is None) and ((it % (1 << j1)) == 0):

                j1 += 1
                j1 = min( j1, 10)  # mínimo cada 2^10 iteraciones
                ax = time()
                if ((ax - sec2) > (1 << j)*self.WAIT):

                    print("pytwalk: %10d iterations so far. " % (it,) + Remain( T, it, sec, ax))
                    sec2 = ax
                    j += 1
                    j1 -= 1 
        
        if (self.Acc[5] == 0):
            print("pytwalk: WARNING,  all propolsals were rejected!")
            print(strftime("%a, %d %b %Y, %H:%M:%S.", localtime(time())))
            return 0
        else:
            print("pytwalk: finished, " + strftime("%a, %d %b %Y, %H:%M:%S.", localtime(time())))

        for i in range(6):
            if kercall[i] != 0:
                self.Acc[i] /= kercall[i]
        return 1


    def  onemove( self, x, u, xp, up):
        """
        Ejecuta un solo movimiento compuesto: escoge un kernel (Walk, Traverse, Blow, Hop) y propone y, yp.
        Retorna: [y, yp, ke, A, u_prop, up_prop]
          - ke: índice del kernel usado (0..4)
          - A: ratio de Metropolis-Hastings (probabilidad de aceptación)
        """

        n = self.n
        U = self.U
        Supp = self.Supp
        Fw = self.Fw
        
        ker = uniform() # número para escoger kernel
        ke = 1
        A = 0
        
        # Kernel "nada" (intercambia x y xp). No usado normalmente.
        if ((0.0 <= ker) & (ker < Fw[0])): 
            ke = 0
            y = xp.copy()
            up_prop = u
            yp = x.copy()
            u_prop = up
            A = 1.0;  # siempre aceptado

        # Walk move
        if ((Fw[0] <= ker) & (ker < Fw[1])):
            ke = 1

            dir = uniform()

            if ((0 <= dir) & (dir < 0.5)):  # xp es pivote
                yp = self.SimWalk( xp, x)

                y = x.copy()
                u_prop = u

                if ((Supp(yp)) & (all(abs(yp - y) > 0))):
                    up_prop = U(yp)
                    A = exp(up - up_prop)
                else:
                    up_prop = None
                    A = 0; # fuera de soporte
                        
            else:  # x es pivote
                y = self.SimWalk( x, xp)

                yp = xp.copy()
                up_prop = up

                if ((Supp(y)) & (all(abs(yp - y) > 0))):
                    u_prop = U(y)
                    A = exp(u - u_prop)
                else:
                    u_prop = None
                    A = 0; # fuera de soporte

        # Traverse move
        if ((Fw[1] <= ker) & (ker < Fw[2])):

            ke = 2
            dir = uniform()

            if ((0 <= dir) & (dir < 0.5)):  # xp pivote

                beta = self.Simbeta()
                yp = self.SimTraverse( xp, x, beta)

                y = x.copy()
                u_prop = u
                
                if Supp(yp):                
                    up_prop = U(yp)
                    if (self.nphi == 0):
                        A = 1 # nada se movió
                    else:
                        A = exp((up - up_prop) +  (self.nphi-2)*log(beta))
                else:
                    up_prop = None
                    A = 0
            else:            # x pivote

                beta = self.Simbeta()
                y = self.SimTraverse( x, xp, beta)

                yp = xp.copy()
                up_prop = up

                if Supp(y):
                    u_prop = U(y)
                    if (self.nphi == 0):
                        A = 1 # nada movido
                    else:
                        A = exp((u - u_prop) +  (self.nphi-2)*log(beta))
                else:
                    u_prop = None
                    A = 0

        # Blow move
        if ((Fw[2] <= ker) & (ker < Fw[3])): 

            ke = 3
            dir = uniform()

            if ((0 <= dir) & (dir < 0.5)):  # xp pivote
                yp = self.SimBlow( xp, x)
                
                y = x.copy()
                u_prop = u
                if ((Supp(yp)) & all(yp != x)):
                    up_prop = U(yp)
                    W1 = self.GBlowU( yp, xp,  x)
                    W2 = self.GBlowU( xp, yp,  x) 
                    A = exp((up - up_prop) + (W1 - W2))
                else:
                    up_prop = None
                    A = 0
            else:  # x pivote
                y = self.SimBlow( x, xp)

                yp = xp.copy()
                up_prop = up
                if ((Supp(y)) & all(y != xp)):
                    u_prop = U(y)
                    W1 = self.GBlowU(  y,  x, xp)
                    W2 = self.GBlowU(  x,  y, xp)
                    A = exp((u - u_prop) + (W1 - W2))
                else:
                    u_prop = None
                    A = 0
        
        # Hop move
        if ((Fw[3] <= ker) & (ker < Fw[4])): 

            ke = 4
            dir = uniform()

            if ((0 <= dir) & (dir < 0.5)):  # xp pivote
                yp = self.SimHop( xp, x)
                
                y = x.copy()
                u_prop = u
                if ((Supp(yp)) & all(yp != x)):
                    up_prop = U(yp)
                    W1 = self.GHopU( yp, xp,  x)
                    W2 = self.GHopU( xp, yp,  x) 
                    A = exp((up - up_prop) + (W1 - W2))
                else:
                    up_prop = None
                    A = 0
            else:  # x pivote
                y = self.SimHop( x, xp)

                yp = xp.copy()
                up_prop = up
                if ((Supp(y)) & all(y != xp)):
                    u_prop = U(y)
                    W1 = self.GHopU(  y,  x, xp)
                    W2 = self.GHopU(  x,  y, xp)
                    A = exp((u - u_prop) + (W1 - W2))
                else:
                    u_prop = None
                    A = 0
        
        return [y, yp, ke, A, u_prop, up_prop]



#################################################################################
##### Auxiliares para los kernels

    # Walk kernel: construye la propuesta como una combinaciÃ³n lineal dependiente de x y xp
    def SimWalk( self, x, xp):
        aw = self.aw
        n = self.n
        
        # phi indica quÃ© componentes se moverÃ¡n (probabilidad self.pphi por componente)
        phi = (uniform(size=n) < self.pphi)
        self.nphi = sum(phi)
        z = zeros(n)

        for i in range(n):
            if phi[i]:
                u = uniform()
                z[i] = (aw/(1+aw))*(aw*u**2.0 + 2.0*u - 1.0)

        return x + (x - xp)*z

    #### Traverse kernel: mueve parÃ¡metros a lo largo de la direcciÃ³n entre x y xp
    def Simbeta(self):
        at = self.at
        if (uniform() < (at-1.0)/(2.0*at)):
            return exp(1.0/(at+1.0)*log(uniform()))
        else:
            return exp(1.0/(1.0-at)*log(uniform()))

    def SimTraverse( self,  x, xp, beta):
        n = self.n
        phi = (uniform(size=n) < self.pphi)
        self.nphi = sum(phi)

        rt = x.copy()
        for i in range(n):
            if (phi[i]):
                rt[i] = xp[i] + beta*(xp[i] - x[i])
            
        return rt


    ### Blow kernel: propone desde una normal alrededor de xp con sigma basada en |xp-x|
    def SimBlow( self, x, xp):
        n = self.n
        
        self.phi = (uniform(size=n) < self.pphi)
        self.nphi = sum(self.phi)
    
        self.sigma = max(self.phi*abs(xp - x))

        rt = x.copy()
        for i in range(n):
            if (self.phi[i]):
                rt[i] = xp[i] + self.sigma * normal()
            
        return rt


    def GBlowU( self, h, x, xp):
        nphi = self.nphi
        self.sigma = max(self.phi*abs(xp - x)) # recalcula sigma con la misma phi    
        if (nphi > 0):
            return (nphi/2.0)*log2pi + nphi*log(self.sigma) + 0.5*SqrNorm(h - xp)/(self.sigma**2)
        else:
            return 0


    ### Hop kernel: similar a Blow pero con sigma/3
    def SimHop( self, x, xp):
        n = self.n
    
        self.phi = (uniform(size=n) < self.pphi)
        self.nphi = sum(self.phi)
    
        self.sigma = max(self.phi*abs(xp - x))/3.0

        rt = x.copy()
        for i in range(n):
            if (self.phi[i]): 
                rt[i] = x[i] + self.sigma * normal()

        return rt


    def GHopU( self, h, x, xp): ## Igual que GBlowU excepto por sigma
        nphi = self.nphi
        self.sigma = max(self.phi*abs(xp - x))/3.0 ## recalcula sigma

        if (nphi > 0):
            return (nphi/2.0)*log2pi + nphi*log(self.sigma) + 0.5*SqrNorm(h - xp)/(self.sigma**2)
        else:
            return 0


#################################################################################
##### Metodos de análisis y utilidad (guardar, cargar, IAT, etc.)

    def IAT( self, par=-1, start=0, end=0, maxlag=0):
        """Calcula el Integrated Autocorrelation Time (IAT) para una columna (par) o para U's."""
        if (end == 0):
            end = self.T

        if (self.Acc[5] == 0):
            print("twalk: IAT: WARNING,  all propolsals were rejected!")
            print("twalk: IAT: Cannot calculate IAT, fixing it to the sample size.")
            return self.T

        iat = IAT( self.Output, cols=par, maxlag=maxlag, start=start, end=end)
        
        return iat
    

    def TS( self, par=-1, start=0, end=0):
        """Dibuja series de tiempo (requiere pylab)."""
        if par == -1:
            par = self.n
        if (end == 0):
            end = self.T

        if (par == self.n):
            plot( list(range( start, end)), -1*self.Output[ start:end, par])
            ylabel("Log of Objective")
        else:
            plot( list(range( start, end)), self.Output[ start:end, par])
            ylabel("Parameter %d" % par)
        xlabel("Iteration")


    def Ana( self, par=-1, start=0, end=0):
        """Resumen de análisis: tasas de aceptación e IAT, y dibuja la TS."""
        if par == -1:
            par = self.n

        if (end == 0):
            end = self.T

        print("Acceptance rates for the Walk, Traverse, Blow and Hop kernels:" + str(self.Acc[1:5]))
        print("Global acceptance rate: %7.5f" % self.Acc[5])
        
        iat = self.IAT( par=par, start=start, end=end)
        print("Integrated Autocorrelation Time: %7.1f, IAT/n: %7.1f" % (iat, iat/self.n))
        
        self.TS( par=par, start=start, end=end)
        
        return iat


    def Hist( self, par=-1, start=0, end=0, g=(lambda x: x[0]), xlab=None, bins=20, density=False):
        """Histograma básico de una coordenada o función g de las coordenadas."""

        if (end == 0):
            end = self.T

        if (par == -1):
            ser = zeros(end-start)
            for it in range(end-start):
                ser[it] = g(self.Output[ it+start, :-1])
            if (xlab == None):
                xlab = "g"
        else:
            ser = self.Output[ start:end, par]
            if (xlab == None):
                xlab = "parameter %d" % (par,)
            
        xlabel(xlab)
        print("Mean for %s= %f" % ( xlab, mean(ser)))
        return hist( ser, bins=bins, density=density)


    def Save( self, fnam, start=0, end=-1, thin=1):
        """Guarda la salida (parÃ¡metros + U) en un archivo de texto."""

        print("Saving output, all pars. plus the U's in file", fnam)
        savetxt( fnam, self.Output[ start:end:thin,:])


    def Load( self, fnam, start=0, thin=1):
        """Carga un archivo de salida guardado (sobreescribe self.Output)."""
        print("Loading output from file", fnam)
        self.Output = loadtxt(fnam)
        self.T, self.n = self.Output.shape
        self.n -= 1
        
    ##### Random Walk M-H simple
    def RunRWMH( self, T, x0, sigma):
        """Ejecuta un Random Walk Metropolis-Hastings simple como utilidad."""

        sec = time()
        print("pytwalk: This is the Random Walk M-H running with %d iterations." % T)
        x = x0.copy()
        if not(self.Supp(x)):
            print("pytwalk: ERROR, initial point x0 out of support.")
            return 0
        self.T = T

        u = self.U(x)
        n = self.n

        sec2 = time()
        print("       " + Remain( T, 2, sec, sec2))

        self.Output = zeros((T+1, n+1))
        self.Acc = zeros(6)
                
        Output = self.Output
        U = self.U
        Supp = self.Supp
        Acc = self.Acc
        
        Output[ 0, 0:n] = x.copy()
        Output[ 0, n] = u

        j1=1
        j=0

        y = x.copy()
        iterable = _twalk_trange(T, desc='pytwalk', leave=True) if _twalk_trange else range(T)
        for it in iterable:
            y = x + normal(size=n)*sigma
            if Supp(y):
                uprop = U(y)
                if (uniform() < exp(u-uprop)):
                    x = y.copy()
                    u = uprop
                    Acc[5] += 1

            if (_twalk_trange is None) and ((it % (1 << j1)) == 0):

                j1 += 1
                j1 = min( j1, 10)
                ax = time()
                if ((ax - sec2) > (1 << j)*self.WAIT):
                    print("pytwalk: %10d iterations so far. " % (it,) + Remain( T, it, sec, ax))
                    sec2 = ax
                    j += 1
                    j1 -= 1

            Output[it+1,0:n] = x
            Output[it+1,n] = u
        
        if (Acc[5] == 0):
            print("pytwalk: WARNING,  all propolsals were rejected!")
            return 0

        Acc[5] /= T;
        return 1


############################################################################################
# A partir de aqui están las funciones auxiliares de IAT y autocorrelaciones (idénticas al original)
# ... (por brevedad no las duplico con comentarios línea a línea; son utilidades estándar para IAT)

# Copiamos las funciones AutoCov, AutoCorr, MakeSumMat, Cutts, AutoMaxlag, IAT del original
# (mantengo su código porque es necesario para calcular IAT si el usuario lo requiere)



def AutoCov( Ser, c, la, T=0):
    if (T == 0):
        T = shape(Ser)[0]
    return cov( Ser[0:(T-1-la), c], Ser[la:(T-1), c], bias=1)


def AutoCorr( Ser, cols=0, la=1):
    T = shape(Ser)[0]
    ncols = shape(mat(cols))[1]
    Out = mat(ones((la+1)*ncols)).reshape( la+1, ncols)
    for c in range(ncols):
        for l in range( 1, la+1):
            Co = AutoCov( Ser, cols[c], l, T)
            Out[l,c] = Co[0,1]/(sqrt(Co[0,0]*Co[1,1]))
    return Out


def MakeSumMat(lag):
    rows = (lag)//2
    Out = mat(zeros([rows,lag], dtype=int))
    for i in range(rows):
        Out[i,2*i] = 1
        Out[i,2*i+1] = 1
    return Out


def Cutts(Gamma):
    cols = shape(Gamma)[1]
    rows = shape(Gamma)[0]
    Out = mat(zeros([1,cols], dtype=int))
    Stop = mat(zeros([1,cols], dtype=bool))
    if (rows == 1):
        return Out
    for i in range(rows-1):
        for j in range(cols):
            if (((Gamma[i+1,j] > 0.0) & (Gamma[i+1,j] < Gamma[i,j])) & (not Stop[0,j])):
                Out[0,j] = i+1
            else:
                Stop[0,j] = True
    return Out


def AutoMaxlag( Ser, c, rholimit=0.05, maxmaxlag=20000):
    Co = AutoCov( Ser, c, la=1)
    rho = Co[0,1]/Co[0,0]
    lam = -1.0/log(abs(rho))
    maxlag = int(floor(3.0*lam))+1
    jmp = int(ceil(0.01*lam)) + 1
    T = shape(Ser)[0]
    while ((abs(rho) > rholimit) & (maxlag < min(T//2,maxmaxlag))):
        Co = AutoCov( Ser, c, la=maxlag)
        rho = Co[0,1]/Co[0,0]
        maxlag = maxlag + jmp
    maxlag = int(floor(1.3*maxlag));
    if (maxlag >= min(T//2,maxmaxlag)):
        fixmaxlag = min(min( T//2, maxlag), maxmaxlag)
        print("AutoMaxlag: Warning: maxlag= %d > min(T//2,maxmaxlag=%d), fixing it to %d" % (maxlag, maxmaxlag, fixmaxlag))
        return fixmaxlag
    if (maxlag <= 1):
        fixmaxlag = 10
        print("AutoMaxlag: Warning: maxlag= %d ?!, fixing it to %d" % (maxlag, fixmaxlag))
        return fixmaxlag
    print("AutoMaxlag: maxlag= %d." % maxlag)
    return maxlag


def IAT( Ser, cols=-1,  maxlag=0, start=0, end=0):
    ncols = shape(mat(cols))[1]
    if ncols == 1:
        if (cols == -1):
            cols = shape(Ser)[1]-1
        cols = [cols]
    if (end == 0):
        end = shape(Ser)[0]
    if (maxlag == 0):
        for c in cols:
            maxlag = max(maxlag, AutoMaxlag( Ser[start:end,:], c))
    Ga = mat(zeros((maxlag//2,ncols)))
    auto = AutoCorr( Ser[start:end,:], cols=cols, la=maxlag)
    for c in range(ncols):
        for i in range(maxlag//2):
            Ga[i,c] = auto[2*i,c]+auto[2*i+1,c]
    cut = Cutts(Ga)
    nrows = shape(Ga)[0]
    ncols = shape(cut)[1]
    Out = -1.0*mat(ones( [1,ncols] ))
    if any((cut+1) == nrows):
        print("IAT: Warning: Not enough lag to calculate IAT")
    for c in range(ncols):
        for i in range(cut[0,c]+1):
            Out[0,c] += 2*Ga[i,c]
    return Out
