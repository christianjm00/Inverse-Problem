Este código implementa un problema simple de asimilación de datos para la ecuación logística usando:

- Runge-Kutta 4 (RK4)
- método adjunto
- cálculo de gradiente
- descenso de gradiente

El objetivo es estimar la condición inicial X0 a partir de observaciones con ruido.


Dependencias

Instalar:

pip install numpy matplotlib


Ejecución

Ejecutar:

python3 nombre_archivo.py


Qué hace el código

1. Resuelve la ecuación logística.
2. Genera datos sintéticos con ruido.
3. Resuelve el problema adjunto.
4. Calcula el gradiente del funcional.
5. Valida el gradiente con diferencias finitas.
6. Optimiza X0.
7. Genera gráficas en PDF.


Parámetros importantes

En el bloque principal pueden modificarse:

X0_true
Valor real usado para generar datos.

X0
Valor inicial de la optimización.

X0b
Background o estimación previa.

B
Peso de regularización.

h
Paso temporal.
Valores pequeños aumentan precisión pero también tiempo de cómputo.

t_obs
Tiempos donde se toman observaciones.

r, K
Parámetros de la ecuación logística.


Optimización

La optimización se ejecuta con:

met_opt(fJ,fgJ,X0,1e-3,10000,1e-6)

Parámetros:

1e-3   -> paso de descenso
10000  -> iteraciones máximas
1e-6   -> tolerancia


Archivos generados

El código genera:

gra_x_lamb_B_*.pdf
gra_E1_E2_B_*.pdf
gra_J_*.pdf
gra_nablaJ_B_*.pdf
gra_X0_B_*.pdf


Notas

Si el método diverge:
- disminuir el paso tau.

Si se quiere mayor precisión:
- reducir h.

El gradiente adjunto se compara con diferencias finitas para validar la implementación.