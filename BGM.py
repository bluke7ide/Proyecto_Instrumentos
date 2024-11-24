import numpy as np
from scipy.stats import norm
from scipy.linalg import cholesky, solve

def simulate_terminal(L, constlam, N, delta, h, idum):
    """
    Simula los valores terminales para una matriz L basada en parámetros.

    Parámetros:
    L (ndarray): Matriz bidimensional que representa las tasas forward.
    constlam (float): Parámetro constante lambda.
    N (int): Número de pasos.
    delta (float): Incremento temporal.
    h (float): Tamaño del paso temporal.
    idum (np.random.Generator): Generador de números aleatorios para reproducibilidad.

    Devuelve:
    None: Actualiza la matriz L en su lugar.
    """
    def gasdev(rng):
        """Genera una variable aleatoria gaussiana estándar."""
        return rng.standard_normal()

    for j in range(N):
        z = gasdev(idum)  # Generar variable aleatoria normal estándar
        # Desplazar los valores de L[i][j] a L[i][j+1]
        for i in range(j + 1):
            L[i, j + 1] = L[i, j]
        for i in range(j + 1, N + 1):
            mu = 0.0
            for k in range(i + 1, N + 1):
                # Calcular el valor esperado mu
                mu -= (delta * constlam**2 * L[k, j]) / (1.0 + delta * L[k, j])
            # Actualizar L[i][j+1] usando la fórmula
            L[i, j + 1] = L[i, j] * np.exp((mu - 0.5 * constlam**2) * h + constlam * np.sqrt(h) * z)

def cblack(sig, k, r, b, T, delta):
    """
    Fórmula de Black para el precio de un caplet.

    Parámetros:
    sig (float): Volatilidad.
    k (float): Tasa de ejercicio (strike).
    r (float): Tasa forward.
    b (float): Factor de descuento.
    T (float): Tiempo hasta el vencimiento.
    delta (float): Incremento temporal.

    Devuelve:
    float: Precio del caplet.
    """
    d1 = (np.log(r / k) + 0.5 * sig**2 * T) / (sig * np.sqrt(T))
    d2 = (np.log(r / k) - 0.5 * sig**2 * T) / (sig * np.sqrt(T))
    return delta * b * (r * norm.cdf(d1) - k * norm.cdf(d2))

def exact_caplet(caplet, L, constlam, N, T, K, delta):
    """
    Calcula el precio exacto de un caplet usando la fórmula de Black.

    Parámetros:
    caplet (ndarray): Array para almacenar los precios del caplet.
    L (ndarray): Matriz de tasas forward.
    constlam (float): Lambda constante.
    N (int): Número de pasos temporales.
    T (ndarray): Array de tiempos hasta el vencimiento.
    K (ndarray): Array de tasas de ejercicio (strike rates).
    delta (float): Incremento temporal.

    Devuelve:
    None: Actualiza el array `caplet` en su lugar.
    """
    P = np.ones(N + 2)  # Inicializar factores de descuento
    for n in range(1, N + 2):
        for j in range(n):
            P[n] *= 1.0 / (1.0 + delta * L[j, 0])  # Calcular el factor de descuento
    for n in range(1, N + 1):
        # Calcular el precio del caplet usando la fórmula de Black
        caplet[n] = cblack(constlam, K[n], L[n, 0], P[n + 1], T[n], delta)

def simulate_caplet(caplet, L, constlam, N, T, K, delta, h, niter):
    """
    Simula los precios de caplets utilizando una simulación Monte Carlo.

    Parámetros:
    caplet (ndarray): Array para almacenar los precios simulados de los caplets.
    L (ndarray): Matriz de tasas forward.
    constlam (float): Lambda constante.
    N (int): Número de pasos temporales.
    T (ndarray): Array de tiempos hasta el vencimiento.
    K (ndarray): Array de tasas de ejercicio (strike rates).
    delta (float): Incremento temporal.
    h (float): Tamaño del paso temporal para la simulación.
    niter (int): Número de iteraciones para la simulación Monte Carlo.

    Devuelve:
    None: Actualiza el array `caplet` en su lugar.
    """
    num = 1.0
    # Calcular el factor inicial de descuento
    for j in range(N + 1):
        num *= 1.0 / (1.0 + delta * L[j, 0])
    caplet.fill(0.0)  # Inicializar los precios del caplet a cero

    # Iteraciones Monte Carlo
    for _ in range(1, niter + 1):
        # Simular las tasas forward hasta el tiempo terminal
        simulate_terminal(L, constlam, N, delta, h, -_)
        for n in range(1, N):
            den = 1.0
            # Calcular el factor de descuento para el denominador
            for j in range(n + 1, N + 1):
                den *= 1.0 / (1.0 + delta * L[j, n + 1])
            # Actualizar el precio del caplet
            caplet[n] += (num * delta * max(L[n, n] - K[n], 0) / den) / niter
        # Precio para el último caplet
        caplet[N] += num * delta * max(L[N, N] - K[N], 0) / niter

def european_swaption(alpha, L, constlam, N, T, K, delta, h, niter):
    """
    Calcula el precio de una swaption europea usando simulación Monte Carlo.

    Parámetros:
    alpha (int): Índice inicial de la swaption.
    L (ndarray): Matriz de tasas forward.
    constlam (float): Lambda constante.
    N (int): Número de pasos temporales.
    T (ndarray): Array de tiempos hasta el vencimiento.
    K (float): Tasa de ejercicio de la swaption.
    delta (float): Incremento temporal.
    h (float): Tamaño del paso temporal para la simulación.
    niter (int): Número de iteraciones para la simulación Monte Carlo.

    Devuelve:
    float: Precio estimado de la swaption.
    """
    def pos(x):
        """Devuelve la parte positiva de x."""
        return max(x, 0)

    num = 1.0
    for j in range(N + 1):
        num *= 1.0 / (1.0 + delta * L[j, 0])
    
    avg = 0.0
    for iter_ in range(1, niter + 1):
        # Simula las tasas forward terminales
        simulate_terminal(L, constlam, N, delta, h, -iter_)
        
        den = 1.0
        for j in range(alpha + 1, N + 1):
            den *= 1.0 / (1.0 + delta * L[j, alpha + 1])
        
        swaption = 0.0
        for i in range(alpha + 1, N + 2):
            Bi = 1.0
            for j in range(alpha, i):
                Bi *= 1.0 / (1.0 + delta * L[j, alpha])
            swaption += Bi * delta * (L[i - 1, alpha] - K)
        
        ans = num * pos(swaption) / den
        avg += ans / niter
    
    return avg

def S(alpha, beta, P, N, delta):
    """
    Calcula el valor S utilizado en el precio de una Bermudan swaption.

    Parámetros:
    alpha (int): Índice inicial.
    beta (int): Índice final.
    P (ndarray): Factores de descuento.
    N (int): Número de pasos.
    delta (float): Incremento temporal.

    Devuelve:
    float: Valor S calculado.
    """
    den = sum(delta * P[i] for i in range(alpha + 1, beta + 1))
    return (P[alpha] - P[beta]) / den

def condexp(coef, order, x):
    """
    Calcula la expectativa condicional utilizando los coeficientes de regresión.

    Parámetros:
    coef (ndarray): Coeficientes de la regresión.
    order (int): Orden de la regresión polinómica.
    x (float): Variable independiente.

    Devuelve:
    float: Expectativa condicional calculada.
    """
    return sum(coef[io] * x**io for io in range(order + 1))

def regress(x, y, n, order):
    """
    Realiza una regresión para determinar los coeficientes de un polinomio de un orden dado.

    Parámetros:
    x (ndarray): Datos de la variable independiente.
    y (ndarray): Datos de la variable dependiente.
    n (int): Número de puntos de datos.
    order (int): Orden del polinomio.

    Devuelve:
    ndarray: Coeficientes de la regresión.
    """
    if order == 1:
        sx = np.sum(x)
        sy = np.sum(y)
        sxy = np.sum(x * y)
        sx2 = np.sum(x**2)
        coef = np.zeros(2)
        coef[1] = (sxy - sx * sy / n) / (sx2 - (sx**2) / n)
        coef[0] = (sy - coef[1] * sx) / n
    elif order == 2:
        sx = np.sum(x)
        sx2 = np.sum(x**2)
        sx3 = np.sum(x**3)
        sx4 = np.sum(x**4)
        sy = np.sum(y)
        sxy = np.sum(x * y)
        sx2y = np.sum(x**2 * y)
        
        a = np.array([
            [n, sx, sx2],
            [sx, sx2, sx3],
            [sx2, sx3, sx4]
        ])
        b = np.array([sy, sxy, sx2y])
        
        # Resolver utilizando descomposición de Cholesky
        L = cholesky(a, lower=True)
        coef = solve(a, b)
    else:
        raise ValueError("Solo se admiten órdenes 1 y 2.")
    
    return coef

def bermudan_swaption(alpha, L, constlam, N, T, K, delta, h, nit, payrec):
    """
    Calcula el precio de una Bermudan swaption utilizando simulación Monte Carlo con regresión.
    """
    def pos(x):
        """Devuelve la parte positiva de x."""
        return max(x, 0)
    
    PAYFIXED = 1  # Constante para indicar una payer swaption
    s = np.zeros((nit, N + 1))
    e = np.zeros((nit, N + 1))
    numer = np.zeros((nit, N + 1))
    cf = np.zeros((nit, N + 1))
    itmflag = np.zeros((nit, N + 1), dtype=int)
    eflag = np.zeros((nit, N + 1), dtype=int)
    
    # Inicializar generador de números aleatorios
    rng = np.random.default_rng(seed=42)

    # Simulaciones Monte Carlo
    for it in range(nit):
        simulate_terminal(L, constlam, N, delta, h, rng)  # Cambiar `-it` por `rng`
        for i in range(alpha, N + 1):
            B = np.ones(N + 2)
            for n in range(i + 1, N + 2):
                for j in range(i, n):
                    B[n] *= 1.0 / (1.0 + delta * L[j, i])
            s[it, i] = S(i, N + 1, B, N, delta)
            e[it, i] = 0.0
            for n in range(i + 1, N + 2):
                if payrec == PAYFIXED:
                    e[it, i] += B[n] * delta * (L[n - 1, i] - K)
                else:
                    e[it, i] += B[n] * delta * (K - L[n - 1, i])
            e[it, i] = pos(e[it, i])
            numer[it, i] = np.prod([1.0 / (1.0 + delta * L[n, i]) for n in range(i, N + 1)])

    # Inducción hacia atrás
    for n in range(N - 1, alpha - 1, -1):
        nitm = np.sum(e[:, n] > 0.0)
        x = np.zeros(nitm)
        y = np.zeros(nitm)
        iitm = 0
        for it in range(nit):
            if e[it, n] > 0.0:
                itmflag[it, n] = 1
                x[iitm] = s[it, n]
                y[iitm] = cf[it, n + 1] * numer[it, n] / numer[it, n + 1]
                iitm += 1
        coef = regress(x, y, nitm, 2)
        iitm = 0
        for it in range(nit):
            if itmflag[it, n]:
                if e[it, n] > condexp(coef, 2, x[iitm]):
                    eflag[it, n] = 1
                    eflag[it, n + 1:] = 0
                else:
                    eflag[it, n] = 0
                iitm += 1
            else:
                eflag[it, n] = 0
        for it in range(nit):
            if eflag[it, n]:
                cf[it, n] = e[it, n]
            else:
                cf[it, n] = cf[it, n + 1] * numer[it, n] / numer[it, n + 1]
    
    # Descuento final
    num = np.prod([1.0 / (1.0 + delta * L[j, 0]) for j in range(N + 1)])
    avg = 0.0
    for it in range(nit):
        disc = num / numer[it, alpha + 1]
        ans = cf[it, alpha] * disc
        avg += ans / nit
    
    return avg

if __name__ == "__main__":
    # Parámetros básicos
    N = 10  # Número de pasos temporales
<<<<<<< HEAD
    T = np.linspace(0.1, 2.0, N + 1)  # Tiempos hasta el vencimiento
    delta = 0.25  # Incremento temporal
    h = 0.1  # Tamaño del paso temporal
    niter = 1000  # Iteraciones Monte Carlo
    constlam = 0.2  # Lambda constante
    K = np.array([0.03] * (N + 1))  # Tasa de ejercicio para cada caplet
=======
    T = np.linspace(0, 2.0, N+1)  # Tiempos hasta el vencimiento
    delta = 0.5  # Incremento temporal
    h = 0.5  # Tamaño del paso temporal
    niter = 10000  # Iteraciones Monte Carlo
    constlam = 0.15  # Lambda constante
    K = 0.0506978  # Tasa de ejercicio de la swaption
>>>>>>> 6a9dda38c32076cc01bafeaea37701a6f348d2e1
    alpha = 2  # Índice inicial para la swaption
    payrec = 1  # Indicar Payer Swaption

    # Inicializar el generador de números aleatorios
    rng = np.random.default_rng(seed=42)

    # Inicializar matriz de tasas forward (L) con valores iniciales
    L = np.zeros((N + 1, N + 1))
    L[:, 0] = np.exp(-0.05*T)  # Tasa inicial del 5%

    # Calcular el precio de la Bermudan Swaption
    precio_swaption = bermudan_swaption(alpha, L, constlam, N, T, K[0], delta, h, niter, payrec)
    print(f"Precio estimado de la Bermudan Swaption: {precio_swaption:.6f}")

    # Calcular el precio exacto del caplet usando la fórmula de Black
    caplet_exact = np.zeros(N + 1)
    exact_caplet(caplet_exact, L, constlam, N, T, K, delta)
    for n in range(1, N + 1):
        print(f"Precio exacto del caplet {n}: {caplet_exact[n]:.6f}")
