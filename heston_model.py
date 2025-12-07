import numpy as np
from scipy.interpolate import interp1d


class HestonModel:
    """Class representing the Heston model for stochastic volatility."""

    def __init__(self, params):

        self.params = params
        self.kappa = params['kappa']  # Mean reversion speed
        self.theta = params['theta']  # Long-term variance
        self.xi = params['xi']        # Volatility of volatility
        self.rho = params['rho']      # Correlation between asset and volatility
        self.v0 = params['v0']        # Initial variance




    def simulate(self, S0, T, r, q, npaths=10**5, nsteps=252, seed=None):
        """
        Simulate asset paths under the Heston model using full truncation Euler and log-spot.

        Parameters
        ----------
        S0 : float
            Initial spot price.
        T : float
            Time to maturity (in years).
        r : float
            Risk-free rate.
        q : float
            Dividend yield.
        npaths : int, optional
            Number of Monte Carlo paths. Default 100000.
        nsteps : int, optional
            Time steps per year. Default 252.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        S : ndarray
            Simulated asset paths of shape (steps+1, npaths).
        nu : ndarray
            Simulated variance paths of shape (steps+1, npaths).
        """

        if seed is not None:
            np.random.seed(seed)

        steps = round(nsteps * T)
        dt = T/steps

        # Initialize arrays
        nu = np.zeros((steps + 1, npaths))
        S = np.zeros((steps + 1, npaths))
        nu[0] = self.v0
        S[0] = S0

        for t in range(1, steps + 1):
            # Correlated Brownian increments
            dW_S = np.random.normal(0, np.sqrt(dt), npaths)
            Z = np.random.normal(0, np.sqrt(dt), npaths)
            dW_nu = self.rho * dW_S + np.sqrt(1 - self.rho**2) * Z

            # Full truncation Euler scheme for variance
            nu_prev = np.maximum(nu[t-1], 0)
            nu[t] = nu[t-1] + self.kappa * (self.theta - nu_prev) * dt + self.xi * np.sqrt(nu_prev) * dW_nu
            nu[t] = np.maximum(nu[t], 0)

            # Log-spot scheme for asset
            S[t] = S[t-1] * np.exp((r - q - 0.5 * nu_prev) * dt + np.sqrt(nu_prev) * dW_S)

        return S, nu

    
    def heston_cf(self, u, T, S0, r, q):
        """
        Characteristic function φ(u) = E[exp(i u ln S_T)] under the risk-neutral measure.

        params
        ----------
        u : float or array_like
            Argument of the characteristic function (real or complex).
            Can be scalar or numpy array.
        T : float
            Time to maturity.
        S0 : float
            Initial spot price.
        r : float
            Risk-free rate.
        q : float
            Continuous dividend yield.
        p : params
            Model params: kappa, theta, xi, rho, v0.

        Returns
        -------
        phi : complex or ndarray of complex
            Value of the characteristic function at u.

        Notes
        -----
        - Uses the Little Heston Trap parametrization for numerical stability.
        - Handles scalar or vector inputs.
        - Guards are added to avoid division by zero in denominators.
        """

        u = np.atleast_1d(u)
        i = 1j
        x0 = np.log(S0)

        a = self.kappa *self.theta
        b = self.kappa -self.rho *self.xi * i * u
        d = np.sqrt(b * b + (self.xi ** 2) * (i * u + u * u))
        g = (b - d) / (b + d)

        df = np.exp(-d * T)
        g1 = np.clip(1 - g * df, 1e-15, None)
        g2 = np.clip(1 - g, 1e-15, None)

        C = (i * u * (r - q) * T + (a / (self.xi ** 2)) * ((b - d) * T - 2.0 * np.log(g1 / g2)))
        D = ((b - d) / (self.xi ** 2)) * ((1 - df) / g1)

        phi = np.exp(C + D * self.v0 + i * u * x0)
        return phi[0] if phi.size == 1 else phi



    def call_transform(self, v, T, S0, r, q, alpha=1.5):
        """
        Fourier transform of the damped call price, as in Carr-Madan (1999).

        Parameters
        ----------
        v : float or array_like
            Frequency variable in Fourier space.
        T : float
            Time to maturity.
        S0 : float
            Spot price.
        r : float
            Risk-free rate.
        q : float
            Dividend yield (or foreign rate in FX).
        alpha : float, optional
            Damping parameter (>0), default is 1.5.

        Returns
        -------
        psi : complex or ndarray of complex
            Value(s) of the Fourier transform of the damped call.
        """
        v = np.atleast_1d(v)
        i = 1j

        phi = self.heston_cf(v - (alpha + 1) * i, T, S0, r, q)
        numerator = np.exp(-r * T) * phi
        denominator = alpha**2 + alpha - v**2 + i * (2 * alpha + 1) * v

        psi = numerator / denominator
        return psi
    

    def carr_madan_call(self, T, S0, r, q, K, alpha=1.5, N=4096, eta=0.25):
        """
        Price European call options using Carr-Madan FFT with interpolation for arbitrary strikes K.

        Parameters
        ----------
        T : float
            Time to maturity.
        S0 : float
            Spot price.
        r : float
            Risk-free rate.
        q : float
            Dividend yield.
        K : float or ndarray
            Strike(s) for which to return prices.
        alpha : float, optional
            Damping factor (>0). Default is 1.5.
        N : int, optional
            Number of FFT points (power of 2). Default 4096.
        eta : float, optional
            Spacing in Fourier domain. Default 0.25.

        Returns
        -------
        call_prices : ndarray
            Call prices corresponding to strikes K.
        """

        # Call transform
        v = np.arange(N) * eta
        psi = self.call_transform(v, T, S0, r, q, alpha=alpha)

        # Trapezoidal weights
        w = eta * np.ones(N)
        w[0] = w[-1] = eta * 0.5

        # FFT centering
        lam = 2.0 * np.pi / (N * eta)
        b = 0.5 * N * lam
        x = psi * np.exp(1j * b * v) * w
        fft_vals = np.fft.fft(x).real

        # Log-strike grid and strikes
        k_grid = -b + lam * np.arange(N)
        K_grid = np.exp(k_grid)

        # Call prices
        call_prices_grid = np.exp(-alpha * k_grid) / np.pi * fft_vals
        interpolator = interp1d(K_grid, call_prices_grid, kind='cubic', fill_value="extrapolate")
        call_prices = interpolator(np.atleast_1d(K))

        return call_prices


    def heston_call(self, T, S0, r, q, K, N=1000, U_max=100):
        """
        European call price under Heston using P1, P2 integrals.

        Parameters
        ----------
        T : float
            Time to maturity.
        S0 : float
            Spot price.
        r : float
            Risk-free rate.
        q : float
            Dividend yield.
        K : ndarray
            Strike array.
        N : int
            Number of integration points (trapezoidal).
        U_max : float
            Upper limit for integration.

        Returns
        -------
        call_prices : ndarray
            Call prices corresponding to K.
        """

        i = 1j
        K = np.atleast_1d(K)
        logK = np.log(K)[:, None] 
        call_prices = np.zeros_like(K, dtype=float)

        # Characteristic functions 
        u = np.linspace(1e-10, U_max, N)  
        phi1 = self.heston_cf(u - i, T, S0, r, q) 
        phi2 = self.heston_cf(u, T, S0, r, q)      
        phi3 = self.heston_cf(-i, T, S0, r, q)

        # Integrands 
        int1 = np.real(np.exp(-i * u * logK) * phi1 / (i * u * phi3))  
        int2 = np.real(np.exp(-i * u * logK) * phi2 / (i * u))

        # Trapezoidal integration over u
        P1 = 0.5 + (1/np.pi) * np.trapz(int1, u, axis=1)
        P2 = 0.5 + (1/np.pi) * np.trapz(int2, u, axis=1)

        # Call prices
        call_prices = S0 * np.exp(-q*T) * P1 - K * np.exp(-r*T) * P2

        return call_prices
    
    
    def monte_carlo_call(self, T, S0, r, q, K, npaths=10**5, nsteps=252, seed=None):
        """
        Price European call options using Monte Carlo simulation under the Heston model.

        Parameters
        ----------
        S0 : float
            Initial spot price.
        T : float
            Time to maturity (in years).
        r : float
            Risk-free rate.
        q : float
            Dividend yield.
        K : float or ndarray
            Strike(s) for which to return prices.
        npaths : int, optional
            Number of Monte Carlo paths. Default 100000.
        nsteps : int, optional
            Time steps per year. Default 252.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        call_prices : ndarray
            Call prices corresponding to strikes K.
        """

        K = np.atleast_1d(K)
        S, _ = self.simulate(S0, T, r, q, npaths=npaths, nsteps=nsteps, seed=seed)
        S_T = S[-1, :]

        call_prices = np.exp(-r * T) * np.maximum(S_T[:, None] - K[None, :], 0).mean(axis=0)
        return call_prices 
