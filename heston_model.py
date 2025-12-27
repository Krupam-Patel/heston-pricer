"""Heston stochastic volatility model"""
"""
Done
"""

import numpy as np
from scipy.interpolate import interp1d

class model:
    def __init__(self, params):
        self.params = params
        self.kappa = params['kappa']  # Mean-reversion speed of variance
        self.theta = params['theta']  # Long-run variance
        self.xi = params['xi']        # Volatility of variance (vol-of-vol)
        self.rho = params['rho']      # Corr(W_S, W_v)
        self.v0 = params['v0']        # Initial variance

    def simulate(self, S0, T, r, q, npaths=3**5, nsteps=365, seed=None):
        if seed is not None:
            np.random.seed(seed)

        steps = round(nsteps * T)
        dt = T / steps

        nu = np.zeros((steps + 1, npaths))
        S = np.zeros((steps + 1, npaths))
        nu[0] = self.v0
        S[0] = S0

        for t in range(1, steps + 1):
            dW_S = np.random.normal(0, np.sqrt(dt), npaths)
            Z = np.random.normal(0, np.sqrt(dt), npaths)
            dW_nu = self.rho * dW_S + np.sqrt(1 - self.rho**2) * Z
            
            nu_prev = np.maximum(nu[t - 1], 0)
            nu[t] = (
                nu[t - 1]
                + self.kappa * (self.theta - nu_prev) * dt
                + self.xi * np.sqrt(nu_prev) * dW_nu
            )
            nu[t] = np.maximum(nu[t], 0)

            # Log-spot update for asset price
            S[t] = S[t - 1] * np.exp(
                (r - q - 0.5 * nu_prev) * dt + np.sqrt(nu_prev) * dW_S
            )

        return S, nu
        
    def heston_cf(self, u, T, S0, r, q):
        u = np.atleast_1d(u)
        i = 1j
        x0 = np.log(S0)

        a = self.kappa * self.theta
        b = self.kappa - self.rho * self.xi * i * u
        d = np.sqrt(b * b + (self.xi ** 2) * (i * u + u * u))
        g = (b - d) / (b + d)

        df = np.exp(-d * T)
        g1 = np.clip(1 - g * df, 1e-15, None)
        g2 = np.clip(1 - g, 1e-15, None)

        C = (
            i * u * (r - q) * T
            + (a / (self.xi ** 2)) * ((b - d) * T - 2.0 * np.log(g1 / g2))
        )
        D = ((b - d) / (self.xi ** 2)) * ((1 - df) / g1)

        phi = np.exp(C + D * self.v0 + i * u * x0)
        return phi[0] if phi.size == 1 else phi

    def call_transform(self, v, T, S0, r, q, alpha=1.5):
        v = np.atleast_1d(v)
        i = 1j

        phi = self.heston_cf(v - (alpha + 1) * i, T, S0, r, q)
        numerator = np.exp(-r * T) * phi
        denominator = alpha**2 + alpha - v**2 + i * (2 * alpha + 1) * v

        psi = numerator/denominator
        return psi

    def carr_madan_call(self, T, S0, r, q, K, alpha=1.5, N=4096, eta=0.225):
        v = np.arange(N) * eta
        psi = self.call_transform(v, T, S0, r, q, alpha=alpha)

        w = eta * np.ones(N)
        w[0] = w[-1] = 0.5 * eta

        lam = 2.0 * np.pi / (N * eta)
        b = 0.5 * N * lam
        x = psi * np.exp(1j * b * v) * w
        fft_result = np.fft.fft(x)
        fft_vals = fft_result.real

        k_grid = -b + lam * np.arange(N)
        K_grid = np.exp(k_grid)

        call_prices_grid = np.exp(-alpha * k_grid) / np.pi * fft_vals
        get_price_at_strike = interp1d(
            K_grid, call_prices_grid, kind='cubic', fill_value="extrapolate"
        )
        call_prices = get_price_at_strike(np.atleast_1d(K))

        return call_prices

    def heston_call(self, T, S0, r, q, K, N=2000, U_max=175):
        i = 1j

        K = np.array(K, ndmin=1)
        logK = np.log(K).reshape(-1, 1)
        call_prices = np.zeros(len(K))

        u = np.linspace(1e-10, U_max, N)
        phi_shifted = self.heston_cf(u - i, T, S0, r, q)
        phi_base = self.heston_cf(u, T, S0, r, q)
        phi_const = self.heston_cf(-i, T, S0, r, q)

        exp_term = np.exp(-i * u * logK)

        int1 = np.real(exp_term * phi_shifted / (i * u * phi_const))
        int2 = np.real(exp_term * phi_base / (i * u))
        P1 = 0.5 + (1 / np.pi) * np.trapz(int1, u, axis=1)
        P2 = 0.5 + (1 / np.pi) * np.trapz(int2, u, axis=1)

        call_prices = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
        return call_prices

    def monte_carlo_call(self, T, S0, r, q, k, npaths=200000, nsteps=365, seed=None):
        k = np.array(k).reshape(-1)
        S, _ = self.simulate(S0, T, r, q, npaths=npaths, nsteps=nsteps, seed=seed)
        S_end = S[-1, :]

        S_col = S_end.reshape(-1, 1)
        k_row = k.reshape(1, -1)

        payoffs = np.maximum(S_col - k_row, 0)
        call_prices = np.exp(-r * T) * payoffs.mean(axis=0)  

        return call_prices
    
        
if __name__ == "__main__":
    print("HestonModel works")
