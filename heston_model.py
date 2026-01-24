"""Heston stochastic volatility model"""

import logging
from random import seed
import numpy as np
from scipy.interpolate import interp1d

from example import S0

logger = logging.getLogger(__name__)


class HestonModel:
    def __init__(self, params: dict):
        self.params = params
        self.kappa = params["kappa"]  # Mean-reversion speed of variance
        self.theta = params["theta"]  # Long-run variance
        self.xi = params["xi"]  # Volatility of variance (vol-of-vol)
        self.rho = params["rho"]  # Corr(W_S, W_v)
        self.v0 = params["v0"]  # Initial variance

        logger.info("Initialized HestonModel with kappa=%s, theta=%s, xi=%s, rho=%s, v0=%s", 
            self.kappa, self.theta, self.xi, self.rho, self.v0)

    def simulate(self, S0: float, T: float, r: float, q: float, npaths: int = 243, nsteps: int = 365,seed: int | None = None) -> tuple:
        if seed is not None:
            np.random.seed(seed)

        num_steps = round(nsteps * T)
        time_step = T / num_steps

        variance_paths = np.zeros((num_steps + 1, npaths))
        spot_paths = np.zeros((num_steps + 1, npaths))
        variance_paths[0] = self.v0
        spot_paths[0] = S0

        for step_idx in range(1, num_steps + 1):
            spot_brownian = np.random.normal(0, np.sqrt(time_step), npaths)
            independent_normal = np.random.normal(0, np.sqrt(time_step), npaths)
            variance_brownian = (self.rho * spot_brownian + np.sqrt(1 - self.rho**2) * independent_normal)

            variance_prev = np.maximum(variance_paths[step_idx - 1], 0)
            variance_paths[step_idx] = (variance_paths[step_idx - 1] + self.kappa * (self.theta - variance_prev) * time_step
                                        + self.xi * np.sqrt(variance_prev) * variance_brownian)
            variance_paths[step_idx] = np.maximum(variance_paths[step_idx], 0)

            drift_term = (r - q - 0.5 * variance_prev) * time_step
            diffusion_term = np.sqrt(variance_prev) * spot_brownian
            spot_paths[step_idx] = spot_paths[step_idx - 1] * np.exp(drift_term + diffusion_term)

        logger.debug("Simulation completed: S0=%s, T=%s, r=%s, q=%s, npaths=%s, nsteps=%s, seed=%s",
            S0, T, r, q, npaths, nsteps, seed)
        return spot_paths, variance_paths

    def heston_cf(self, u: np.ndarray, T: float, S0: float, r: float, q: float) -> np.ndarray:
        u = np.atleast_1d(u)
        i = 1j
        log_spot = np.log(S0)

        mean_var_term = self.kappa * self.theta
        mean_reversion_term = self.kappa - self.rho * self.xi * i * u
        sqrt_discriminant = np.sqrt(mean_reversion_term**2 + (self.xi**2) * (i * u + u * u))
        ratio = (mean_reversion_term - sqrt_discriminant) / (mean_reversion_term + sqrt_discriminant)

        exp_sqrt_disc_T = np.exp(-sqrt_discriminant * T)
        ratio_numerator = np.clip(1 - ratio * exp_sqrt_disc_T, 1e-15, None)
        ratio_denominator = np.clip(1 - ratio, 1e-15, None)

        log_char_func = (i * u * (r - q) * T + (mean_var_term / (self.xi**2)) * ((mean_reversion_term - sqrt_discriminant) 
                                                                                 * T - 2.0 * np.log(ratio_numerator / ratio_denominator)))
        variance_scaling = ((mean_reversion_term - sqrt_discriminant) / (self.xi**2)) * ((1 - exp_sqrt_disc_T) / ratio_numerator)

        char_func = np.exp(log_char_func + variance_scaling * self.v0 + i * u * log_spot)

        return char_func[0] if char_func.size == 1 else char_func

    def call_transform(
        self, v: np.ndarray, T: float, S0: float, r: float, q: float, alpha: float = 1.5) -> np.ndarray:
        v = np.atleast_1d(v)
        i = 1j

        phi = self.heston_cf(v - (alpha + 1) * i, T, S0, r, q)
        numerator = np.exp(-r * T) * phi
        denominator = alpha**2 + alpha - v**2 + i * (2 * alpha + 1) * v

        psi = numerator / denominator
        return psi
    
    def carr_madan_call(self, T: float, S0: float, r: float, q: float, K: np.ndarray | float, alpha: float = 1.5, N: int = 4096, eta: float = 0.225) -> np.ndarray:
        integration_var = np.arange(N) * eta
        transform_values = self.call_transform(integration_var, T, S0, r, q, alpha=alpha)

        # Trapezoidal weights for numerical integration
        integration_weights = eta * np.ones(N)
        integration_weights[0] = integration_weights[-1] = 0.5 * eta

        # Log-strike spacing and FFT grid setup
        log_strike_spacing = 2.0 * np.pi / (N * eta)
        max_log_strike = 0.5 * N * log_strike_spacing
        fft_input = (transform_values * np.exp(1j * max_log_strike * integration_var) * integration_weights)
        fft_output = np.fft.fft(fft_input)
        fft_real_part = fft_output.real

        # Log-strike grid and call prices
        log_strike_grid = -max_log_strike + log_strike_spacing * np.arange(N)
        strike_grid = np.exp(log_strike_grid)

        call_prices_grid = np.exp(-alpha * log_strike_grid) / np.pi * fft_real_part
        price_interpolator = interp1d(strike_grid, call_prices_grid, kind="cubic", fill_value="extrapolate")
        call_prices = price_interpolator(np.atleast_1d(K))

        logger.debug("Carr-Madan call priced: T=%s, S0=%s, K=%s, alpha=%s, N=%s, eta=%s",
            T, S0, K, alpha, N, eta)
        return call_prices
# Fix variables now from here 
    def heston_call(self, T: float, S0: float, r: float, q: float, K: np.ndarray | float, N: int = 2000, U_max: float = 175) -> np.ndarray:
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

        logger.debug("Heston call priced via characteristic function: T=%s, S0=%s, K=%s, N=%s, U_max=%s", 
            T, S0, K, N, U_max)
        return call_prices

    def monte_carlo_call(self, T: float, S0: float, r: float, q: float, K: np.ndarray | float, npaths: int = 200_000, nsteps: int = 365, seed: int | None = None) -> np.ndarray:
        K = np.array(K).reshape(-1)
        S, _ = self.simulate(S0, T, r, q, npaths=npaths, nsteps=nsteps, seed=seed)
        S_end = S[-1, :]

        S_col = S_end.reshape(-1, 1)
        K_row = K.reshape(1, -1)

        payoffs = np.maximum(S_col - K_row, 0)
        call_prices = np.exp(-r * T) * payoffs.mean(axis=0)

        logger.debug("Monte Carlo call priced: T=%s, S0=%s, K=%s, npaths=%s, nsteps=%s, seed=%s", 
                     T, S0, K, npaths, nsteps, seed)
        
        return call_prices


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info("HestonModel module ran as main script.")
