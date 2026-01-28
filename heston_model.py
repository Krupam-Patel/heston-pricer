"""Heston stochastic volatility model for commodities (energy, metals, ags)"""

# Fully done 1/28/2026
# Commenting done 1/28/2026

import logging
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class hCommModel:
    def __init__(self, h_param: dict):
        self.parameters = h_param
        self.kappa = h_param["kappa"] # Variance mean-reversion speed
        self.theta = h_param["theta"] # Long-run variance level
        self.xi = h_param["xi"]   # Volatility of volatility
        self.rho = h_param["rho"] # Correlation between spot and variance
        self.v0 = h_param["v0"] # Initial variance level
        
        # Validating the Feller condition
        feller_lhs = 2.0 * self.kappa * self.theta
        feller_rhs = self.xi ** 2
        if feller_lhs < feller_rhs:
            logger.warning("Feller condition violated: 2*kappa*theta (%.5f) < xi^2 (%.5f). " 
                           "Variance process may reach zero.", feller_lhs, feller_rhs)
        
        logger.info("Initialized hCommModel: "
            "kappa=%.5f, theta=%.5f, xi=%.5f, rho=%.5f, v0=%.5f", self.kappa, self.theta, self.xi, self.rho, self.v0)
        

    class simConfig:
        S0: float # Initial spot price
        T: float # Time to maturity (years)
        r: float # Risk-free rate
        q: float # Convenience yield
        
        # Simulation parameters
        M: int = 10000 # Number of paths
        N: int = 365 # Number of time steps
        seed: int | None = None # Random seed
        
        # Mean reversion overlay (aka the spot price)
        alpha: float = 0.0 # Mean reversion speed
        mu: float = 0.0 # Long-term mean level
        
        # Seasonality overlay (some commodities do better in certain months)
        A: float = 0.0 # Seasonal amplitude
        phi: float = 0.0 # Seasonal phase


    def simulate(self, config: simConfig) -> tuple[np.ndarray, np.ndarray]:
        if config.seed is not None:
            np.random.seed(config.seed)

        dt = config.T / config.N

        variance_paths = np.zeros((config.N + 1, config.M))
        spot_paths = np.zeros((config.N + 1, config.M))

        variance_paths[0] = self.v0
        spot_paths[0] = config.S0

        for t in range(1, config.N + 1):
            # Correlated Brownian motions (Heston structure) (also dW is just a short form for Brownian)
            dW_S = np.random.normal(0.0, np.sqrt(dt), config.M)
            dW_indep = np.random.normal(0.0, np.sqrt(dt), config.M)
            dW_v = self.rho * dW_S + np.sqrt(1.0 - self.rho ** 2) * dW_indep

            # CIR variance process (ensure non-negative)
            v_prev = np.maximum(variance_paths[t - 1], 0.0)
            variance_paths[t] = (variance_paths[t - 1] + self.kappa * (self.theta - v_prev) * dt + self.xi * np.sqrt(v_prev) * dW_v)
            variance_paths[t] = np.maximum(variance_paths[t], 0.0)

            # Commodity spot price SDE with overlays
            drift = ((config.r - config.q - 0.5 * v_prev) * dt + config.A * np.sin(2.0 * np.pi * (t * dt + config.phi) / config.T) 
                     + config.alpha * (config.mu - spot_paths[t - 1]) * dt)
            diffusion = np.sqrt(v_prev) * dW_S

            spot_paths[t] = spot_paths[t - 1] * np.exp(drift + diffusion)

        logger.debug("Commodity paths simulated: S0=%.5f, T=%.5f, M=%d, N=%d " "(seasonal=%s, mean_reversion=%s)",
                    config.S0, config.T, config.M, config.N, bool(config.A), bool(config.alpha))

        return spot_paths, variance_paths


# Analytical pricing methods
    def h_char_func(self, u: np.ndarray, T: float, S0: float, r: float, q: float) -> np.ndarray:
                # u: Fourier transform variable
                # T: Time to maturity
                # S0: Initial spot price
                # r: Risk-free rate
                # q: Convenience yield
            u = np.atleast_1d(u)
            i = 1j
            log_S0 = np.log(S0)

            kappa_theta = self.kappa * self.theta
            d_term = self.kappa - self.rho * self.xi * i * u
            
            g = np.sqrt(d_term ** 2 + (self.xi ** 2) * (i * u + u * u))
            
            g_ratio = (d_term - g) / (d_term + g)

            exp_g = np.exp(-g * T)
            num = np.clip(1.0 - g_ratio * exp_g, 1e-15, None)
            denom = np.clip(1.0 - g_ratio, 1e-15, None)

            log_cf = (i * u * (r - q) * T + (kappa_theta / (self.xi ** 2)) * ((d_term - g) * T - 2.0 * np.log(num / denom)))

            v_scale = ((d_term - g) / (self.xi ** 2)) * ((1.0 - exp_g) / num)

            cf_values = np.exp(log_cf + v_scale * self.v0 + i * u * log_S0)

            if cf_values.size == 1:
                return cf_values[0]
            
            return cf_values

# Carr-Madan damping transform of the call payoff
# This makes the payoff square-integrable and suitable for FFT pricing
    def carr_madan_call_transform(self, u: np.ndarray, T: float, S0: float, r: float, q: float, alpha: float = 1.5) -> np.ndarray:
            # alpha: Damping parameter (typically 1.5)

            u = np.atleast_1d(u)
            i = 1j

            shifted_cf = self.h_char_func(u - (alpha + 1.0) * i, T, S0, r, q)

            discount = np.exp(-r * T)
            num = discount * shifted_cf
            denom = (alpha ** 2 + alpha - u ** 2 + i * (2.0 * alpha + 1.0) * u)
            
            transform = num / denom

            return transform

# Fast FFT pricing for a strip of vanilla calls under Heston
# Returns call prices interpolated at requested k values
    def carr_madan_call_prices(self, T: float, S0: float, r: float, q: float, k: np.ndarray | float, alpha: float = 1.5, N: int = 4096, eta: float = 0.25) -> np.ndarray:
                # k: Strike price(s)
                # alpha: Carr-Madan damping factor
                # N: Number of FFT grid points
                # eta: Step size in Fourier space
            u_grid = np.arange(N) * eta
            transform = self.carr_madan_call_transform(u_grid, T, S0, r, q, alpha=alpha)

            # Trapezoidal weights
            weights = eta * np.ones(N)
            weights[0] = weights[-1] = 0.5 * eta

            # FFT setup on log-strike grid
            lambda_spacing = 2.0 * np.pi / (N * eta)
            b = 0.5 * N * lambda_spacing

            fft_input = transform * np.exp(1j * b * u_grid) * weights
            fft_output = np.fft.fft(fft_input)
            fft_real = fft_output.real

            log_strikes = -b + lambda_spacing * np.arange(N)
            K_grid = np.exp(log_strikes)
            C_grid = np.exp(-alpha * log_strikes) / np.pi * fft_real

            k_array = np.atleast_1d(k)
            interp = interp1d(K_grid, C_grid, kind="cubic", fill_value="extrapolate")
            prices = interp(k_array)

            logger.debug("Carr-Madan FFT strip: T=%.2f, S0=%.2f, k=%s, N=%d", T, S0, k, N)

            return prices

# Heston vanilla call price using Little-Heston trapezoidal integration
# Slower than FFT but good for isolated k values or high accuracy needs
    def h_call_prices_trapezoid(self, T: float, S0: float, r: float, q: float, k: np.ndarray | float, N: int = 2000, U_max: float = 175.0) -> np.ndarray:
                # N: Number of integration points
                # U_max: Upper bound for Fourier integration
            i = 1j
            k_array = np.array(k, ndmin=1)
            log_K = np.log(k_array).reshape(-1, 1)

            u_grid = np.linspace(1e-10, U_max, N)

            cf_shifted = self.h_char_func(u_grid - i, T, S0, r, q)
            cf_base = self.h_char_func(u_grid, T, S0, r, q)
            cf_norm = self.h_char_func(-i, T, S0, r, q)

            exp_term = np.exp(-i * u_grid * log_K)
            integrand_P1 = np.real(exp_term * cf_shifted / (i * u_grid * cf_norm))
            integrand_P2 = np.real(exp_term * cf_base / (i * u_grid))

            P1 = 0.5 + (1.0 / np.pi) * np.trapezoid(integrand_P1, u_grid, axis=1)
            P2 = 0.5 + (1.0 / np.pi) * np.trapezoid(integrand_P2, u_grid, axis=1)

            prices = S0 * np.exp(-q * T) * P1 - k_array * np.exp(-r * T) * P2

            logger.debug("Heston call (Trapezoid): T=%.2f, S0=%.2f, k=%s, Umax=%.1f", T, S0, k, U_max)

            return prices
    
# Monte Carlo vanilla call pricing
# Intended as a fallback for exotic payoffs or validation
    def monte_carlo_call_prices(self, T: float, S0: float, r: float, q: float, k: np.ndarray | float, M: int = 200_000, N: int = 365, seed: int | None = None) -> np.ndarray:
                # M: Number of Monte Carlo paths
                # N: Time steps per year
                # seed: Random seed for reproducibility
                
            k_array = np.array(k).reshape(-1)
            
            config = self.simConfig()
            config.S0 = S0
            config.T = T
            config.r = r
            config.q = q
            config.M = M
            config.N = N
            config.seed = seed
            
            spot_paths, _ = self.simulate(config)
            S_T = spot_paths[-1, :]

            S_T_col = S_T.reshape(-1, 1)
            K_row = k_array.reshape(1, -1)
            payoffs = np.maximum(S_T_col - K_row, 0.0)
            prices = np.exp(-r * T) * payoffs.mean(axis=0)

            logger.debug("MC call pricing: T=%.2f, S0=%.2f, k=%s, paths=%d", T, S0, k, M)

            return prices


# Risk Management
# Computed using central finite difference
    def delta(self, T: float, S0: float, r: float, q: float, k: np.ndarray | float, dS: float = 0.01) -> np.ndarray:
            
            # Delta: sensitivity of call price to the spot price dC/dS
            
            C_up = self.h_call_prices_trapezoid(T, S0 + dS, r, q, k)
            C_down = self.h_call_prices_trapezoid(T, S0 - dS, r, q, k)
            delta = (C_up - C_down) / (2.0 * dS)

            logger.debug("Delta computed: S0=%.2f, k=%s, mean_delta=%.6f", S0, k, 
                         delta.mean() if isinstance(delta, np.ndarray) else float(delta))
            return delta

# Computed using central finite difference
# gamma: second derivative, curvature of the option price in spot
    def gamma(self, T: float, S0: float, r: float, q: float, k: np.ndarray | float, dS: float = 0.01) -> np.ndarray:
            
            C_up = self.h_call_prices_trapezoid(T, S0 + dS, r, q, k)
            C_mid = self.h_call_prices_trapezoid(T, S0, r, q, k)
            C_down = self.h_call_prices_trapezoid(T, S0 - dS, r, q, k)
            gamma = (C_up - 2.0 * C_mid + C_down) / (dS ** 2)

            logger.debug("Gamma computed: S0=%.2f, k=%s, mean_gamma=%.6f", S0, k,
                         gamma.mean() if isinstance(gamma, np.ndarray) else float(gamma))
            
            return gamma

# Vega: sensitivity of call price to initial variance v0 (dC/dv0)
    def vega(self, T: float, S0: float, r: float, q: float, k: np.ndarray | float, dv: float = 0.01) -> np.ndarray:
            # This is vega with respect to variance, not volatility

            params_bumped = self.parameters.copy()
            params_bumped["v0"] = self.v0 + dv
            model_bumped = hCommModel(params_bumped)

            C_bumped = model_bumped.h_call_prices_trapezoid(T, S0, r, q, k)
            C_base = self.h_call_prices_trapezoid(T, S0, r, q, k)
            vega = (C_bumped - C_base) / dv

            logger.debug("Vega computed: S0=%.2f, k=%s, mean_vega=%.6f", S0, k,
                         vega.mean() if isinstance(vega, np.ndarray) else float(vega))
            
            return vega

# Theta: time decay of the option price dC/dT, per one day step
# Computed using forward finite difference (T -> T - dt)
    def theta(self, T: float, S0: float, r: float, q: float, k: np.ndarray | float, dt: float = 1.0 / 365.0) -> np.ndarray:
            C_now = self.h_call_prices_trapezoid(T, S0, r, q, k)
            C_shorter = self.h_call_prices_trapezoid(T - dt, S0, r, q, k)
            theta = -(C_shorter - C_now) / dt

            logger.debug("Theta computed: T=%.4f, S0=%.2f, k=%s, mean_theta=%.6f", T, S0, k,
                         theta.mean() if isinstance(theta, np.ndarray) else float(theta))
            
            return theta

# Rho: sensitivity of call price to the interest rate dC/dr
# Computed using central finite difference
    def rho(self, T: float, S0: float, r: float, q: float, k: np.ndarray | float, dr: float = 0.001) -> np.ndarray:
            C_up = self.h_call_prices_trapezoid(T, S0, r + dr, q, k)
            C_down = self.h_call_prices_trapezoid(T, S0, r - dr, q, k)
            rho = (C_up - C_down) / (2.0 * dr)

            logger.debug("Rho computed: S0=%.2f, k=%s, mean_rho=%.6f", S0, k,
                         rho.mean() if isinstance(rho, np.ndarray) else float(rho))
            
            return rho

# Calibration (Model Fitting)
# Calibrate Heston parameters to market vanilla option prices
# Minimizes mean squared error between model prices and market prices
    def calibrate_to_market(self, market_prices: np.ndarray, market_k: np.ndarray, T: float, S0: float, r: float, q: float, bounds: dict | None = None) -> dict:
            # market_prices: Observed market prices
            # market_k: Corresponding strike prices
            # bounds: Optional dict of (min, max) bounds for parameters
    
            if bounds is None:
                bounds = { "kappa": (0.01, 10.0), "theta": (0.001, 1.0), "xi": (0.01, 2.0), "rho": (-0.99, 0.99), "v0": (0.001, 1.0)}

            opt_bounds = [bounds["kappa"], bounds["theta"], bounds["xi"], bounds["rho"], bounds["v0"]]

# Objective function: mean squared error between model and market prices
            def objective(x: np.ndarray) -> float:
                kappa_c, theta_c, xi_c, rho_c, v0_c = x

                # Quick bounds check for safety
                for idx, val in enumerate(x):
                    lb, ub = opt_bounds[idx]
                    if (val < lb) or (val > ub):
                        return 1e10

                temp_model = hCommModel({ "kappa": kappa_c, "theta": theta_c, "xi": xi_c, "rho": rho_c, "v0": v0_c})
                
                try:
                    model_prices = temp_model.h_call_prices_trapezoid(T, S0, r, q, market_k)
                except Exception as e:
                    logger.warning("Pricing failed during calibration: %s", e)
                    return 1e10

                mse = np.mean((model_prices - market_prices) ** 2)

                return mse

            x0 = [self.kappa, self.theta, self.xi, self.rho, self.v0]

            result = minimize(objective, x0, method="L-BFGS-B", bounds=opt_bounds, options={"maxiter": 500, "ftol": 1e-8})

            kappa_cal, theta_cal, xi_cal, rho_cal, v0_cal = result.x

            params_cal = {"kappa": kappa_cal, "theta": theta_cal, "xi": xi_cal, "rho": rho_cal, "v0": v0_cal}

            logger.info("Calibration complete: success=%s, MSE=%.6f, " "kappa=%.4f, theta=%.4f, xi=%.4f, rho=%.4f, v0=%.4f",
                        result.success, result.fun, kappa_cal, theta_cal, xi_cal, rho_cal, v0_cal)

            # Update internal parameters to calibrated values
            self.parameters = params_cal
            self.kappa = kappa_cal
            self.theta = theta_cal
            self.xi = xi_cal
            self.rho = rho_cal
            self.v0 = v0_cal

            return {"success": result.success, "mse": result.fun, "params": params_cal, "iterations": result.nit}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    
    # Example usage
    h_params = {"kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7, "v0": 0.04}
    
    model = hCommModel(h_params)
    
    # Price a call option
    S0 = 100.0
    k = 100.0
    T = 1.0
    r = 0.05
    q = 0.02
    
    price_fft = model.carr_madan_call_prices(T, S0, r, q, k)
    price_trap = model.h_call_prices_trapezoid(T, S0, r, q, k)
    
    logger.info("Call price (FFT): %.4f", price_fft[0])
    logger.info("Call price (Trapezoid): %.4f", price_trap[0])
    
    logger.info("Heston Commodity Model module is good")
