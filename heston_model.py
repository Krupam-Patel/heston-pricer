"""Heston stochastic volatility model for commodities (energy, metals, ags)"""

import logging
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class hCommModel:
    def __init__(self, heston_param: dict):
        self.parameters = heston_param
        self.kappa = heston_param["kappa"] # Variance mean-reversion speed
        self.theta = heston_param["theta"] # Long-run variance level
        self.xi = heston_param["xi"]   # Volatility of volatility
        self.rho = heston_param["rho"] # Correlation between spot and variance
        self.v0 = heston_param["v0"] # Initial variance level
        
        # Validating the Feller condition
        feller_lhs = 2.0 * self.kappa * self.theta
        feller_rhs = self.xi ** 2
        if feller_lhs < feller_rhs:
            logger.warning("Feller condition violated: 2*kappa*theta (%.4f) < xi^2 (%.4f). " 
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
            variance_paths[t] = (
                variance_paths[t - 1] + 
                self.kappa * (self.theta - v_prev) * dt + 
                self.xi * np.sqrt(v_prev) * dW_v
            )
            variance_paths[t] = np.maximum(variance_paths[t], 0.0)

            # Commodity spot price SDE with overlays
            drift = (
                (config.r - config.q - 0.5 * v_prev) * dt +
                config.A * np.sin(2.0 * np.pi * (t * dt + config.phi) / config.T) +
                config.alpha * (config.mu - spot_paths[t - 1]) * dt
            )
            diffusion = np.sqrt(v_prev) * dW_S

            spot_paths[t] = spot_paths[t - 1] * np.exp(drift + diffusion)

        logger.debug(
            "Commodity paths simulated: S0=%.2f, T=%.2f, M=%d, N=%d "
            "(seasonal=%s, mean_reversion=%s)",
            config.S0, config.T, config.M, config.N,
            bool(config.A), bool(config.alpha)
        )

        return spot_paths, variance_paths

    # ==================== ANALYTICAL PRICING ====================
    def heston_characteristic_function(
        self,
        fourier_variable: np.ndarray,
        maturity_time_years: float,
        initial_spot_price: float,
        risk_free_rate: float,
        convenience_yield: float,
    ) -> np.ndarray:
        """
        Heston characteristic function in the Fourier domain.
        
        This is the key analytical ingredient for FFT and semi-analytical pricing.
        
        Args:
            fourier_variable: Fourier transform variable (can be array)
            maturity_time_years: Time to maturity
            initial_spot_price: Current spot price
            risk_free_rate: Risk-free rate
            convenience_yield: Convenience yield
            
        Returns:
            Characteristic function values
        """
        fourier_variable = np.atleast_1d(fourier_variable)
        imaginary_unit = 1j
        log_initial_spot = np.log(initial_spot_price)

        mean_variance_product = self.kappa * self.theta
        mean_reversion_term = (
            self.kappa - self.rho * self.xi * imaginary_unit * fourier_variable
        )
        discriminant_term = np.sqrt(
            mean_reversion_term ** 2
            + (self.xi ** 2)
            * (imaginary_unit * fourier_variable + fourier_variable * fourier_variable)
        )
        discriminant_ratio = (mean_reversion_term - discriminant_term) / (
            mean_reversion_term + discriminant_term
        )

        exponential_discriminant = np.exp(-discriminant_term * maturity_time_years)
        ratio_numerator = np.clip(
            1.0 - discriminant_ratio * exponential_discriminant, 1e-15, None
        )
        ratio_denominator = np.clip(1.0 - discriminant_ratio, 1e-15, None)

        log_characteristic_function = (
            imaginary_unit
            * fourier_variable
            * (risk_free_rate - convenience_yield)
            * maturity_time_years
            + (mean_variance_product / (self.xi ** 2))
            * (
                (mean_reversion_term - discriminant_term) * maturity_time_years
                - 2.0 * np.log(ratio_numerator / ratio_denominator)
            )
        )

        variance_scaling_term = (
            (mean_reversion_term - discriminant_term) / (self.xi ** 2)
        ) * ((1.0 - exponential_discriminant) / ratio_numerator)

        characteristic_function_values = np.exp(
            log_characteristic_function
            + variance_scaling_term * self.v0
            + imaginary_unit * fourier_variable * log_initial_spot
        )

        if characteristic_function_values.size == 1:
            return characteristic_function_values[0]
        return characteristic_function_values

    def carr_madan_call_transform(
        self,
        fourier_grid: np.ndarray,
        maturity_time_years: float,
        initial_spot_price: float,
        risk_free_rate: float,
        convenience_yield: float,
        damping_factor: float = 1.5,
    ) -> np.ndarray:
        """
        Carr-Madan damping transform of the call payoff.

        This makes the payoff square-integrable and suitable for FFT pricing.
        
        Args:
            fourier_grid: Grid of Fourier variables
            maturity_time_years: Time to maturity
            initial_spot_price: Current spot price
            risk_free_rate: Risk-free rate
            convenience_yield: Convenience yield
            damping_factor: Damping parameter (typically 1.5)
            
        Returns:
            Transformed payoff values
        """
        fourier_grid = np.atleast_1d(fourier_grid)
        imaginary_unit = 1j

        shifted_characteristic_function = self.heston_characteristic_function(
            fourier_grid - (damping_factor + 1.0) * imaginary_unit,
            maturity_time_years,
            initial_spot_price,
            risk_free_rate,
            convenience_yield,
        )

        discounted_factor = np.exp(-risk_free_rate * maturity_time_years)
        numerator = discounted_factor * shifted_characteristic_function
        denominator = (
            damping_factor ** 2
            + damping_factor
            - fourier_grid ** 2
            + imaginary_unit * (2.0 * damping_factor + 1.0) * fourier_grid
        )
        transformed_payoff = numerator / denominator
        return transformed_payoff

    def carr_madan_call_prices(
        self,
        maturity_time_years: float,
        initial_spot_price: float,
        risk_free_rate: float,
        convenience_yield: float,
        strikes: np.ndarray | float,
        damping_factor: float = 1.5,
        num_fft_points: int = 4096,
        fourier_step_size: float = 0.25,
    ) -> np.ndarray:
        """
        Fast FFT pricing for a strip of vanilla calls under Heston.

        Returns call prices interpolated at requested strikes.
        
        Args:
            maturity_time_years: Time to maturity
            initial_spot_price: Current spot price
            risk_free_rate: Risk-free rate
            convenience_yield: Convenience yield
            strikes: Strike price(s)
            damping_factor: Carr-Madan damping factor
            num_fft_points: Number of FFT grid points
            fourier_step_size: Step size in Fourier space
            
        Returns:
            Call option prices at requested strikes
        """
        fourier_grid = np.arange(num_fft_points) * fourier_step_size
        transformed_payoff = self.carr_madan_call_transform(
            fourier_grid,
            maturity_time_years,
            initial_spot_price,
            risk_free_rate,
            convenience_yield,
            damping_factor=damping_factor,
        )

        # Trapezoidal weights
        integration_weights = fourier_step_size * np.ones(num_fft_points)
        integration_weights[0] = integration_weights[-1] = 0.5 * fourier_step_size

        # FFT setup on log-strike grid
        log_strike_spacing = 2.0 * np.pi / (num_fft_points * fourier_step_size)
        max_log_strike = 0.5 * num_fft_points * log_strike_spacing

        fft_input_values = (
            transformed_payoff
            * np.exp(1j * max_log_strike * fourier_grid)
            * integration_weights
        )
        fft_output_values = np.fft.fft(fft_input_values)
        fft_real_component = fft_output_values.real

        log_strike_grid = -max_log_strike + log_strike_spacing * np.arange(
            num_fft_points
        )
        strike_grid = np.exp(log_strike_grid)
        call_price_grid = (
            np.exp(-damping_factor * log_strike_grid) / np.pi * fft_real_component
        )

        strikes_array = np.atleast_1d(strikes)
        price_interpolator = interp1d(
            strike_grid,
            call_price_grid,
            kind="cubic",
            fill_value="extrapolate",
        )
        interpolated_call_prices = price_interpolator(strikes_array)

        logger.debug(
            "Carr-Madan FFT strip: T=%.2f, S0=%.2f, strikes=%s, N=%d",
            maturity_time_years,
            initial_spot_price,
            strikes,
            num_fft_points,
        )
        return interpolated_call_prices

    def heston_call_prices_trapezoid(
        self,
        maturity_time_years: float,
        initial_spot_price: float,
        risk_free_rate: float,
        convenience_yield: float,
        strikes: np.ndarray | float,
        num_integration_points: int = 2000,
        fourier_upper_bound: float = 175.0,
    ) -> np.ndarray:
        """
        Heston vanilla call price using Little-Heston trapezoidal integration.

        Slower than FFT but good for isolated strikes or high accuracy needs.
        
        Args:
            maturity_time_years: Time to maturity
            initial_spot_price: Current spot price
            risk_free_rate: Risk-free rate
            convenience_yield: Convenience yield
            strikes: Strike price(s)
            num_integration_points: Number of integration points
            fourier_upper_bound: Upper bound for Fourier integration
            
        Returns:
            Call option prices
        """
        imaginary_unit = 1j
        strikes_array = np.array(strikes, ndmin=1)
        log_strikes = np.log(strikes_array).reshape(-1, 1)

        fourier_grid = np.linspace(
            1e-10, fourier_upper_bound, num_integration_points
        )

        shifted_characteristic_function = self.heston_characteristic_function(
            fourier_grid - imaginary_unit,
            maturity_time_years,
            initial_spot_price,
            risk_free_rate,
            convenience_yield,
        )
        characteristic_function_base = self.heston_characteristic_function(
            fourier_grid,
            maturity_time_years,
            initial_spot_price,
            risk_free_rate,
            convenience_yield,
        )
        characteristic_function_normalizer = self.heston_characteristic_function(
            -imaginary_unit,
            maturity_time_years,
            initial_spot_price,
            risk_free_rate,
            convenience_yield,
        )

        exponential_term = np.exp(-imaginary_unit * fourier_grid * log_strikes)
        integrand_P1 = np.real(
            exponential_term
            * shifted_characteristic_function
            / (imaginary_unit * fourier_grid * characteristic_function_normalizer)
        )
        integrand_P2 = np.real(
            exponential_term
            * characteristic_function_base
            / (imaginary_unit * fourier_grid)
        )

        probability_P1 = 0.5 + (1.0 / np.pi) * np.trapezoid(
            integrand_P1, fourier_grid, axis=1
        )
        probability_P2 = 0.5 + (1.0 / np.pi) * np.trapezoid(
            integrand_P2, fourier_grid, axis=1
        )

        call_prices = (
            initial_spot_price
            * np.exp(-convenience_yield * maturity_time_years)
            * probability_P1
            - strikes_array
            * np.exp(-risk_free_rate * maturity_time_years)
            * probability_P2
        )

        logger.debug(
            "Heston call (Trapezoid): T=%.2f, S0=%.2f, K=%s, Umax=%.1f",
            maturity_time_years,
            initial_spot_price,
            strikes,
            fourier_upper_bound,
        )
        return call_prices

    def monte_carlo_call_prices(
        self,
        maturity_time_years: float,
        initial_spot_price: float,
        risk_free_rate: float,
        convenience_yield: float,
        strikes: np.ndarray | float,
        num_paths: int = 200_000,
        num_time_steps: int = 365,
        random_seed: int | None = None,
    ) -> np.ndarray:
        """
        Monte Carlo vanilla call pricing.

        Intended as a fallback for exotic payoffs or validation.
        
        Args:
            maturity_time_years: Time to maturity
            initial_spot_price: Current spot price
            risk_free_rate: Risk-free rate
            convenience_yield: Convenience yield
            strikes: Strike price(s)
            num_paths: Number of Monte Carlo paths
            num_time_steps: Time steps per year
            random_seed: Random seed for reproducibility
            
        Returns:
            Call option prices
        """
        strikes_array = np.array(strikes).reshape(-1)
        spot_paths, _ = self.simulate(
            initial_spot_price=initial_spot_price,
            maturity_time_years=maturity_time_years,
            risk_free_rate=risk_free_rate,
            convenience_yield=convenience_yield,
            num_paths=num_paths,
            num_time_steps=num_time_steps,
            random_seed=random_seed,
        )
        terminal_spot_prices = spot_paths[-1, :]

        terminal_spot_column = terminal_spot_prices.reshape(-1, 1)
        strike_row = strikes_array.reshape(1, -1)
        payoff_matrix = np.maximum(terminal_spot_column - strike_row, 0.0)
        discounted_call_prices = (
            np.exp(-risk_free_rate * maturity_time_years) * payoff_matrix.mean(axis=0)
        )

        logger.debug(
            "MC call pricing: T=%.2f, S0=%.2f, K=%s, paths=%d",
            maturity_time_years,
            initial_spot_price,
            strikes,
            num_paths,
        )
        return discounted_call_prices

    # ==================== GREEKS (Risk Management) ====================
    def delta(
        self,
        maturity_time_years: float,
        initial_spot_price: float,
        risk_free_rate: float,
        convenience_yield: float,
        strikes: np.ndarray | float,
        spot_bump_size: float = 0.01,
    ) -> np.ndarray:
        """
        Delta: sensitivity of call price to the spot price dC/dS.
        
        Computed using central finite difference.
        """
        call_prices_up = self.heston_call_prices_trapezoid(
            maturity_time_years,
            initial_spot_price + spot_bump_size,
            risk_free_rate,
            convenience_yield,
            strikes,
        )
        call_prices_down = self.heston_call_prices_trapezoid(
            maturity_time_years,
            initial_spot_price - spot_bump_size,
            risk_free_rate,
            convenience_yield,
            strikes,
        )
        delta_values = (call_prices_up - call_prices_down) / (2.0 * spot_bump_size)

        logger.debug(
            "Delta computed: S0=%.2f, K=%s, mean_delta=%.6f",
            initial_spot_price,
            strikes,
            delta_values.mean()
            if isinstance(delta_values, np.ndarray)
            else float(delta_values),
        )
        return delta_values

    def gamma(
        self,
        maturity_time_years: float,
        initial_spot_price: float,
        risk_free_rate: float,
        convenience_yield: float,
        strikes: np.ndarray | float,
        spot_bump_size: float = 0.01,
    ) -> np.ndarray:
        """
        Gamma: second derivative d²C/dS², curvature of the option price in spot.
        
        Computed using central finite difference.
        """
        call_prices_up = self.heston_call_prices_trapezoid(
            maturity_time_years,
            initial_spot_price + spot_bump_size,
            risk_free_rate,
            convenience_yield,
            strikes,
        )
        call_prices_mid = self.heston_call_prices_trapezoid(
            maturity_time_years,
            initial_spot_price,
            risk_free_rate,
            convenience_yield,
            strikes,
        )
        call_prices_down = self.heston_call_prices_trapezoid(
            maturity_time_years,
            initial_spot_price - spot_bump_size,
            risk_free_rate,
            convenience_yield,
            strikes,
        )
        gamma_values = (
            call_prices_up - 2.0 * call_prices_mid + call_prices_down
        ) / (spot_bump_size ** 2)

        logger.debug(
            "Gamma computed: S0=%.2f, K=%s, mean_gamma=%.6f",
            initial_spot_price,
            strikes,
            gamma_values.mean()
            if isinstance(gamma_values, np.ndarray)
            else float(gamma_values),
        )
        return gamma_values

    def vega(
        self,
        maturity_time_years: float,
        initial_spot_price: float,
        risk_free_rate: float,
        convenience_yield: float,
        strikes: np.ndarray | float,
        variance_bump_size: float = 0.01,
    ) -> np.ndarray:
        """
        Vega: sensitivity of call price to initial variance v0 (dC/dv0).
        
        Note: This is vega with respect to variance, not volatility.
        """
        bumped_parameters = self.parameters.copy()
        bumped_parameters["v0"] = self.v0 + variance_bump_size
        bumped_model = hCommModel(bumped_parameters)

        call_prices_bumped = bumped_model.heston_call_prices_trapezoid(
            maturity_time_years,
            initial_spot_price,
            risk_free_rate,
            convenience_yield,
            strikes,
        )
        call_prices_base = self.heston_call_prices_trapezoid(
            maturity_time_years,
            initial_spot_price,
            risk_free_rate,
            convenience_yield,
            strikes,
        )
        vega_values = (call_prices_bumped - call_prices_base) / variance_bump_size

        logger.debug(
            "Vega computed: S0=%.2f, K=%s, mean_vega=%.6f",
            initial_spot_price,
            strikes,
            vega_values.mean()
            if isinstance(vega_values, np.ndarray)
            else float(vega_values),
        )
        return vega_values

    def theta(
        self,
        maturity_time_years: float,
        initial_spot_price: float,
        risk_free_rate: float,
        convenience_yield: float,
        strikes: np.ndarray | float,
        time_bump_size: float = 1.0 / 365.0,
    ) -> np.ndarray:
        """
        Theta: time decay of the option price dC/dT, per one day step.
        
        Computed using forward finite difference (T -> T - dt).
        """
        call_prices_now = self.heston_call_prices_trapezoid(
            maturity_time_years,
            initial_spot_price,
            risk_free_rate,
            convenience_yield,
            strikes,
        )
        call_prices_shorter_maturity = self.heston_call_prices_trapezoid(
            maturity_time_years - time_bump_size,
            initial_spot_price,
            risk_free_rate,
            convenience_yield,
            strikes,
        )
        theta_values = -(
            call_prices_shorter_maturity - call_prices_now
        ) / time_bump_size

        logger.debug(
            "Theta computed: T=%.4f, S0=%.2f, K=%s, mean_theta=%.6f",
            maturity_time_years,
            initial_spot_price,
            strikes,
            theta_values.mean()
            if isinstance(theta_values, np.ndarray)
            else float(theta_values),
        )
        return theta_values

    def rho(
        self,
        maturity_time_years: float,
        initial_spot_price: float,
        risk_free_rate: float,
        convenience_yield: float,
        strikes: np.ndarray | float,
        rate_bump_size: float = 0.001,
    ) -> np.ndarray:
        """
        Rho: sensitivity of call price to the interest rate dC/dr.
        
        Computed using central finite difference.
        """
        call_prices_up = self.heston_call_prices_trapezoid(
            maturity_time_years,
            initial_spot_price,
            risk_free_rate + rate_bump_size,
            convenience_yield,
            strikes,
        )
        call_prices_down = self.heston_call_prices_trapezoid(
            maturity_time_years,
            initial_spot_price,
            risk_free_rate - rate_bump_size,
            convenience_yield,
            strikes,
        )
        rho_values = (call_prices_up - call_prices_down) / (2.0 * rate_bump_size)

        logger.debug(
            "Rho computed: S0=%.2f, K=%s, mean_rho=%.6f",
            initial_spot_price,
            strikes,
            rho_values.mean()
            if isinstance(rho_values, np.ndarray)
            else float(rho_values),
        )
        return rho_values

    # ==================== CALIBRATION (Model Fitting) ====================
    def calibrate_to_market(
        self,
        market_option_prices: np.ndarray,
        market_strikes: np.ndarray,
        maturity_time_years: float,
        initial_spot_price: float,
        risk_free_rate: float,
        convenience_yield: float,
        parameter_bounds: dict | None = None,
    ) -> dict:
        """
        Calibrate Heston parameters to market vanilla option prices.

        Minimizes mean squared error between model prices and market prices.
        
        Args:
            market_option_prices: Observed market prices
            market_strikes: Corresponding strike prices
            maturity_time_years: Option maturity
            initial_spot_price: Current spot price
            risk_free_rate: Risk-free rate
            convenience_yield: Convenience yield
            parameter_bounds: Optional dict of (min, max) bounds for parameters
            
        Returns:
            Dictionary with calibration results
        """
        if parameter_bounds is None:
            parameter_bounds = {
                "kappa": (0.01, 10.0),
                "theta": (0.001, 1.0),
                "xi": (0.01, 2.0),
                "rho": (-0.99, 0.99),
                "v0": (0.001, 1.0),
            }

        optimization_bounds = [
            parameter_bounds["kappa"],
            parameter_bounds["theta"],
            parameter_bounds["xi"],
            parameter_bounds["rho"],
            parameter_bounds["v0"],
        ]

        def calibration_objective(parameter_vector: np.ndarray) -> float:
            """
            Objective function: mean squared error between model and market prices.
            """
            (
                candidate_kappa,
                candidate_theta,
                candidate_xi,
                candidate_rho,
                candidate_v0,
            ) = parameter_vector

            # Quick bounds check for safety
            for index, parameter_value in enumerate(parameter_vector):
                lower_bound, upper_bound = optimization_bounds[index]
                if (parameter_value < lower_bound) or (
                    parameter_value > upper_bound
                ):
                    return 1e10

            temporary_model = hCommModel(
                {
                    "kappa": candidate_kappa,
                    "theta": candidate_theta,
                    "xi": candidate_xi,
                    "rho": candidate_rho,
                    "v0": candidate_v0,
                }
            )
            
            try:
                model_generated_prices = temporary_model.heston_call_prices_trapezoid(
                    maturity_time_years,
                    initial_spot_price,
                    risk_free_rate,
                    convenience_yield,
                    market_strikes,
                )
            except Exception as e:
                logger.warning("Pricing failed during calibration: %s", e)
                return 1e10

            mse_value = np.mean((model_generated_prices - market_option_prices) ** 2)
            return mse_value

        initial_guess = [
            self.kappa,
            self.theta,
            self.xi,
            self.rho,
            self.v0,
        ]

        optimization_result = minimize(
            calibration_objective,
            initial_guess,
            method="L-BFGS-B",
            bounds=optimization_bounds,
            options={"maxiter": 500, "ftol": 1e-8},
        )

        (
            calibrated_kappa,
            calibrated_theta,
            calibrated_xi,
            calibrated_rho,
            calibrated_v0,
        ) = optimization_result.x

        calibrated_parameters = {
            "kappa": calibrated_kappa,
            "theta": calibrated_theta,
            "xi": calibrated_xi,
            "rho": calibrated_rho,
            "v0": calibrated_v0,
        }

        logger.info(
            "Calibration complete: success=%s, MSE=%.6f, "
            "kappa=%.4f, theta=%.4f, xi=%.4f, rho=%.4f, v0=%.4f",
            optimization_result.success,
            optimization_result.fun,
            calibrated_kappa,
            calibrated_theta,
            calibrated_xi,
            calibrated_rho,
            calibrated_v0,
        )

        # Update internal parameters to calibrated values
        self.parameters = calibrated_parameters
        self.kappa = calibrated_kappa
        self.theta = calibrated_theta
        self.xi = calibrated_xi
        self.rho = calibrated_rho
        self.v0 = calibrated_v0

        return {
            "success": optimization_result.success,
            "mse": optimization_result.fun,
            "params": calibrated_parameters,
            "iterations": optimization_result.nit,
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    
    # Example usage
    heston_params = {
        "kappa": 2.0,
        "theta": 0.04,
        "xi": 0.3,
        "rho": -0.7,
        "v0": 0.04,
    }
    
    model = hCommModel(heston_params)
    
    # Price a call option
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.02
    
    price_fft = model.carr_madan_call_prices(T, S0, r, q, K)
    price_trap = model.heston_call_prices_trapezoid(T, S0, r, q, K)
    
    logger.info("Call price (FFT): %.4f", price_fft[0])
    logger.info("Call price (Trapezoid): %.4f", price_trap[0])
    
    logger.info("Heston Commodity Model module loaded")
