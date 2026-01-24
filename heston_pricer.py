"""Heston Option Pricing Engine: exotic and vanilla (specific for commodities)"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class HestonCommodityPricer:
    def __init__(self, model):
        self.model = model
        logger.info("Initialized HP-commodities with model: %s", type(model).__name__)

    def price_european(self, maturity: float, spot: float, rate: float, delta: float, k: float, option_type: str = "Call") -> float:
        call_price = self.model.heston_call(maturity, spot, rate, delta, k)
        if option_type.lower() == "call":
            price = call_price[0]
        else:
            put_price = (call_price - spot * np.exp(-delta * maturity) + k * np.exp(-rate * maturity))
            price = put_price[0]

        logger.debug("European %s priced: maturity=%s, spot=%s, k=%s, rate=%s, delta=%s, price=%s", 
                    option_type, maturity, spot, k, rate, delta, price)
        
        return price

    def price_digital(self, maturity: float, spot: float, rate: float, delta: float, k: float, option_type: str = "Call", payout: float = 1.0,
                      n_paths: int = 250_000, seed: int | None = None) -> float:
        
        sim_paths, _ = self.model.simulate(S0=spot, T=maturity, r=rate, q=delta, npaths=n_paths, nsteps=252, seed=seed)
        terminal_prices = sim_paths[-1, :]
        if option_type.lower() == "call":
            payoff = (terminal_prices > k).astype(float) * payout
        else:
            payoff = (terminal_prices < k).astype(float) * payout

        disc_payoff = np.exp(-rate * maturity) * np.mean(payoff)

        logger.debug("Digital %s priced: maturity=%s, spot=%s, k=%s, rate=%s, delta=%s, " "payout=%s, n_paths=%s, seed=%s, price=%s",
                     option_type, maturity, spot, k, rate, delta, payout, n_paths, seed, disc_payoff)
        
        return disc_payoff

    def price_barrier(self, maturity: float, spot: float, rate: float, delta: float, k: float, barrier: float, barrier_type: str = "UpAndOut",
        option_type: str = "Call", n_paths: int = 250_000, n_steps: int = 365, seed: int | None = None) -> float:

        if barrier_type in ("UpAndOut", "UpAndIn") and barrier <= spot:
            logger.warning("UpBarrier %s <= spot %s may not work well", barrier, spot)
        elif barrier_type in ("DownAndOut", "DownAndIn") and barrier >= spot:
            logger.warning("DownBarrier %s >= spot %s may not work well", barrier, spot)

        paths, _ = self.model.simulate(S0=spot, T=maturity, r=rate, q=delta, npaths=n_paths, nsteps=n_steps, seed=seed)

        if barrier_type in ("UpAndOut", "UpAndIn"):
            hit = paths.max(axis=0) >= barrier
        elif barrier_type in ("DownAndOut", "DownAndIn"):
            hit = paths.min(axis=0) <= barrier
        else:
            logger.error("Invalid barrier type: %s", barrier_type)
            raise ValueError("Invalid barrier type")

        terminal = paths[-1, :]
        if option_type.lower() == "call":
            payoff = np.maximum(terminal - k, 0)
        elif option_type.lower() == "put":
            payoff = np.maximum(k - terminal, 0)
        else:
            logger.error("Invalid option type: %s", option_type)
            raise ValueError("Invalid option type")

        if barrier_type.endswith("Out"):
            payoff[hit] = 0
        else:
            payoff[hit == False] = 0

        price = np.exp(-rate * maturity) * np.mean(payoff)

        logger.debug("Barrier %s %s priced: maturity=%s, spot=%s, k=%s, barrier=%s, " 
                     "rate=%s, delta=%s, n_paths=%s, n_steps=%s, seed=%s, price=%s",
            barrier_type, option_type, maturity, spot, k, barrier, rate, delta, n_paths, n_steps, seed, price)
        
        return price

    def price_on_futures(self, maturity: float, futures_price: float, rate: float, k: float, option_type: str = "Call", n_paths: int = 250_000, seed: int | None = None) -> float:
        sim_paths, _ = self.model.simulate( S0=futures_price, T=maturity, r=0.0, q=0.0, npaths=n_paths, nsteps=252, seed=seed)
        terminal_prices = sim_paths[-1, :]

        if option_type.lower() == "call":
            payoff = np.maximum(terminal_prices - k, 0)
        else:
            payoff = np.maximum(k - terminal_prices, 0)

        discounted_payoff = np.exp(-rate * maturity) * np.mean(payoff)

        logger.debug("Futures option %s priced: maturity=%s, futures_price=%s, k=%s, rate=%s, " "n_paths=%s, seed=%s, price=%s",
            option_type, maturity, futures_price, k, rate, n_paths, seed, discounted_payoff)
        
        return discounted_payoff

    def price_with_mean_reversion(self, maturity: float, spot: float, rate: float, delta: float, k: float, mean_reversion_speed: float, mean_reversion_level: float,
                                  option_type: str = "Call", n_paths: int = 250_000, seed: int | None = None) -> float:
        if seed is not None:
            np.random.seed(seed)

        T = maturity
        nsteps = 252
        steps = round(nsteps * T)
        dt = T / steps

        spot_paths = np.zeros((steps + 1, n_paths))
        spot_paths[0] = spot

        variance_paths, _ = self.model.simulate(S0=spot, T=T, r=rate, q=delta, npaths=n_paths, nsteps=nsteps, seed=seed)

        for step_idx in range(1, steps + 1):
            brownian_increment = np.random.normal(0, np.sqrt(dt), n_paths)
            variance_prev = np.maximum(variance_paths[step_idx - 1], 0)

            mean_reversion_term = (mean_reversion_speed * (mean_reversion_level - spot_paths[step_idx - 1])* dt)
            drift_term = (rate - delta - 0.5 * variance_prev) * dt
            diffusion_term = np.sqrt(variance_prev) * brownian_increment

            spot_paths[step_idx] = spot_paths[step_idx - 1] * np.exp(mean_reversion_term + drift_term + diffusion_term)

        terminal_prices = spot_paths[-1, :]

        if option_type.lower() == "call":
            payoff = np.maximum(terminal_prices - k, 0)
        else:
            payoff = np.maximum(k - terminal_prices, 0)

        discounted_payoff = np.exp(-rate * maturity) * np.mean(payoff)

        logger.debug("Mean-reversion option %s priced: maturity=%s, spot=%s, k=%s, " "rate=%s, delta=%s, mr_speed=%s, mr_level=%s, price=%s", 
                     option_type, maturity, spot, k, rate, delta, mean_reversion_speed, mean_reversion_level, discounted_payoff)
        
        return discounted_payoff

    def price_with_seasonality(self, maturity: float, spot: float, rate: float,delta: float, k: float, seasonal_amplitude: float, seasonal_phase: float, 
                               option_type: str = "Call", n_paths: int = 250_000, seed: int | None = None) -> float:
        if seed is not None:
            np.random.seed(seed)

        T = maturity
        nsteps = 252
        steps = round(nsteps * T)
        dt = T / steps

        spot_paths = np.zeros((steps + 1, n_paths))
        spot_paths[0] = spot

        variance_paths, _ = self.model.simulate(S0=spot, T=T, r=rate, q=delta, npaths=n_paths, nsteps=nsteps, seed=seed)

        for step_idx in range(1, steps + 1):
            brownian_increment = np.random.normal(0, np.sqrt(dt), n_paths)
            variance_prev = np.maximum(variance_paths[step_idx - 1], 0)

            time_fraction = step_idx / steps
            seasonal_multiplier = 1.0 + seasonal_amplitude * np.sin(2 * np.pi * time_fraction + seasonal_phase)
            adjusted_variance = variance_prev * seasonal_multiplier**2

            drift_term = (rate - delta - 0.5 * adjusted_variance) * dt
            diffusion_term = np.sqrt(adjusted_variance) * brownian_increment

            spot_paths[step_idx] = spot_paths[step_idx - 1] * np.exp(drift_term + diffusion_term)

        terminal_prices = spot_paths[-1, :]

        if option_type.lower() == "call":
            payoff = np.maximum(terminal_prices - k, 0)
        else:
            payoff = np.maximum(k - terminal_prices, 0)

        discounted_payoff = np.exp(-rate * maturity) * np.mean(payoff)

        logger.debug("Seasonal option %s priced: maturity=%s, spot=%s, k=%s, "
                     "rate=%s, delta=%s, seasonal_amplitude=%s, seasonal_phase=%s, price=%s",
                     option_type,maturity, spot, k, rate, delta, seasonal_amplitude, seasonal_phase, discounted_payoff)
        
        return discounted_payoff


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger.info("HP- commodities module ran")
