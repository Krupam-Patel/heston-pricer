"""Heston Option Pricing Engine: exotic and vanilla (specific for commodities)"""

# Fully done 12/26/2025
# Commenting done 1/26/2026

# I tried putting comments to explain my thought process for everything in the code
# If something is still fuzzy, there is a typo, or there is a better way to do something, please dm on linkedin!
#https://www.linkedin.com/in/krupam-patel/
import logging
import numpy as np

logger = logging.getLogger(__name__)

class hCommPricer:
    def __init__(self, model):
        # Stores the Heston model instance for pricing and simulation calls
        self.model = model
        logger.info("Initialized HP-commodities with model: %s", type(model).__name__)

    def _validate_ot(self, ot: str) -> str:
        # Standardize option type to lowercase and validate it's call/put
        ot_lower = ot.lower()
        if ot_lower not in ("call", "put"):
            logger.error("Invalid ot: %s", ot)
            raise ValueError("ot must be 'call' or 'put'")
        return ot_lower

    def validate_positive(self, name: str, value: float) -> None:
        # Ensure parameters like maturity, spot, strike are positive (financial requirement)
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    # European Vanilla Pricer
    def price_european(self, maturity: float, spot: float, rate: float, delta: float, k: float, ot: str = "Call") -> float:
        # Validate all positive inputs first
        self.validate_positive("maturity", maturity)
        self.validate_positive("spot", spot)
        self.validate_positive("k", k)
        ot = self._validate_ot(ot)

        # Use model's analytical Heston call formula (tried doing a non-MC method for vanilla for fastness)
        call_price = self.model.heston_call(maturity, spot, rate, delta, k)
        c = float(call_price[0])

        # Convert call to put using put-call parity
        if ot == "call":
            price = c
        else:
            put_price = c - spot * np.exp(-delta * maturity) + k * np.exp(-rate * maturity)
            price = float(put_price)

        logger.debug("European %s priced: maturity=%s, spot=%s, k=%s, rate=%s, delta=%s, price=%s",
                    ot, maturity, spot, k, rate, delta, price)
        return price

    # Digital Option Pricer
    def price_digital(self, maturity: float, spot: float, rate: float, delta: float, k: float, ot: str = "Call", 
                      payout: float = 1.0, n_paths: int = 250_000, seed: int | None = None) -> float:
        # Validate inputs (digital needs path count for MC simulation)
        self.validate_positive("maturity", maturity)
        self.validate_positive("spot", spot)
        self.validate_positive("k", k)
        self.validate_positive("n_paths", n_paths)
        ot = self._validate_ot(ot)

        # Generate Heston paths, take terminal values
        sim_paths, _ = self.model.simulate(S0=spot, T=maturity, r=rate, q=delta, npaths=n_paths, nsteps=252, seed=seed)
        terminal_prices = sim_paths[-1, :]

        # Digital payoff: 1 if ITM, 0 otherwise (binary way of saying cash-or-nothing basicly)
        if ot == "call":
            payoff = (terminal_prices > k).astype(float) * payout
        else:
            payoff = (terminal_prices < k).astype(float) * payout

        # Discount mean payoff (MC estimator)
        disc_payoff = np.exp(-rate * maturity) * float(np.mean(payoff))

        logger.debug("Digital %s priced: maturity=%s, spot=%s, k=%s, rate=%s, delta=%s, payout=%s, n_paths=%s, seed=%s, price=%s",
                    ot, maturity, spot, k, rate, delta, payout, n_paths, seed, disc_payoff)
        return disc_payoff

    # Barrier Option Pricer
    def price_barrier(self, maturity: float, spot: float, rate: float, delta: float, k: float, barrier: float, 
                      barrier_type: str = "UpAndOut", ot: str = "Call",
                      n_paths: int = 250_000, n_steps: int = 365, seed: int | None = None) -> float:
        # Validate inputs (more steps for barrier monitoring)
        self.validate_positive("maturity", maturity)
        self.validate_positive("spot", spot)
        self.validate_positive("k", k)
        self.validate_positive("n_paths", n_paths)
        self.validate_positive("n_steps", n_steps)
        ot = self._validate_ot(ot)

        # Warn if barrier positioning doesn't make sense (e.g., up barrier below spot)
        if barrier_type in ("UpAndOut", "UpAndIn") and barrier <= spot:
            logger.warning("Up barrier %s <= spot %s may not work well", barrier, spot)
        elif barrier_type in ("DownAndOut", "DownAndIn") and barrier >= spot:
            logger.warning("Down barrier %s >= spot %s may not work well", barrier, spot)

        # Simulate full paths to monitor barrier hits
        paths, _ = self.model.simulate(S0=spot, T=maturity, r=rate, q=delta, npaths=n_paths, nsteps=n_steps, seed=seed)

        # Check if barrier hit: max/min across path for up/down barriers
        if barrier_type in ("UpAndOut", "UpAndIn"):
            hit = paths.max(axis=0) >= barrier
        elif barrier_type in ("DownAndOut", "DownAndIn"):
            hit = paths.min(axis=0) <= barrier
        else:
            logger.error("Invalid barrier type: %s", barrier_type)
            raise ValueError("Invalid barrier type")

        terminal = paths[-1, :]

        # Vanilla payoff first
        if ot == "call":
            payoff = np.maximum(terminal - k, 0.0)
        else:
            payoff = np.maximum(k - terminal, 0.0)

        # Apply barrier condition: knock-out (payoff=0 if hit) or knock-in (payoff=0 if not hit)
        if barrier_type.endswith("Out"):
            payoff[hit] = 0.0
        else:  # AndIn
            payoff[hit == False] = 0.0

        # Discount MC average
        price = np.exp(-rate * maturity) * float(np.mean(payoff))

        logger.debug("Barrier %s %s priced: maturity=%s, spot=%s, k=%s, barrier=%s, rate=%s, delta=%s, n_paths=%s, n_steps=%s, seed=%s, price=%s",
                    barrier_type, ot, maturity, spot, k, barrier, rate, delta, n_paths, n_steps, seed, price)
        return price

#Futures option pricer
    def price_on_futures(self, maturity: float, futures_price: float, rate: float, k: float, ot: str = "Call", 
                         n_paths: int = 250_000, seed: int | None = None) -> float:
        # Validate inputs (futures-specific)
        self.validate_positive("maturity", maturity)
        self.validate_positive("futures_price", futures_price)
        self.validate_positive("k", k)
        self.validate_positive("n_paths", n_paths)
        ot = self._validate_ot(ot)

        # Futures have no cost of carry, discount separately at end
        sim_paths, _ = self.model.simulate(S0=futures_price, T=maturity, r=0.0, q=0.0, npaths=n_paths, nsteps=252, seed=seed)
        terminal_prices = sim_paths[-1, :]

        # Standard option payoff on futures terminal value
        if ot == "call":
            payoff = np.maximum(terminal_prices - k, 0.0)
        else:
            payoff = np.maximum(k - terminal_prices, 0.0)

        disc_payoff = np.exp(-rate * maturity) * float(np.mean(payoff))

        logger.debug("Futures option %s priced: maturity=%s, futures_price=%s, k=%s, rate=%s, n_paths=%s, seed=%s, price=%s", 
                    ot, maturity, futures_price, k, rate, n_paths, seed, disc_payoff)
        return disc_payoff

# Mean reversion wrapper
    def price_with_mean_reversion(self, maturity: float, spot: float, rate: float, delta: float, k: float, 
                                  mean_reversion_speed: float, mean_reversion_level: float, ot: str = "Call", 
                                  n_paths: int = 250_000, seed: int | None = None) -> float:
        # Validate additional mean-reversion parameters
        self.validate_positive("maturity", maturity)
        self.validate_positive("spot", spot)
        self.validate_positive("k", k)
        self.validate_positive("mean_reversion_speed", mean_reversion_speed)
        self.validate_positive("n_paths", n_paths)
        ot = self._validate_ot(ot)

        # Pass mean-reversion params to Heston simulator
        paths, _ = self.model.simulate(S0=spot, T=maturity, r=rate, q=delta, npaths=n_paths, nsteps=252, 
                                      seed=seed, mean_reversion_speed=mean_reversion_speed, 
                                      mean_reversion_level=mean_reversion_level)
        terminal_prices = paths[-1, :]

        # Standard vanilla payoff on mean-reverting paths
        if ot == "call":
            payoff = np.maximum(terminal_prices - k, 0.0)
        else:
            payoff = np.maximum(k - terminal_prices, 0.0)

        disc_payoff = np.exp(-rate * maturity) * float(np.mean(payoff))

        logger.debug("Mean-reversion option %s priced: maturity=%s, spot=%s, k=%s, rate=%s, delta=%s, mr_speed=%s, mr_level=%s, price=%s",
                    ot, maturity, spot, k, rate, delta, mean_reversion_speed, mean_reversion_level, disc_payoff)
        return disc_payoff

# Seasonality wrapper
    def price_with_seasonality(self, maturity: float, spot: float, rate: float, delta: float, k: float, 
                               seasonal_amplitude: float, seasonal_phase: float, ot: str = "Call", 
                               n_paths: int = 250_000, seed: int | None = None) -> float:
        # Validate inputs (seasonality for commodities like energy/agri)
        self.validate_positive("maturity", maturity)
        self.validate_positive("spot", spot)
        self.validate_positive("k", k)
        self.validate_positive("n_paths", n_paths)
        ot = self._validate_ot(ot)

        # Pass seasonality parameters to simulator (deterministic seasonal drift)
        paths, _ = self.model.simulate(S0=spot, T=maturity, r=rate, q=delta, npaths=n_paths, nsteps=252, 
                                      seed=seed, seasonal_amplitude=seasonal_amplitude, 
                                      seasonal_phase=seasonal_phase)
        terminal_prices = paths[-1, :]

        # Standard vanilla payoff on seasonal paths
        if ot == "call":
            payoff = np.maximum(terminal_prices - k, 0.0)
        else:
            payoff = np.maximum(k - terminal_prices, 0.0)

        disc_payoff = np.exp(-rate * maturity) * float(np.mean(payoff))

        logger.debug("Seasonal option %s priced: maturity=%s, spot=%s, k=%s, rate=%s, delta=%s, seasonal_amplitude=%s, seasonal_phase=%s, price=%s",
                    ot, maturity, spot, k, rate, delta, seasonal_amplitude, seasonal_phase, disc_payoff)
        return disc_payoff

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger.info("HP-commodities module ran")
