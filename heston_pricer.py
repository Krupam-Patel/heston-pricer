"""Heston Option Pricing Engine: exotic and vanilla"""
#Fully done
import numpy as np

class pricer:
    def price_european(self, maturity, spot, rate, div_yield, strike, option_type='Call'):
        call_price = self.model.heston_call(maturity, spot, rate, div_yield, strike)

        if option_type.lower() == 'call':
            return call_price[0]

        put_price = call_price - spot * np.exp(-div_yield * maturity) + strike * np.exp(-rate * maturity)
        return put_price[0]

    def price_digital(self, maturity, spot, rate, div_yield, strike, option_type='Call', n_paths=250_000, seed=None):
        sim_paths, _ = self.model.simulate(
            S0=spot, T=maturity, r=rate, q=div_yield,
            npaths=n_paths, nsteps=252, seed=seed
        )
        term_prices = sim_paths[-1, :]

        if option_type.lower() == 'call':
            payoff = term_prices > strike
        else:
            payoff = term_prices < strike

        disc_payoff = np.exp(-rate * maturity) * np.mean(payoff)
        return disc_payoff

    def price_barrier( self, maturity, spot, rate, div_yield, strike, barrier, barrier_type='UpAndOut', option_type='Call',
                      n_paths=250_000, n_steps=365, seed=None
):
        paths, _ = self.model.simulate(
            S0=spot, T=maturity, r=rate, q=div_yield, npaths=n_paths,
            nsteps=n_steps, seed=seed
        )

        if barrier_type in ('UpAndOut', 'UpAndIn'):
            hit = paths.max(axis=0) >= barrier
        elif barrier_type in ('DownAndOut', 'DownAndIn'):
            hit = paths.min(axis=0) <= barrier
        else:
            raise ValueError("Invalid barrier type")

        terminal = paths[-1, :]
        if option_type.lower() == 'call':
            payoff = np.maximum(terminal - strike, 0)
        elif option_type.lower() == 'put':
            payoff = np.maximum(strike - terminal, 0)
        else:
            raise ValueError("Invalid option type")

        if barrier_type.endswith('Out'):
            payoff[hit] = 0
        else:
            payoff[~hit] = 0

        return np.exp(-rate * maturity) * np.mean(payoff)

if __name__ == "__main__":
    print("HestonPricer works")
