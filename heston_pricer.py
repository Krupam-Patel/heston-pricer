"""Heston Option Pricing Engine: exotic and vanilla"""
import numpy as np

class Pricer:
    def __init__(self, model):
        self.model = model


    def european(self, T, S0, r, q, K, type='Call'):
        call_price = self.model.heston_call(T, S0, r, q, K)
        price = call_price if type == 'Call' else call_price - S0*np.exp(-q*T) + K*np.exp(-r*T)
        return price[0]


    def digital(self, T, S0, r, q, K, type='Call', npaths=250000, seed=None):
        S_t, _ = self.model.simulate(S0, T, r, q, npaths=npaths, nsteps=252, seed=seed)
        S_T = S_t[-1, :]

        if type == 'Call':
            price = np.exp(-r*T) * np.mean(S_T > K)
        else:
            price = np.exp(-r*T) * np.mean(S_T < K)
        return price


    def barrier(self, T, S0, r, q, K, B, barrier_type='UpAndOut', option_type='Call',
                npaths=250000, nsteps=365, seed=None):
        S_t, _ = self.model.simulate(S0, T, r, q, npaths=npaths, nsteps=nsteps, seed=seed)

        if barrier_type in ['UpAndOut', 'UpAndIn']:
            barrier_hit = np.any(S_t >= B, axis=0)
        elif barrier_type in ['DownAndOut', 'DownAndIn']:
            barrier_hit = np.any(S_t <= B, axis=0)
        else:
            raise ValueError("Invalid barrier type.")

        if option_type == 'Call':
            payoff = np.maximum(S_t[-1, :] - K, 0)
        elif option_type == 'Put':
            payoff = np.maximum(K - S_t[-1, :], 0)
        else:
            raise ValueError("Invalid option type.")

        if barrier_type.endswith('Out'):
            payoff[barrier_hit] = 0
        else:
            payoff[~barrier_hit] = 0

        price = np.exp(-r*T) * np.mean(payoff)
        return price
