"""Heston Option Pricing Engine: exotic and vanilla"""
import numpy as np

class Pricer:
    """
    Pricing engine for European, digital, and barrier options using a model
    that provides pricing and simulation methods (e.g., Heston model).
    """

    def __init__(self, model):
        """
        Store the model instance used for pricing

        Parameters
        ----------
        model: object
            Model with methods: heston_call(), simulate(), etc
        """
        self.model = model


    def european(self, T, S0, r, q, K, type='Call'):
        """
        Price a European call or put using the model's closed-form Heston method

        Parameters
        ----------
        T: float
            Maturity
        S0: float
            Spot price
        r: float
            Risk-free rate
        q: float
            Dividend yield
        K: float or ndarray
            Strike(s)
        type: {'Call','Put'}
            Option type

        Returns
        -------
        price: float
            European option price
        """
        call_price = self.model.heston_call(T, S0, r, q, K)
        # Put–call parity adjustment for put price
        price = call_price if type == 'Call' else call_price - S0*np.exp(-q*T) + K*np.exp(-r*T)
        return price[0]


    def digital(self, T, S0, r, q, K, type='Call', npaths=250000, seed=None):
        """
        Price a European digital option using Monte Carlo simulation.

        Parameters
        ----------
        T: float
            Maturity
        S0: float
            Spot price
        r: float
            Risk-free rate
        q: float
            Dividend yield
        K: float
            Strike
        type: {'Call','Put'}
            Digital payoff direction
        npaths: int
            Number of simulated paths
        seed: int
            RNG seed

        Returns
        -------
        price: float
            Digital option price
        """
        S_t, _ = self.model.simulate(S0, T, r, q, npaths=npaths, nsteps=252, seed=seed)
        S_T = S_t[-1, :]

        # Binary payoff under risk-neutral discounting
        if type == 'Call':
            price = np.exp(-r*T) * np.mean(S_T > K)
        else:
            price = np.exp(-r*T) * np.mean(S_T < K)

        return price


    def barrier(self, T, S0, r, q, K, B, barrier_type='UpAndOut', option_type='Call',
                npaths=250000, nsteps=365, seed=None):
        """
        Price barrier options using Monte Carlo simulation

        Parameters
        ----------
        T: float
            Maturity
        S0: float
            Spot price
        r: float
            Risk-free rate
        q: float
            Dividend yield
        K: float
            Strike
        B: float
            Barrier level
        barrier_type: {'UpAndOut','DownAndOut','UpAndIn','DownAndIn'}
            Barrier direction and knock-in/knock-out type
        option_type: {'Call','Put'}
            Payoff direction
        npaths: int
            Number of simulated paths
        nsteps: int
            Time discretization
        seed: int
            RNG seed

        Returns
        -------
        price: float
            Barrier option price
        """
        S_t, _ = self.model.simulate(S0, T, r, q, npaths=npaths, nsteps=nsteps, seed=seed)

        # Barrier monitoring along each path
        if barrier_type in ['UpAndOut', 'UpAndIn']:
            barrier_hit = np.any(S_t >= B, axis=0)
        elif barrier_type in ['DownAndOut', 'DownAndIn']:
            barrier_hit = np.any(S_t <= B, axis=0)
        else:
            raise ValueError("Invalid barrier type.")

        # European payoff at maturity
        if option_type == 'Call':
            payoff = np.maximum(S_t[-1, :] - K, 0)
        elif option_type == 'Put':
            payoff = np.maximum(K - S_t[-1, :], 0)
        else:
            raise ValueError("Invalid option type.")

        # Apply barrier condition
        if barrier_type.endswith('Out'):
            payoff[barrier_hit] = 0
        else:  # Knock-in
            payoff[~barrier_hit] = 0

        price = np.exp(-r*T) * np.mean(payoff)
        return price
