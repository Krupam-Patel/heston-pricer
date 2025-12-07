import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.optimize import brentq


def calibrate(heston_model, df_surf, S0, r, T, q=0, alpha=1.5, N_fft=4096, eta=0.225):
    """
    Calibrate Heston model parameters to market-implied vols for a single maturity T.
    The function extracts a slice of the implied volatility surface, converts the vols
    into market call prices using Black–Scholes, and then minimizes the squared error
    between market prices and Heston model prices (via Carr–Madan FFT).
    """

    # Extract market moneyness & implied vol slice at maturity T
    # Convert moneyness (%) into absolute strikes K = (moneyness/100) * S0
    moneyness, sigma_market = get_vol_slice(df_surf, T)
    K_market = moneyness * S0 / 100

    # Convert implied vols into market call prices using Black–Scholes
    C_market = bs_call_price(S0, K_market, T, r, q, sigma_market)

    # Initial guess for parameters: [kappa, xi, rho, v0, phi]
    # phi is an auxiliary parameter used to construct theta for enforcing the Feller condition
    x0 = [1.0, 0.4, -0.7, 0.05, 0.2]

    # Parameter bounds to keep optimization in a stable region
    bounds = [
        (0.01, 5),     # kappa
        (0.05, 2),      # xi (vol-of-vol)
        (-0.95, 1),  # rho (correlation)
        (1e-4, 0.5),      # v0 (initial variance)
        (0, 1)          # phi (auxiliary term)
    ]

    K_market = np.atleast_1d(K_market)
    C_market = np.atleast_1d(C_market)
    history = []  # Store (params, error) for debugging or visualization if needed

    # Objective: minimize squared error between market and model call prices
    def objective(params):
        kappa, xi, rho, v0, phi = params

        # Compute theta using auxiliary 'phi' to guarantee Feller condition: 2*kappa*theta ≥ xi^2
        theta = (xi**2 + phi) / (2 * kappa)

        # Update model parameters in-place before pricing
        heston_model.kappa = kappa
        heston_model.theta = theta
        heston_model.xi = xi
        heston_model.rho = rho
        heston_model.v0 = v0

        # Price calls using the Carr–Madan FFT under the current parameter set
        C_model = heston_model.carr_madan_call(
            T, S0, r, q, K_market, alpha=alpha, N=N_fft, eta=eta
        )

        # Sum of squared pricing errors
        error = np.sum((C_model - C_market)**2)
        history.append((params, error))
        return error

    # Run optimization using L-BFGS-B with bounds
    result = minimize(
        objective,
        x0,
        bounds=bounds,
        method='L-BFGS-B',
        options={'disp': True, 'maxiter': 500}
    )

    # If calibration succeeded, update the model with the optimized parameters
    if result.success:
        kappa, xi, rho, v0, phi = result.x
        theta = (xi**2 + phi) / (2 * kappa)

        # Update calibrated parameters
        heston_model.kappa = kappa
        heston_model.theta = theta
        heston_model.xi = xi
        heston_model.rho = rho
        heston_model.v0 = v0

        print("Calibration successful:")
        print(f"Iterations: {result.nit}")
        print(f"Total Error: {result.fun:.5e}.")
        print("\nParameters:")
        print(f"kappa: {kappa:.5f}, theta: {theta:.5f}, xi: {xi:.5f}, "
              f"rho: {rho:.5f}, v0: {v0:.5f}\n")

        # Check the Feller condition for variance positivity
        # Condition: 2*kappa*theta ≥ xi^2
        if 2 * kappa * theta < xi**2:
            print("Warning: Feller condition NOT satisfied!\n")

        # Compare calibrated model prices vs market prices
        C_model = heston_model.carr_madan_call(
            T, S0, r, q, K_market, alpha=alpha, N=N_fft, eta=eta
        )
        print("Market vs Model price differences (%):")
        for k, cm, cm_model in zip(K_market, C_market, C_model):
            dif = (cm_model - cm) / cm * 100
            print(f"Moneyness {100 * k/S0:.2f}%: Difference = {dif:.4f}%")

    else:
        print("Calibration failed:", result.message)

    return result


def bs_call_price(S0, K, T, r, q, sigma):
    """
    Compute the Black–Scholes price of a European call option with continuous dividend yield q.
    Supports vectorized inputs for K and sigma.
    """
    K = np.atleast_1d(K)
    sigma = np.atleast_1d(sigma)

    # Forward price under continuous dividend yield
    F = S0 * np.exp((r - q) * T)

    # Black–Scholes d1, d2 terms
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Discounted expected payoff under risk-neutral measure
    return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))


def bs_implied_vol(S0, K, T, r, q, market_price):
    """
    Compute the implied volatility using Brent's root-finding method.
    Solves for sigma such that Black–Scholes price(sigma) = market_price.
    Returns NaN if no implied vol can be found in the search interval.
    """

    def objective(sigma):
        return bs_call_price(S0, K, T, r, q, sigma) - market_price

    try:
        return brentq(objective, 1e-6, 5.0)  # Solve for sigma in a reasonable range
    except ValueError:
        return np.nan  # No implied volatility exists


def get_vol_slice(df_vol, T):
    """
    Extract and interpolate market implied volatilities for a given maturity T.
    
    Parameters
    ----------
    df_vol : pandas.DataFrame
        Volatility surface with maturities as the index (year fractions)
        and moneyness/strike levels as the columns.
    T : float
        Target maturity in years.

    Returns
    -------
    moneyness : ndarray
        Moneyness or strikes expressed as a percentage of spot (e.g., 80, 90, 100, 110).
    sigma_market : ndarray
        Interpolated implied volatilities at maturity T for each moneyness level.
    """

    df = df_vol.copy()

    # Extract array of maturities and moneyness values
    maturities = df.index.to_numpy().astype(float)
    moneyness = df.columns.to_numpy().astype(float)
    sigma_market = []

    # Interpolate vol term structure for each moneyness level independently
    for k in moneyness:
        vols_k = df[k].values.astype(float)
        f = interp1d(
            maturities, vols_k,
            kind='linear',
            fill_value="extrapolate"  # Allow extrapolation if T is outside data range
        )
        sigma_market.append(f(T))

    sigma_market = np.array(sigma_market)

    # Return data sorted by strike/moneyness
    sort_idx = np.argsort(moneyness)
    return moneyness[sort_idx], sigma_market[sort_idx]

