import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.optimize import brentq

def calibrate(heston_model, df_surf, S0, r, T, q = 0, alpha=1.5, N_fft=4096, eta=0.25):
    """
    Calibrate Heston model parameters to market implied vols for a single maturity.
    Market vols are extracted internally from df_surf and converted to call prices via Black-Scholes.
    """

    # Extract moneyness and vol and convert to prices
    moneyness, sigma_market = get_vol_slice(df_surf, T)
    K_market = moneyness * S0 / 100
    C_market = bs_call_price(S0, K_market, T, r, q, sigma_market)

    # Initial guess and bounds: [kappa, xi, rho, v0, phi]
    x0 = [0.5, 0.3, -0.5, 0.1, 0.2]  
    bounds = [(1e-3, 10), (1e-3, 2), (-0.99, 0.99), (1e-3, 1), (0, 1)]

    K_market = np.atleast_1d(K_market)
    C_market = np.atleast_1d(C_market)
    history = []

    # Objective function for calibration
    def objective(params):
        kappa, xi, rho, v0, phi = params
        theta = (xi**2 + phi) / (2 * kappa)  # Feller guaranteed

        heston_model.kappa = kappa
        heston_model.theta = theta
        heston_model.xi = xi
        heston_model.rho = rho
        heston_model.v0 = v0

        C_model = heston_model.carr_madan_call(T, S0, r, q, K_market, alpha=alpha, N=N_fft, eta=eta)
        error = np.sum((C_model - C_market)**2)   
        history.append((params, error))
        return error

    # Minimization 
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B', options={'disp': True, 'maxiter': 500})

    # Calibration results 
    if result.success:
        kappa, xi, rho, v0, phi = result.x
        theta = (xi**2 + phi) / (2 * kappa)
        heston_model.kappa = kappa
        heston_model.theta = theta
        heston_model.xi = xi
        heston_model.rho = rho
        heston_model.v0 = v0

        print(f"Calibration successful:")
        print(f"Iterations: {result.nit}")
        print(f"Total Error: {result.fun:.5e}.")
        print("\nParameters:")
        print(f"kappa: {kappa:.5f}, theta: {theta:.5f}, xi: {xi:.5f}, "f"rho: {rho:.5f}, v0: {v0:.5f}\n")

        # Feller condition
        if 2 * kappa * theta < xi**2: print("Warning: Feller condition NOT satisfied!\n")

        # Compare model prices vs market prices
        C_model = heston_model.carr_madan_call(T, S0, r, q, K_market, alpha=alpha, N=N_fft, eta=eta)
        print("Market vs Model prices differences (%):")
        for k, cm, cm_model in zip(K_market, C_market, C_model):
            dif = (cm_model - cm)/cm * 100
            print(f"Moneyness {100 * k/S0:.2f}%: Model vs Market difference: {dif:.4f}%")
    else:
        print("Calibration failed:", result.message)

    return result



def bs_call_price(S0, K, T, r, q, sigma):
    """
    Black-Scholes price of a European call with continuous dividend yield.
    """
    K = np.atleast_1d(K)
    sigma = np.atleast_1d(sigma)
    F = S0 * np.exp((r - q) * T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return call_price


def bs_implied_vol(S0, K, T, r, q, market_price):
    """
    Implied volatility via Black-Scholes using Brent's method.
    """

    def objective(sigma):
        price = bs_call_price(S0, K, T, r, q, sigma)
        return price - market_price
    try:
        implied_vol = brentq(objective, 1e-6, 5.0)
    except ValueError:
        implied_vol = np.nan  # If no solution found

    return implied_vol


def get_vol_slice(df_vol, T):
    """
    Extract and interpolate market implied volatilities for a given maturity T.

    Parameters:
    df_vol: pandas DataFrame with volatility surface (strikes as columns, year fractions as index)
    T: float, target maturity in years
    
    Returns:
    moneyness: ndarray of strikes (as percentage of spot)
    sigma_market: ndarray of interpolated market implied volatilities
    """

    df = df_vol.copy()

    # Get maturities and moneynesses 
    maturities = df.index.to_numpy().astype(float)
    moneyness = df.columns.to_numpy().astype(float)
    sigma_market = []

    # Interpolate vols
    for k in moneyness:
        vols_k = df[k].values.astype(float)
        f = interp1d(maturities, vols_k, kind='linear', fill_value="extrapolate")
        sigma_market.append(f(T))

    sigma_market = np.array(sigma_market)

    # Sort by strike
    sort_idx = np.argsort(moneyness)
    return moneyness[sort_idx], sigma_market[sort_idx]
