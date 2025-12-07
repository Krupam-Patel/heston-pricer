import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from heston_model import HestonModel
from heston_calibration import calibrate, bs_implied_vol, get_vol_slice
from heston_pricer import Pricer


warnings.filterwarnings("ignore")


def main():
    # -------------------------------
    # 1. Calibration on SPY surface
    # -------------------------------

    # Initial Heston parameters (starting guess)
    init_params = {
        "kappa": 1.0,
        "theta": 0.04,
        "xi": 0.4,
        "rho": -0.7,
        "v0": 0.05,
    }
    model = HestonModel(init_params)

    # Input data: SPY-style calibration template
    file = r"SPY_Calibration_Template.xlsx"
    market_data = pd.read_excel(file, sheet_name="Market_Data")
    surf = pd.read_excel(file, sheet_name="Vol_Matrix", index_col=0)

    # Choose calibration maturity (in years)
    T = 1.0  # 1-year maturity

    # Risk-free rate curve
    rate_curve = (
        market_data[["Year_Frac", "Risk_Free_Rate"]]
        .drop_duplicates()
        .sort_values("Year_Frac")
    )
    times_rf = rate_curve["Year_Frac"].values
    rates = rate_curve["Risk_Free_Rate"].values
    r = np.interp(T, times_rf, rates)

    # Dividend yield curve
    div_curve = (
        market_data[["Year_Frac", "Div_Yield"]]
        .drop_duplicates()
        .sort_values("Year_Frac")
    )
    times_div = div_curve["Year_Frac"].values
    divs = div_curve["Div_Yield"].values
    q = np.interp(T, times_div, divs)

    # Spot price
    S0 = float(market_data["S0"].iloc[0])

    # Calibration
    res = calibrate(model, surf, S0, r, T, q)

    # -------------------------------
    # 2. Volatility Smiles (Heston vs Market)
    # -------------------------------

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Maturities to visualize
    T_values = [0.5, 1.0, 1.5, 2.0]

    for i, T_i in enumerate(T_values):
        moneyness, market_vols = get_vol_slice(surf, T_i)
        K_vals = moneyness * S0 / 100.0

        # Heston call prices at these strikes
        heston_prices = model.heston_call(T_i, S0, r, q, K_vals)

        # Convert Heston prices to implied vols via Black–Scholes
        heston_vols = [
            bs_implied_vol(S0, K, T_i, r, q, C)
            for K, C in zip(K_vals, heston_prices)
        ]

        ax = axes[i]
        ax.plot(moneyness, heston_vols, "o-", label="Heston Implied Vols")
        ax.plot(moneyness, market_vols, "x-", label="Market Implied Vols")
        ax.set_xlabel("Moneyness (%)")
        ax.set_ylabel("Implied Volatility")
        ax.set_title(f"Volatility Smile T = {T_i}")
        ax.legend()
        ax.grid(True)

    fig.suptitle(
        f"Heston vs Market Volatility Smiles (Calibration T = {T})",
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

    # -------------------------------
    # 3. Volatility Process Simulation
    # -------------------------------

    S_paths, v_paths = model.simulate(S0, T, r, q, npaths=1000)
    plt.figure(figsize=(8, 5))
    plt.plot(np.sqrt(v_paths))
    plt.xlabel("Time step")
    plt.ylabel("Volatility")
    plt.title("Simulated Volatility Paths (Heston)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -------------------------------
    # 4. Call Prices Check: MC vs Heston vs Carr–Madan
    # -------------------------------

    K_grid = np.linspace(0.5 * S0, 2.0 * S0, 20)

    mc_prices = model.monte_carlo_call(T, S0, r, q, K_grid)
    heston_prices_grid = model.heston_call(T, S0, r, q, K_grid)
    fft_prices = model.carr_madan_call(T, S0, r, q, K_grid)

    max_rel_err = np.max(np.abs(fft_prices / heston_prices_grid - 1.0))
    print(
        f"Max relative error between Carr–Madan FFT and Heston semi-analytic: "
        f"{max_rel_err:.4e}"
    )

    plt.figure(figsize=(8, 5))
    plt.plot(K_grid, mc_prices, "o-", label="Monte Carlo")
    plt.plot(K_grid, heston_prices_grid, "s-", label="Heston (P1/P2)")
    plt.plot(K_grid, fft_prices, "^-", label="Carr–Madan FFT")
    plt.xlabel("Strike K")
    plt.ylabel("Call Price")
    plt.title("Call Prices under Heston Model (SPY Surface)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -------------------------------
    # 5. Pricing Examples with Pricer
    # -------------------------------

    K = 200.0      # strike
    B = 150.0      # barrier level
    n = 1000       # notional multiplier
    seed = 2025
    paths = 10**5
    steps = 1000

    pricer = Pricer(model)

    # Example MTMs below are placeholders; replace with real SPY quotes if you have them

    # European Call
    mtm = 21_522.81
    call_price = pricer.european(T, S0, r, q, K, type="Call") * n
    print(
        f"European Call Price: {call_price:.2f}; "
        f"Relative error: {(call_price / mtm - 1):.3%}"
    )

    # European Put
    mtm = 26_149.40
    put_price = pricer.european(T, S0, r, q, K, type="Put") * n
    print(
        f"European Put Price: {put_price:.2f}; "
        f"Relative error: {(put_price / mtm - 1):.3%}"
    )

    # Digital Call
    mtm = 395.72
    digital_call_price = (
        pricer.digital(T, S0, r, q, K, type="Call", npaths=paths, seed=seed) * n
    )
    print(
        f"Digital Call Price: {digital_call_price:.2f}; "
        f"Relative error: {(digital_call_price / mtm - 1):.3%}"
    )

    # Digital Put
    mtm = 571.86
    digital_put_price = (
        pricer.digital(T, S0, r, q, K, type="Put", npaths=paths, seed=seed) * n
    )
    print(
        f"Digital Put Price: {digital_put_price:.2f}; "
        f"Relative error: {(digital_put_price / mtm - 1):.3%}"
    )

    # Barrier Up-and-In Call
    mtm = 21_502.32
    barrier_type = "UpAndIn"
    barrier_price = (
        pricer.barrier(
            T,
            S0,
            r,
            q,
            K,
            B,
            barrier_type=barrier_type,
            option_type="Call",
            npaths=paths,
            nsteps=steps,
            seed=seed,
        )
        * n
    )
    print(
        f"Barrier {barrier_type} Price: {barrier_price:.2f}; "
        f"Relative error: {(barrier_price / mtm - 1):.3%}"
    )

    # Barrier Up-and-Out Call
    mtm = 0.00  # placeholder
    barrier_type = "UpAndOut"
    barrier_price = (
        pricer.barrier(
            T,
            S0,
            r,
            q,
            K,
            B,
            barrier_type=barrier_type,
            option_type="Call",
            npaths=paths,
            nsteps=steps,
            seed=seed,
        )
        * n
    )
    print(
        f"Barrier {barrier_type} Price: {barrier_price:.2f}; "
        f"Relative error: "
        f"{(barrier_price / mtm - 1):.3% if mtm != 0 else float('nan')}%"
    )

    # Barrier Down-and-Out Call
    mtm = 20_164.98
    barrier_type = "DownAndOut"
    barrier_price = (
        pricer.barrier(
            T,
            S0,
            r,
            q,
            K,
            B,
            barrier_type=barrier_type,
            option_type="Call",
            npaths=paths,
            nsteps=steps,
            seed=seed,
        )
        * n
    )
    print(
        f"Barrier {barrier_type} Price: {barrier_price:.2f}; "
        f"Relative error: {(barrier_price / mtm - 1):.3%}"
    )

    # Barrier Down-and-In Call
    mtm = 1_362.50
    barrier_type = "DownAndIn"
    barrier_price = (
        pricer.barrier(
            T,
            S0,
            r,
            q,
            K,
            B,
            barrier_type=barrier_type,
            option_type="Call",
            npaths=paths,
            nsteps=steps,
            seed=seed,
        )
        * n
    )
    print(
        f"Barrier {barrier_type} Price: {barrier_price:.2f}; "
        f"Relative error: {(barrier_price / mtm - 1):.3%}"
    )


if __name__ == "__main__":
    main()
