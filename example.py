"""
Heston Pricing Example – SPY Surface
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

from heston_model import HestonModel
from heston_calibration import calibrate, bs_implied_vol, get_vol_slice
from heston_pricer import Pricer

try:
    plt.style.use("seaborn-v0_8")
except Exception:
    plt.style.use("seaborn")

mpl.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "axes.grid": True,
    "axes.grid.which": "both",
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "legend.frameon": False,
    "lines.linewidth": 2.2,
    "lines.markersize": 6,
})

def main():
    init_params = dict(
        kappa=1.0,
        theta=0.04,
        xi=0.4,
        rho=-0.7,
        v0=0.05,
    )
    model = HestonModel(init_params)

    file = "SPY_Calibration_Template.xlsx"
    market_data = pd.read_excel(file, sheet_name="Market_Data")
    surf = pd.read_excel(file, sheet_name="Vol_Matrix", index_col=0)

    T = 1.0

    rate_curve = (
        market_data[["Year_Frac", "Risk_Free_Rate"]]
        .drop_duplicates()
        .sort_values("Year_Frac")
    )
    r = np.interp(T, rate_curve["Year_Frac"], rate_curve["Risk_Free_Rate"])

    div_curve = (
        market_data[["Year_Frac", "Div_Yield"]]
        .drop_duplicates()
        .sort_values("Year_Frac")
    )
    q = np.interp(T, div_curve["Year_Frac"], div_curve["Div_Yield"])

    S0 = float(market_data["S0"].iloc[0])

    print("Calibrating Heston model...")
    res = calibrate(model, surf, S0, r, T, q)
    print("Calibration result:", res)

    fig, ax = plt.subplots()
    im = ax.imshow(
        surf.values,
        aspect="auto",
        cmap="viridis",
        origin="lower",
    )
    fig.colorbar(im, ax=ax, label="Implied Volatility")

    ax.set_xticks(range(len(surf.columns)))
    ax.set_xticklabels(surf.columns.astype(str), rotation=45, ha="right")

    ax.set_yticks(range(len(surf.index)))
    ax.set_yticklabels(surf.index.astype(str))

    ax.set_xlabel("Moneyness (%)")
    ax.set_ylabel("Maturity (Years)")
    ax.set_title("SPY Implied Volatility Surface (Heatmap)")

    plt.tight_layout()
    plt.show()

    X = surf.columns.astype(float)
    Y = surf.index.astype(float)
    X_grid, Y_grid = np.meshgrid(X, Y)
    Z = surf.values

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf_plot = ax.plot_surface(
        X_grid,
        Y_grid,
        Z,
        cmap="viridis",
        edgecolor="none",
        antialiased=True,
    )

    ax.set_title("SPY 3D Implied Volatility Surface")
    ax.set_xlabel("Moneyness (%)")
    ax.set_ylabel("Maturity (Years)")
    ax.set_zlabel("Implied Volatility")

    fig.colorbar(surf_plot, shrink=0.6, aspect=12, pad=0.08)
    ax.view_init(elev=25, azim=-135)

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    T_values = [0.5, 1.0, 1.5, 2.0]

    for i, T_i in enumerate(T_values):
        mny, market_vols = get_vol_slice(surf, T_i)
        K_vals = mny * S0 / 100

        h_prices = model.heston_call(T_i, S0, r, q, K_vals)
        h_iv = [
            bs_implied_vol(S0, K, T_i, r, q, C)
            for K, C in zip(K_vals, h_prices)
        ]

        ax = axes[i]
        ax.plot(mny, market_vols, "o-", label="Market IV")
        ax.plot(mny, h_iv, "s--", label="Heston IV")

        ax.set_title(f"T = {T_i} years")
        ax.set_xlabel("Moneyness (%)")
        ax.set_ylabel("Implied Volatility")
        ax.legend()

    fig.suptitle("Heston vs Market Volatility Smiles", fontsize=14, weight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.show()

    mny_T, mkt_T = get_vol_slice(surf, T)
    K_T = mny_T * S0 / 100

    h_prices_T = model.heston_call(T, S0, r, q, K_T)
    h_iv_T = [
        bs_implied_vol(S0, K, T, r, q, C)
        for K, C in zip(K_T, h_prices_T)
    ]

    fig, ax = plt.subplots()

    ax.plot(mny_T, mkt_T, "o-", label="Market Smile")
    ax.plot(mny_T, h_iv_T, "s--", label="Heston Smile")

    ax.fill_between(mny_T, mkt_T, h_iv_T, alpha=0.25, label="Error Region")

    ax.set_title(f"Volatility Smile Comparison (T = {T} years)")
    ax.set_xlabel("Moneyness (%)")
    ax.set_ylabel("Implied Volatility")
    ax.legend()

    plt.tight_layout()
    plt.show()

    S_paths, v_paths = model.simulate(S0, T, r, q, npaths=1000)

    n_plot = 40
    fig, ax = plt.subplots()
    ax.plot(np.sqrt(v_paths[:, :n_plot]), alpha=0.35)

    ax.set_title("Simulated Volatility Paths (Heston)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Volatility")

    plt.tight_layout()
    plt.show()

    subset = np.sqrt(v_paths[:, :300])
    p10 = np.percentile(subset, 10, axis=1)
    p50 = np.percentile(subset, 50, axis=1)
    p90 = np.percentile(subset, 90, axis=1)

    t_idx = np.arange(len(p10))

    fig, ax = plt.subplots()
    ax.fill_between(t_idx, p10, p90, alpha=0.3, label="10–90% Range")
    ax.plot(t_idx, p50, color="black", label="Median Vol")

    ax.set_title("Heston Volatility Distribution Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Volatility")
    ax.legend()

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    ax.hist(S_paths[-1], bins=60, density=True, alpha=0.75)

    ax.set_title("Distribution of Terminal Prices S(T)")
    ax.set_xlabel("S(T)")
    ax.set_ylabel("Density")

    plt.tight_layout()
    plt.show()

    K_grid = np.linspace(0.5 * S0, 2.0 * S0, 200)

    mc = model.monte_carlo_call(T, S0, r, q, K_grid)
    cf = model.heston_call(T, S0, r, q, K_grid)
    fft = model.carr_madan_call(T, S0, r, q, K_grid)

    max_err = np.max(np.abs(fft / cf - 1))
    print("Max relative error (FFT vs Closed-form):", max_err)

    fig, ax = plt.subplots()
    ax.plot(K_grid, cf, label="Closed Form")
    ax.plot(K_grid, fft, "--", label="Carr–Madan FFT")
    ax.plot(K_grid, mc, "o", alpha=0.65, ms=3, label="Monte Carlo")

    ax.set_title("Call Prices under Heston Model")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Call Price")
    ax.legend()

    plt.tight_layout()
    plt.show()

    pricer = Pricer(model)
    K = 200.0
    B = 150.0
    n = 1000
    seed = 2025
    paths = 10**5
    steps = 1000

    print("\n--- Pricing Examples (Heston Pricer) ---")

    mtm = 21522.81
    call_price = pricer.european(T, S0, r, q, K, type="Call") * n
    print(f"European Call: {call_price:.2f}, Rel. error vs MTM: {call_price/mtm - 1:.3%}")

    mtm = 26149.40
    put_price = pricer.european(T, S0, r, q, K, type="Put") * n
    print(f"European Put:  {put_price:.2f}, Rel. error vs MTM: {put_price/mtm - 1:.3%}")

    mtm = 395.72
    digital_call = pricer.digital(T, S0, r, q, K, type="Call", npaths=paths, seed=seed) * n
    print(f"Digital Call:  {digital_call:.2f}, Rel. error vs MTM: {digital_call/mtm - 1:.3%}")

    mtm = 571.86
    digital_put = pricer.digital(T, S0, r, q, K, type="Put", npaths=paths, seed=seed) * n
    print(f"Digital Put:   {digital_put:.2f}, Rel. error vs MTM: {digital_put/mtm - 1:.3%}")

if __name__ == "__main__":
    main()
