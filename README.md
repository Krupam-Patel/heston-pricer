# Heston-Pricer

A Heston pricing engine for valuing vanilla and exotic equity options. This repository implements the Heston stochastic volatility model, a widely used framework in quantitative finance where volatility follows a mean-reverting square-root process. The model incorporates realistic features such as volatility clustering and correlation between price and volatility.

The implementation provides fast and accurate pricing for vanilla options (standard calls and puts) using the semi-closed-form Heston characteristic function, and can be extended to price more complex exotic derivatives, including barriers, digitals, and other path-dependent structures.

## Features

- Fast and accurate pricing for vanilla options (calls and puts)
- Built-in support for selected exotic structures (digital and barrier options via Monte Carlo simulation)
- Model calibration to market option data using a Carr–Madan FFT-based approach

## Main Python Components

### `HestonModel` (in `heston_model.py`)

Represents the Heston stochastic volatility model used for equity-derivatives pricing. This class provides:

- Construction and storage of model parameters (`kappa`, `theta`, `xi`, `rho`, `v0`).
- Simulation of joint underlying-asset and variance paths under the risk-neutral measure.
- Calculation of the Heston characteristic function for the log-asset price.
- Analytical and numerical pricing of European call options using several approaches:
  - Direct numerical integration of the Heston semi-closed-form solution
  - Fast Fourier Transform (Carr–Madan) for efficient pricing across strikes
  - Monte Carlo simulation of the stochastic volatility dynamics

#### Key Methods

- `simulate(S0, T, r, q, npaths, nsteps, seed)`:  
  Simulates paths for the underlying asset price and variance under the Heston dynamics.

- `heston_cf(u, T, S0, r, q)`:  
  Computes the Heston characteristic function, which is used in Fourier-based pricing methods.

- `heston_call(...)`:  
  Prices European call options using the semi-analytical Heston formula based on the model’s characteristic function (direct numerical integration).

- `carr_madan_call(...)`:  
  Implements the Carr–Madan FFT approach to price European options efficiently across a range of strikes using Fourier inversion.

- `monte_carlo_call(...)`:  
  Uses Monte Carlo simulation of the Heston stochastic volatility process to estimate European option prices, with flexibility for extensions to path-dependent payoffs.

---

### `Pricer` (in `heston_pricer.py`)

A high-level wrapper for pricing vanilla and exotic options using an instance of `HestonModel`. This class provides:

- User-friendly methods to price European, digital, and barrier options.
- Integration with model calibration and simulation routines, choosing the appropriate numerical method under the hood.

#### Key Methods

- `european(T, S0, r, q, K, type)`:  
  Prices standard European options (calls or puts) using the Heston model.

- `digital(T, S0, r, q, K, type, npaths)`:  
  Prices digital (binary) options using Monte Carlo simulation of the Heston dynamics.

- `barrier(T, S0, r, q, K, B, type, npaths, nsteps, seed)`:  
  Prices barrier options (e.g., knock-in/knock-out) using path simulation under the Heston model.

---

### Calibration Functions (in `heston_calibration.py`)

The main calibration routines for fitting Heston model parameters to market option data.

- Use market implied volatilities (from a volatility surface) and convert them to prices via the Black–Scholes formula.
- Fit the Heston parameters (`kappa`, `theta`, `xi`, `rho`, `v0`) by minimizing the squared difference between market prices and Heston model prices produced by the Carr–Madan FFT method.
- Automatically check the Feller condition for variance process positivity and flag violations.
- Provide detailed output about the calibration process, including estimated parameters, objective value (error), and price differences.

#### Key Methods / Functions

- `calibrate(heston_model, df_surf, S0, r, T, q, ...)`:  
  Fits model parameters to market data for a given maturity slice using least-squares minimization.

- `bs_call_price(...)`:  
  Computes Black–Scholes call prices used to convert implied volatilities into option prices.

- `get_vol_slice(...)`:  
  Extracts and interpolates implied volatilities for a given maturity from a volatility surface DataFrame.

## Example Usage

See `Example.ipynb` for demonstrations of model simulation, calibration, and option pricing using the classes and functions described above.
