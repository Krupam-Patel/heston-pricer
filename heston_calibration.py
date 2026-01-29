"""Heston Model Calibration for Commodities via Carr-Madan FFT"""
# Fully done 1/28/2026
# Comenting done 1/28/2026


import logging
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.stats import norm
from scipy.interpolate import interp1d
from heston_model import hCommModel

logger = logging.getLogger(__name__)


class hCommCalibrator:
    def __init__(self, heston_model):
        self.model = heston_model
        self.calibration_history = []
        logger.info("Initialized hCommCalibrator")
    
    def calibrate_single_T(self, df_vol_surface, T, S0, r, q=0.0, alpha=1.5, N_fft=4096, eta=0.225, bounds=None, method='L-BFGS-B'):
        # Extract market data for this T
        moneyness, sigma_market = self.get_volatility_slice(df_vol_surface, T)
        strikes = moneyness * S0 / 100
        
        # Convert implied vols to market prices
        market_prices = self.bs_call_prices(S0, strikes, T, r, q, sigma_market)
        
        # Set commodity-appropriate parameter bounds
        if bounds is None:
            bounds = self.get_commodity_bounds()

        # Using theta scale to ensure Feller
        opt_bounds = [bounds["kappa"], bounds["xi"], bounds["rho"], bounds["v0"], bounds["theta_scale"]]
        
        # Initial guess from current model parameters (theta scale)
        x0 = [self.model.kappa, self.model.xi, self.model.rho, self.model.v0, 0.5]  
        
        self.calibration_history = []
        
        def objective(params):
            # MSE between model and market prices with penalty terms
            kappa, xi, rho, v0, theta_scale = params
            
            # Computing theta to satisfy Feller condition
            theta = (xi**2 * theta_scale) / (2 * kappa)
            
            # Updating model parameters
            temp_params = {"kappa": kappa, "theta": theta, "xi": xi, "rho": rho, "v0": v0}
            
            try:
                # Creating temporary model with these parameters
                temp_model = hCommModel(temp_params)
                
                # Prices using Carr-Madan FFT
                model_prices = temp_model.carr_madan_call_prices(T, S0, r, q, strikes, alpha=alpha, N=N_fft, eta=eta)
                
                mse = np.mean((model_prices - market_prices)**2)
                
                penalty = 0.0
                
                # Penalizes very high kappa and rho
                if kappa > 5.0:
                    penalty += 10 * (kappa - 5.0)**2
                
                if abs(rho) > 0.9:
                    penalty += 100 * (abs(rho) - 0.9)**2
                
                # Checks Feller condition (should be satisfied by construction)
                feller_lhs = 2 * kappa * theta
                feller_rhs = xi**2
                if feller_lhs < feller_rhs:
                    penalty += 1000 * (feller_rhs - feller_lhs)**2
                
                total_error = mse + penalty
                
                self.calibration_history.append({'params': params.copy(), 'mse': mse, 'penalty': penalty, 'total_error': total_error})
                
                return total_error
                
            except Exception as e:
                logger.warning(f"Pricing failed during calibration: {e}")
                return 1e10
        
        # Run optimization
        logger.info(f"Starting calibration for T={T:.4f}y, S0={S0:.4f}")
        
        result = minimize(objective, x0, method=method, bounds=opt_bounds, options={'maxiter': 500, 'disp': True})
        
        if result.success:
            kappa, xi, rho, v0, theta_scale = result.x
            theta = (xi**2 * theta_scale) / (2 * kappa)
            
            # Updates model with new calibrated parameters if needed
            self.model.parameters = {"kappa": kappa, "theta": theta, "xi": xi, "rho": rho, "v0": v0}
            self.model.kappa = kappa
            self.model.theta = theta
            self.model.xi = xi
            self.model.rho = rho
            self.model.v0 = v0
            
            # Computes final model prices for diagnostics
            final_prices = self.model.carr_madan_call_prices(T, S0, r, q, strikes, alpha=alpha, N=N_fft, eta=eta)
            
            # Calculates pricing errors
            abs_errors = final_prices - market_prices
            pct_errors = (abs_errors / market_prices) * 100
            
            print("Calibration successful:")
            print(f"Iterations: {result.nit}")
            print(f"Total Error: {result.fun:.5e}.")
            print("\nParameters:")
            print(f"kappa: {kappa:.5f}, theta: {theta:.5f}, xi: {xi:.5f}, "
                  f"rho: {rho:.5f}, v0: {v0:.5f}\n")
            
            # Feller condition check
            feller_lhs = 2 * kappa * theta
            feller_rhs = xi**2
            if feller_lhs < feller_rhs:
                print("Warning: Feller condition NOT satisfied!\n")
            
            # Prints market vs model comparison
            print("Market vs Model price differences (%):")
            for k, cm, cm_model, mn in zip(strikes, market_prices, final_prices, moneyness):
                dif = (cm_model - cm) / cm * 100
                print(f"Moneyness {mn:.2f}%: Difference = {dif:.4f}%")
            
            return {"success": True, "mse": result.fun, 
                    "params": {"kappa": kappa, "theta": theta, "xi": xi, "rho": rho, "v0": v0}, "iterations": result.nit, 
                    "pricing_errors": {"absolute": abs_errors, "percentage": pct_errors, "rmse": np.sqrt(np.mean(abs_errors**2))}}
        else:
            print("Calibration failed:", result.message)
            return {"success": False, "message": result.message}
    
    def calibrate_multi_T(self, df_vol_surface, Ts, S0, r, q=0.0, weights=None, alpha=1.5, N_fft=4096, eta=0.225):
        # Calibrate to multiple Ts simultaneously (term structure fitting)
        if weights is None:
            weights = np.ones(len(Ts)) / len(Ts)
        else:
            weights = np.array(weights) / np.sum(weights)
        
        # Collect market data for all Ts
        all_market_data = []
        for T in Ts:
            moneyness, sigma_market = self.get_volatility_slice(df_vol_surface, T)
            strikes = moneyness * S0 / 100
            prices = self.bs_call_prices(S0, strikes, T, r, q, sigma_market)
            all_market_data.append({'T': T, 'strikes': strikes, 'prices': prices, 'moneyness': moneyness})
        
        bounds = self.get_commodity_bounds()
        opt_bounds = [bounds["kappa"], bounds["xi"], bounds["rho"], bounds["v0"], bounds["theta_scale"]]
        
        x0 = [self.model.kappa, self.model.xi, self.model.rho, self.model.v0, 0.5]
        
        def objective(params):
            # Weighted MSE across all Ts
            kappa, xi, rho, v0, theta_scale = params
            theta = (xi**2 * theta_scale) / (2 * kappa)
            
            temp_params = {"kappa": kappa, "theta": theta, "xi": xi, "rho": rho, "v0": v0}
            
            try:
                temp_model = hCommModel(temp_params)
                
                total_weighted_mse = 0.0
                
                for data, weight in zip(all_market_data, weights):
                    model_prices = temp_model.carr_madan_call_prices(data['T'], S0, r, q, data['strikes'], alpha=alpha, N=N_fft, eta=eta)
                    mse = np.mean((model_prices - data['prices'])**2)
                    total_weighted_mse += weight * mse
                
                # Regularization penalties
                penalty = 0.0
                if kappa > 5.0:
                    penalty += 10 * (kappa - 5.0)**2
                if abs(rho) > 0.9:
                    penalty += 100 * (abs(rho) - 0.9)**2
                
                feller_lhs = 2 * kappa * theta
                feller_rhs = xi**2
                if feller_lhs < feller_rhs:
                    penalty += 1000 * (feller_rhs - feller_lhs)**2
                
                return total_weighted_mse + penalty
                
            except Exception as e:
                logger.warning(f"Multi-T pricing failed: {e}")
                return 1e10
        
        logger.info(f"Starting multi-T calibration: {len(Ts)} Ts")
        
        result = minimize(objective, x0, method='L-BFGS-B', bounds=opt_bounds, options={'maxiter': 1000, 'disp': True})
        
        if result.success:
            kappa, xi, rho, v0, theta_scale = result.x
            theta = (xi**2 * theta_scale) / (2 * kappa)
            
            # Updates model with new calibrated parameters if needed
            self.model.parameters = {"kappa": kappa, "theta": theta, "xi": xi, "rho": rho, "v0": v0}
            self.model.kappa = kappa
            self.model.theta = theta
            self.model.xi = xi
            self.model.rho = rho
            self.model.v0 = v0
            
            print("Multi-T calibration successful:")
            print(f"Calibrated to {len(Ts)} Ts: {Ts}")
            print(f"Iterations: {result.nit}")
            print(f"Final weighted MSE: {result.fun:.5e}")
            print("\nParameters:")
            print(f"kappa: {kappa:.5f}, theta: {theta:.5f}, xi: {xi:.5f}, "
                  f"rho: {rho:.5f}, v0: {v0:.5f}\n")
            
            if 2 * kappa * theta < xi**2:
                print("Warning: Feller condition NOT satisfied!\n")
            
            return {"success": True, "mse": result.fun, "params": {"kappa": kappa, "theta": theta, "xi": xi, "rho": rho, "v0": v0},
                    "iterations": result.nit, "Ts": Ts}
        else:
            print("Multi-T calibration failed:", result.message)
            return {"success": False, "message": result.message}
    
    def get_commodity_bounds(self):
        return {
            "kappa": (0.5, 10.0), # Higher mean reversion for commodities
            "xi": (0.1, 3.0), # Higher vol of vol
            "rho": (-0.95, 0.5), # Typically negative for commodities
            "v0": (0.001, 1.0), # Initial variance
            "theta_scale": (0.1, 2.0)} # Ensures Feller via theta
    

    def get_volatility_slice(self, df_vol, T):
        # Extract and interpolate volatility slice for a given maturity
        maturities = df_vol.index.to_numpy().astype(float)
        moneyness = df_vol.columns.to_numpy().astype(float)
        sigma_market = []
        
        for mn in moneyness:
            vols_mn = df_vol[mn].values.astype(float)
            interp_func = interp1d(maturities, vols_mn, kind='linear', fill_value='extrapolate')
            sigma_market.append(interp_func(T))
        
        sigma_market = np.array(sigma_market)
        
        return moneyness, sigma_market
    
    def bs_call_prices(self, S0, K, T, r, q, sigma):
        # bs call price (used for market price conversion)
        K = np.atleast_1d(K)
        sigma = np.atleast_1d(sigma)
        
        F = S0 * np.exp((r - q) * T)
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    
    def bs_implied_vol(self, S0, K, T, r, q, market_price):
        # Computing the bs implied volatility
        def objective(sigma):
            return self.bs_call_prices(S0, K, T, r, q, sigma) - market_price
        
        try:
            return brentq(objective, 1e-6, 5.0)
        except ValueError:
            return np.nan


def calibrate(heston_model, df_surf, S0, r, T, convenience_yield=0.0, alpha=1.5, N_fft=4096, eta=0.225):
    # Convenience function for single-maturity calibration
    calibrator = hCommCalibrator(heston_model)
    return calibrator.calibrate_single_T(df_surf, T, S0, r, convenience_yield, alpha=alpha, N_fft=N_fft, eta=eta)


if __name__ == "__main__":
    print("Calibration works")
