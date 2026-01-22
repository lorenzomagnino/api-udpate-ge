import os
import sys
import warnings

# from plotly.subplots import make_subplots
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from arch import arch_model
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.feature_selection import RFECV
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import logging
from data_preprocessor import DataProcessor
from gcs_utils import gcs_manager
from helpers import OUTPUT_CONFIG, Visualizer, ensure_directory_exists, log_message

warnings.filterwarnings("ignore", category=UserWarning)
project_dir = os.path.dirname(os.path.abspath("."))
sys.path.append(project_dir)
sys.path.append(os.getcwd())


plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 100)


def setup_output_directories():
    """Create necessary output directories."""
    ensure_directory_exists(OUTPUT_CONFIG["visualization_path"])
    ensure_directory_exists(OUTPUT_CONFIG["results_path"])
    ensure_directory_exists(OUTPUT_CONFIG["testing_path"])
    # ensure_directory_exists(os.path.dirname(OUTPUT_CONFIG['model_save_path']))


def detect_seasonality(ts, max_lag=60):
    """
    Detect seasonal patterns in a time series.

    Parameters
    ----------
    ts : pd.Series
        Time series data to analyze for seasonality
    max_lag : int, optional
        Maximum lag to check for autocorrelation (default: 60)

    Returns
    -------
    list
        List of potential seasonal periods (days) where autocorrelation > 0.2
    """
    # if len(ts) < max_lag * 2:
    #     max_lag = len(ts) // 3

    # fig, axes = plt.subplots(2, 2, figsize=(15, 8))

    # axes[0, 0].plot(ts)
    # axes[0, 0].set_title("Original Series")
    # axes[0, 0].grid(True, alpha=0.3)

    # ts_stationary = make_stationary(ts)[0]
    # axes[0, 1].plot(ts_stationary)
    # axes[0, 1].set_title("Stationary Series")
    # axes[0, 1].grid(True, alpha=0.3)

    # plot_acf(ts, lags=max_lag, ax=axes[1, 0], title="ACF - Original Series")
    # plot_pacf(ts, lags=max_lag, ax=axes[1, 1], title="PACF - Original Series")

    # plt.tight_layout()
    # plt.show()

    autocorr = [ts.autocorr(lag) for lag in range(1, max_lag + 1)]
    seasonal_candidates = []
    for period in [7, 14, 21, 30, 60, 90]:
        if period < len(autocorr) and abs(autocorr[period - 1]) > 0.2:
            seasonal_candidates.append(period)

    # logging.debug("ACF/PACF Interpretation:")
    # logging.debug(
    #     "- ACF lines under shaded region means autocorrelations become statistically insignificant"
    # )
    # logging.debug("- ACF decays slowly, series likely needs differencing")
    # logging.debug(
    #     "- PACF spikes at lag 1 and near 1 suggest strong AR component, may need differencing"
    # )
    # logging.debug(
    #     f"- Potential seasonal periods based on autocorrelations > 0.2: {seasonal_candidates}"
    # )

    return seasonal_candidates


def make_stationary(ts):
    """
    Transform a time series to make it stationary using various transformations.

    Tests different transformations (log, log-diff, diff, double-diff) and returns
    the first one that results in a stationary series according to ADF test.

    Parameters
    ----------
    ts : pd.Series
        Time series data to make stationary

    Returns
    -------
    tuple
        (transformed_series, transform_type) where transform_type is one of:
        'none', 'log', 'log_diff', 'diff', 'diff2'
    """
    adf_orig = adfuller(ts)[1]

    if adf_orig <= 0.05:
        return ts, "none"

    if (ts > 0).all():
        ts_log = np.log(ts)
        adf_log = adfuller(ts_log)[1]
        if adf_log <= 0.05:
            return ts_log, "log"

        ts_log_diff = ts_log.diff().dropna()
        adf_log_diff = adfuller(ts_log_diff)[1]
        if adf_log_diff <= 0.05:
            return ts_log_diff, "log_diff"

    ts_diff = ts.diff().dropna()
    adf_diff = adfuller(ts_diff)[1]
    if adf_diff <= 0.05:
        return ts_diff, "diff"

    ts_diff2 = ts.diff().diff().dropna()
    return ts_diff2, "diff2"


def get_optimal_models(ts, seasonal_periods=[]):
    """
    This function fits the ARIMA and SARIMAX models to the data and returns the optimal models.
    It tests for heteroscedasticity and fits the GARCH model to the data and returns the optimal model.

    Args:
        ts: The time series data to fit the models to.
        seasonal_periods: The seasonal periods to fit the models to.

    Returns:
        ts_final: The final time series data.
        transform: The transform used to make the time series stationary.
        best_arima_order: The optimal ARIMA order.
        best_seasonal: The optimal seasonal order.
        has_hetero: Whether the time series is heteroscedastic.
        best_garch_name: The optimal GARCH model name.
        best_garch_params: The optimal GARCH model parameters.
    """
    ts_final, transform = make_stationary(ts)

    best_aic, best_arima_order, best_seasonal = np.inf, (1, 0, 1), (0, 0, 0, 0)

    for p in range(4):
        for d in range(3):
            for q in range(4):
                try:
                    model = ARIMA(ts_final, order=(p, d, q)).fit()
                    if model.aic < best_aic:
                        best_aic, best_arima_order = model.aic, (p, d, q)
                except:
                    continue

    for s in seasonal_periods:
        if s > len(ts_final) // 3:
            continue
        for P in range(2):
            for D in range(2):
                for Q in range(2):
                    try:
                        model = SARIMAX(
                            ts_final,
                            order=best_arima_order,
                            seasonal_order=(P, D, Q, s),
                        ).fit()
                        if model.aic < best_aic:
                            best_aic = model.aic
                            best_seasonal = (P, D, Q, s)
                    except:
                        continue

    arima_model = ARIMA(ts_final, order=best_arima_order).fit()
    has_hetero = test_heteroscedasticity(arima_model.resid)
    best_garch_name, best_garch_params = None, None

    if has_hetero:
        best_garch_name, best_garch_params = find_best_garch_model(
            ts_final, best_arima_order
        )

    return (
        transform,
        best_arima_order,
        best_seasonal,
        has_hetero,
        best_garch_name,
        best_garch_params,
    )


def test_heteroscedasticity(residuals):
    """
    Test for heteroscedasticity in model residuals using multiple tests.

    Parameters
    ----------
    residuals : pd.Series
        Residuals from a fitted model

    Returns
    -------
    bool
        True if heteroscedasticity is detected (p-value < 0.05 in any test),
        False otherwise
    """
    residuals = residuals.dropna()
    n = len(residuals)
    time_trend = np.arange(n)
    exog = np.column_stack([np.ones(n), time_trend])

    bp_stat, bp_pvalue = het_breuschpagan(residuals, exog)[:2]
    white_stat, white_pvalue = het_white(residuals, exog)[:2]

    lags = min(10, n // 4)
    autocorrs = [residuals.autocorr(k) ** 2 for k in range(1, lags + 1)]
    ljung_box_stat = (
        n * (n + 2) * sum([autocorrs[k - 1] / (n - k) for k in range(1, lags + 1)])
    )
    ljung_box_pvalue = 1 - stats.chi2.cdf(ljung_box_stat, df=lags)

    # logging.debug(f"Breusch-Pagan test: stat={bp_stat:.4f}, p-value={bp_pvalue:.4f}")
    # logging.debug(f"White test: stat={white_stat:.4f}, p-value={white_pvalue:.4f}")
    # logging.debug(
    #     f"Ljung-Box on squared residuals: stat={ljung_box_stat:.4f}, p-value={ljung_box_pvalue:.4f}"
    # )

    return bp_pvalue < 0.05 or white_pvalue < 0.05 or ljung_box_pvalue < 0.05


def find_best_garch_model(ts, arima_order):
    """
    Find the best GARCH model for modeling volatility in ARIMA residuals.

    Tests multiple GARCH variants (GARCH, EGARCH, GJR-GARCH) with different
    distributions (normal, t-distribution) and selects the one with lowest AIC.

    Parameters
    ----------
    ts : pd.Series
        Stationary time series data
    arima_order : tuple
        ARIMA order (p, d, q) used to fit the mean model

    Returns
    -------
    tuple
        (best_model_name, best_params) where best_params is a dict with
        GARCH model parameters
    """
    arima_model = ARIMA(ts, order=arima_order).fit()
    residuals = arima_model.resid

    models = {
        "GARCH(1,1)": {"vol": "GARCH", "p": 1, "q": 1, "dist": "normal"},
        "GARCH(1,2)": {"vol": "GARCH", "p": 1, "q": 2, "dist": "normal"},
        "GARCH(2,1)": {"vol": "GARCH", "p": 2, "q": 1, "dist": "normal"},
        "EGARCH(1,1)": {"vol": "EGARCH", "p": 1, "q": 1, "dist": "normal"},
        "GJR-GARCH(1,1)": {"vol": "GARCH", "p": 1, "o": 1, "q": 1, "dist": "normal"},
        "GARCH-t(1,1)": {"vol": "GARCH", "p": 1, "q": 1, "dist": "t"},
        "EGARCH-t(1,1)": {"vol": "EGARCH", "p": 1, "q": 1, "dist": "t"},
    }

    results = {}
    for name, params in models.items():
        try:
            fit = robust_garch_fit(residuals, **params)
            results[name] = {
                "AIC": fit.aic,
                "BIC": fit.bic,
                "LogLikelihood": fit.loglikelihood,
            }
        except:
            results[name] = {"AIC": np.inf, "BIC": np.inf, "LogLikelihood": -np.inf}

    best_model_name = min(results.keys(), key=lambda x: results[x]["AIC"])
    best_params = models[best_model_name]

    results_df = pd.DataFrame(results).T.sort_values("AIC")
    # logging.debug("GARCH Model Comparison:")
    # logging.debug(results_df.round(2))
    # logging.debug(f"\nBest model: {best_model_name}")

    return best_model_name, best_params


def robust_garch_fit(residuals, vol="GARCH", p=1, q=1, o=0, dist="normal"):
    """
    Fit a GARCH model with multiple optimization methods for robustness.

    Tries different optimization algorithms (L-BFGS-B, SLSQP, TNC) and returns
    the best fit based on log-likelihood.

    Parameters
    ----------
    residuals : pd.Series
        Residuals from ARIMA model to model with GARCH
    vol : str, optional
        Volatility model type: 'GARCH' or 'EGARCH' (default: 'GARCH')
    p : int, optional
        Number of ARCH terms (default: 1)
    q : int, optional
        Number of GARCH terms (default: 1)
    o : int, optional
        Number of asymmetry terms for GJR-GARCH (default: 0)
    dist : str, optional
        Error distribution: 'normal' or 't' (default: 'normal')

    Returns
    -------
    arch.univariate.base.ARCHModelResult
        Fitted GARCH model result

    Raises
    ------
    ValueError
        If all optimization methods fail
    """
    model = arch_model(residuals, vol=vol, p=p, q=q, o=o, dist=dist, rescale=False)

    optimization_methods = [
        {"method": "L-BFGS-B", "options": {"maxiter": 2000}},
        {"method": "SLSQP", "options": {"maxiter": 1000, "ftol": 1e-9}},
        {"method": "TNC", "options": {"maxiter": 1000}},
    ]

    best_fit = None
    best_llf = -np.inf

    for opt in optimization_methods:
        try:
            fit = model.fit(disp="off", options=opt["options"])
            if fit.loglikelihood > best_llf:
                best_llf = fit.loglikelihood
                best_fit = fit
        except:
            continue

    if best_fit is None:
        try:
            best_fit = model.fit(disp="off")
        except:
            raise ValueError("All optimization methods failed")

    return best_fit


def forecast_arima(ts, transform, order, seasonal_order=(0, 0, 0, 0)):
    """
    Generate one-step ahead forecast using ARIMA or SARIMAX model.

    Parameters
    ----------
    ts : pd.Series
        Original time series data
    transform : str
        Transformation type applied: 'none', 'log', 'log_diff', 'diff', 'diff2'
    order : tuple
        ARIMA order (p, d, q)
    seasonal_order : tuple, optional
        SARIMAX seasonal order (P, D, Q, s). If s=0, uses ARIMA (default: (0,0,0,0))

    Returns
    -------
    tuple
        (forecast, ci_lower, ci_upper) in original scale
    """
    try:
        ts_final, _ = make_stationary(ts)

        if len(ts_final) < 10:
            return ts.iloc[-1], ts.iloc[-1], ts.iloc[-1]

        if seasonal_order[3] > 0:
            model = SARIMAX(ts_final, order=order, seasonal_order=seasonal_order).fit()
        else:
            model = ARIMA(ts_final, order=order).fit()

        forecast_result = model.get_forecast(steps=1)
        forecast = forecast_result.predicted_mean.iloc[0]
        ci = forecast_result.conf_int().iloc[0]

        if transform == "log":
            return np.exp(forecast), np.exp(ci.iloc[0]), np.exp(ci.iloc[1])
        elif transform == "log_diff":
            last_log = np.log(ts.iloc[-1])
            return (
                np.exp(forecast + last_log),
                np.exp(ci.iloc[0] + last_log),
                np.exp(ci.iloc[1] + last_log),
            )
        elif transform == "diff":
            last_val = ts.iloc[-1]
            return forecast + last_val, ci.iloc[0] + last_val, ci.iloc[1] + last_val
        else:
            return forecast, ci.iloc[0], ci.iloc[1]
    except:
        return ts.iloc[-1], ts.iloc[-1], ts.iloc[-1]


def forecast_garch(
    original_ts, stationary_ts, arima_order, garch_params, transform_type
):
    """
    Generate one-step ahead forecast using ARIMA-GARCH model.

    Combines ARIMA for mean forecasting and GARCH for volatility modeling.

    Parameters
    ----------
    original_ts : pd.Series
        Original time series data
    stationary_ts : pd.Series
        Stationary transformed time series
    arima_order : tuple
        ARIMA order (p, d, q) for mean model
    garch_params : dict
        GARCH model parameters (vol, p, q, o, dist)
    transform_type : str
        Transformation type: 'log_diff', 'diff', or 'none'

    Returns
    -------
    tuple
        (price_forecast, ci_lower, ci_upper, volatility) in original scale
    """
    stationary_ts = pd.to_numeric(stationary_ts, errors="coerce")
    stationary_ts = stationary_ts.dropna()
    stationary_ts = stationary_ts.astype("float64")
    arima_model = ARIMA(stationary_ts, order=arima_order).fit()
    residuals = arima_model.resid

    garch_model = arch_model(residuals, **garch_params, rescale=False)
    garch_fit = garch_model.fit(disp="off", options={"maxiter": 1000})
    arima_forecast = arima_model.get_forecast(steps=1)
    diff_forecast = arima_forecast.predicted_mean.iloc[0]

    garch_forecast = garch_fit.forecast(horizon=1, reindex=False)
    volatility = np.sqrt(garch_forecast.variance.iloc[-1, 0])

    if transform_type == "log_diff":
        last_log_price = np.log(original_ts.iloc[-1])
        price_forecast = np.exp(last_log_price + diff_forecast)
        ci_lower = np.exp(last_log_price + diff_forecast - 1.96 * volatility)
        ci_upper = np.exp(last_log_price + diff_forecast + 1.96 * volatility)
    elif transform_type == "diff":
        last_price = original_ts.iloc[-1]
        price_forecast = last_price + diff_forecast
        ci_lower = last_price + diff_forecast - 1.96 * volatility
        ci_upper = last_price + diff_forecast + 1.96 * volatility
    else:
        price_forecast = diff_forecast
        ci_lower = diff_forecast - 1.96 * volatility
        ci_upper = diff_forecast + 1.96 * volatility

    return price_forecast, ci_lower, ci_upper, volatility


def extract_model_equations(model, model_type, transform, current_values=None):
    """
    Extract mathematical equations from fitted ARIMA or GARCH model.

    Parameters
    ----------
    model : statsmodels model or arch model
        Fitted ARIMA/SARIMAX or GARCH model
    model_type : str
        Type of model: 'ARIMA' or 'GARCH'
    transform : str
        Transformation applied: 'diff', 'log_diff', or 'none'
    current_values : dict, optional
        Current values for equation display (default: None)

    Returns
    -------
    dict
        Dictionary containing model equations as strings
    """
    equations = {}

    if model_type == "ARIMA":
        params = model.params
        p, d, q = model.model.order

        ar_terms = []
        ma_terms = []

        if p > 0:
            for i in range(p):
                coef = params[f"ar.L{i + 1}"] if f"ar.L{i + 1}" in params else 0
                ar_terms.append(f"{coef:.6f}*X(t-{i + 1})")

        if q > 0:
            for i in range(q):
                coef = params[f"ma.L{i + 1}"] if f"ma.L{i + 1}" in params else 0
                ma_terms.append(f"{coef:.6f}*ε(t-{i + 1})")

        const = params["const"] if "const" in params else 0

        ar_part = " + ".join(ar_terms) if ar_terms else "0"
        ma_part = " + ".join(ma_terms) if ma_terms else "0"

        if transform == "diff":
            equations["stationary_eq"] = (
                f"ΔX(t) = {const:.6f} + {ar_part} + ε(t) + {ma_part}"
            )
            equations["price_eq"] = "X(t) = X(t-1) + ΔX(t)"
        elif transform == "log_diff":
            equations["stationary_eq"] = (
                f"Δlog(X(t)) = {const:.6f} + {ar_part} + ε(t) + {ma_part}"
            )
            equations["price_eq"] = "X(t) = X(t-1) * exp(Δlog(X(t)))"
        else:
            equations["stationary_eq"] = (
                f"X(t) = {const:.6f} + {ar_part} + ε(t) + {ma_part}"
            )
            equations["price_eq"] = "X(t) = stationary_value"

    elif model_type == "GARCH":
        garch_params = model.params

        mean_eq = "μ(t) = constant"

        omega = garch_params["omega"] if "omega" in garch_params else 0

        # Extract all ARCH terms (alpha parameters)
        alpha_terms = []
        i = 1
        while f"alpha[{i}]" in garch_params:
            alpha_val = garch_params[f"alpha[{i}]"]
            alpha_terms.append(f"{alpha_val:.6f}*ε²(t-{i})")
            i += 1

        # Extract all GARCH terms (beta parameters)
        beta_terms = []
        i = 1
        while f"beta[{i}]" in garch_params:
            beta_val = garch_params[f"beta[{i}]"]
            beta_terms.append(f"{beta_val:.6f}*σ²(t-{i})")
            i += 1

        all_terms = alpha_terms + beta_terms
        variance_eq = f"σ²(t) = {omega:.6f} + " + " + ".join(all_terms)

        if transform == "log_diff":
            equations["mean_eq"] = "Δlog(X(t)) = μ(t) + σ(t)*z(t)"
            equations["variance_eq"] = variance_eq
            equations["price_eq"] = "X(t) = X(t-1) * exp(Δlog(X(t)))"
        elif transform == "diff":
            equations["mean_eq"] = "ΔX(t) = μ(t) + σ(t)*z(t)"
            equations["variance_eq"] = variance_eq
            equations["price_eq"] = "X(t) = X(t-1) + ΔX(t)"
        else:
            equations["mean_eq"] = "X(t) = μ(t) + σ(t)*z(t)"
            equations["variance_eq"] = variance_eq
            equations["price_eq"] = "X(t) = predicted_value"

    return equations


def capture_prediction_equations(
    original_ts,
    stationary_ts,
    transform,
    arima_order,
    seasonal_order,
    garch_params,
    has_hetero,
    date,
    actual_values=None,
):
    """
    Capture detailed prediction equations and calculations for a specific date.

    Parameters
    ----------
    original_ts : pd.Series
        Original time series data
    stationary_ts : pd.Series
        Stationary transformed time series
    transform : str
        Transformation type applied
    arima_order : tuple
        ARIMA order (p, d, q)
    seasonal_order : tuple
        SARIMAX seasonal order (P, D, Q, s)
    garch_params : dict or None
        GARCH model parameters if heteroscedasticity detected
    has_hetero : bool
        Whether heteroscedasticity was detected
    date : pd.Timestamp
        Date for which to capture equations
    actual_values : dict, optional
        Actual values for comparison (default: None)

    Returns
    -------
    dict
        Dictionary containing all equation details and calculations
    """
    equations_data = {
        "date": date,
        "transform_type": transform,
        "last_price": original_ts.iloc[-1],
        "last_stationary": stationary_ts.iloc[-1] if len(stationary_ts) > 0 else None,
    }

    try:
        ts_final, _ = make_stationary(original_ts)
        if seasonal_order[3] > 0:
            arima_model = SARIMAX(
                ts_final, order=arima_order, seasonal_order=seasonal_order
            ).fit()
            model_type = "SARIMAX"
        else:
            arima_model = ARIMA(ts_final, order=arima_order).fit()
            model_type = "ARIMA"

        arima_eqs = extract_model_equations(arima_model, "ARIMA", transform)
        equations_data.update({f"arima_{k}": v for k, v in arima_eqs.items()})

        forecast_result = arima_model.get_forecast(steps=1)
        predicted_diff = forecast_result.predicted_mean.iloc[0]
        equations_data["arima_predicted_diff"] = predicted_diff

        if transform == "log_diff":
            final_price = original_ts.iloc[-1] * np.exp(predicted_diff)
            equations_data["arima_calculation"] = (
                f"{original_ts.iloc[-1]:.4f} * exp({predicted_diff:.6f}) = {final_price:.4f}"
            )
        elif transform == "diff":
            final_price = original_ts.iloc[-1] + predicted_diff
            equations_data["arima_calculation"] = (
                f"{original_ts.iloc[-1]:.4f} + {predicted_diff:.6f} = {final_price:.4f}"
            )
        else:
            final_price = predicted_diff
            equations_data["arima_calculation"] = (
                f"direct_prediction = {final_price:.4f}"
            )

        equations_data["arima_final_price"] = final_price

    except Exception as e:
        equations_data["arima_error"] = str(e)

    if has_hetero and garch_params:
        try:
            arima_model_for_garch = ARIMA(stationary_ts, order=arima_order).fit()
            residuals = arima_model_for_garch.resid

            garch_model = arch_model(residuals, **garch_params, rescale=False)
            garch_fit = garch_model.fit(disp="off", options={"maxiter": 1000})

            garch_eqs = extract_model_equations(garch_fit, "GARCH", transform)
            equations_data.update({f"garch_{k}": v for k, v in garch_eqs.items()})

            arima_forecast = arima_model_for_garch.get_forecast(steps=1)
            mean_forecast = arima_forecast.predicted_mean.iloc[0]

            garch_forecast = garch_fit.forecast(horizon=1, reindex=False)
            volatility = np.sqrt(garch_forecast.variance.iloc[-1, 0])

            equations_data["garch_mean_forecast"] = mean_forecast
            equations_data["garch_volatility"] = volatility

            if transform == "log_diff":
                garch_price = original_ts.iloc[-1] * np.exp(mean_forecast)
                equations_data["garch_calculation"] = (
                    f"{original_ts.iloc[-1]:.4f} * exp({mean_forecast:.6f}) = {garch_price:.4f}"
                )
            elif transform == "diff":
                garch_price = original_ts.iloc[-1] + mean_forecast
                equations_data["garch_calculation"] = (
                    f"{original_ts.iloc[-1]:.4f} + {mean_forecast:.6f} = {garch_price:.4f}"
                )
            else:
                garch_price = mean_forecast
                equations_data["garch_calculation"] = (
                    f"direct_prediction = {garch_price:.4f}"
                )

            equations_data["garch_final_price"] = garch_price
            equations_data["garch_ci_calculation"] = (
                f"[{garch_price - 1.96 * volatility:.4f}, {garch_price + 1.96 * volatility:.4f}]"
            )

        except Exception as e:
            equations_data["garch_error"] = str(e)

    return equations_data


def enhanced_forecast_single_step(
    original_ts,
    stationary_ts,
    transform,
    arima_order,
    seasonal_order,
    garch_params,
    has_hetero,
    date,
):
    """
    Generate enhanced one-step ahead forecast with detailed calculation report.

    Combines ARIMA, SARIMAX (if seasonal), and GARCH (if heteroscedasticity detected)
    to produce forecasts with detailed calculation breakdown.

    Parameters
    ----------
    original_ts : pd.Series
        Original time series data
    stationary_ts : pd.Series
        Stationary transformed time series
    transform : str
        Transformation type applied
    arima_order : tuple
        ARIMA order (p, d, q)
    seasonal_order : tuple
        SARIMAX seasonal order (P, D, Q, s)
    garch_params : dict or None
        GARCH model parameters if heteroscedasticity detected
    has_hetero : bool
        Whether heteroscedasticity was detected
    date : pd.Timestamp
        Date for which to generate forecast

    Returns
    -------
    tuple
        (forecasts_dict, equations_dict, report_string) where:
        - forecasts_dict contains predictions for ARIMA, SARIMAX, GARCH
        - equations_dict contains model equations
        - report_string contains detailed calculation report
    """
    arima_pred, arima_ci_l, arima_ci_u = forecast_arima(
        original_ts, transform, arima_order
    )

    if seasonal_order[3] > 0:
        sarimax_pred, sarimax_ci_l, sarimax_ci_u = forecast_arima(
            original_ts, transform, arima_order, seasonal_order
        )
    else:
        sarimax_pred = sarimax_ci_l = sarimax_ci_u = arima_pred

    if has_hetero:
        garch_pred, garch_ci_l, garch_ci_u, garch_vol = forecast_garch(
            original_ts, stationary_ts, arima_order, garch_params, transform
        )
    else:
        garch_pred = garch_ci_l = garch_ci_u = arima_pred
        garch_vol = 0.02

    forecasts = {
        "arima": (arima_pred, arima_ci_l, arima_ci_u),
        "sarimax": (sarimax_pred, sarimax_ci_l, sarimax_ci_u),
        "garch": (garch_pred, garch_ci_l, garch_ci_u, garch_vol),
    }

    equations = capture_prediction_equations(
        original_ts,
        stationary_ts,
        transform,
        arima_order,
        seasonal_order,
        garch_params,
        has_hetero,
        date,
    )

    arimareport = ""
    arimareport += "\n" + "=" * 80 + "\n"
    arimareport += "DETAILED ARIMA CALCULATION FOR DAY-AHEAD FORECAST\n"
    arimareport += "=" * 80 + "\n"
    arimareport += f"\nPrediction Date: {date}\n"
    arimareport += f"Model: ARIMA{arima_order}\n"
    arimareport += f"Transformation: {transform}\n"
    arimareport += f"\nLast Price: {original_ts.iloc[-1]:.4f} EUR\n"

    try:
        ts_final, _ = make_stationary(original_ts)
        if seasonal_order[3] > 0:
            model = SARIMAX(
                ts_final, order=arima_order, seasonal_order=seasonal_order
            ).fit()
        else:
            model = ARIMA(ts_final, order=arima_order).fit()

        arimareport += "\n--- Model Parameters ---\n"
        for param_name, param_value in model.params.items():
            arimareport += f"{param_name}: {param_value:.6f}\n"

        arimareport += "\n--- Model Equation ---\n"
        if "arima_stationary_eq" in equations:
            arimareport += f"Stationary: {equations['arima_stationary_eq']}\n"
        if "arima_price_eq" in equations:
            arimareport += f"Price: {equations['arima_price_eq']}\n"

        forecast_result = model.get_forecast(steps=1)
        predicted_diff = forecast_result.predicted_mean.iloc[0]
        ci = forecast_result.conf_int().iloc[0]

        arimareport += "\n--- Calculation Steps ---\n"
        arimareport += f"Predicted difference: {predicted_diff:.6f}\n"

        if transform == "log_diff":
            arimareport += "Transformation: exp(log(last_price) + predicted_diff)\n"
            arimareport += f"Calculation: exp(log({original_ts.iloc[-1]:.4f}) + {predicted_diff:.6f})\n"
            arimareport += f"           = exp({np.log(original_ts.iloc[-1]):.6f} + {predicted_diff:.6f})\n"
            arimareport += f"           = exp({np.log(original_ts.iloc[-1]) + predicted_diff:.6f})\n"
            arimareport += f"           = {arima_pred:.4f} EUR\n"
        elif transform == "diff":
            arimareport += "Transformation: last_price + predicted_diff\n"
            arimareport += (
                f"Calculation: {original_ts.iloc[-1]:.4f} + {predicted_diff:.6f}\n"
            )
            arimareport += f"           = {arima_pred:.4f} EUR\n"
        else:
            arimareport += f"Direct prediction: {arima_pred:.4f} EUR\n"

        arimareport += "\n--- Confidence Interval (95%) ---\n"
        arimareport += f"Lower bound: {arima_ci_l:.4f} EUR\n"
        arimareport += f"Upper bound: {arima_ci_u:.4f} EUR\n"

        arimareport += "\n--- Summary ---\n"
        arimareport += f"Final Prediction: {arima_pred:.4f} EUR\n"
        change = arima_pred - original_ts.iloc[-1]
        pct_change = ((arima_pred / original_ts.iloc[-1]) - 1) * 100
        arimareport += f"Change: {change:+.4f} EUR ({pct_change:+.2f}%)\n"

    except Exception as e:
        arimareport += f"\nError in detailed calculation: {e!s}\n"

    arimareport += "=" * 80 + "\n"

    return forecasts, equations, arimareport


def unified_model_comparison(
    data, target_col="EUA_benchmark_settlement", split_date="2025-01-01"
):
    """
    This function compares the performance of the ARIMA, SARIMAX, and GARCH models.
    It splits the data into training and test sets, and then fits the models to the training data.
    It then compares the performance of the models on the test data.
    It returns the results dataframe, the model configuration, and the equations dataframe.

    Args:
        data: The data to compare the models on.
        target_col: The column to compare the models on.
        split_date: The date to split the data into training and test sets.

    Returns:
        results_df: The results dataframe.
        model_config: The model configuration.
        equations_df: The equations dataframe.
    """
    data = data.sort_index()
    train_data = data[data.index <= split_date]
    test_data = data[data.index > split_date]

    train_ts = train_data[target_col].dropna()

    # logging.debug("=== ANALYZING SERIES ===")
    seasonal_periods = detect_seasonality(train_ts)

    transform, arima_order, seasonal_order, has_hetero, garch_name, garch_params = (
        get_optimal_models(train_ts, seasonal_periods)
    )

    # logging.debug("\n=== ROLLING FORECAST EVALUATION ===")
    results = []
    current_original = train_ts.copy()
    current_stationary, _ = make_stationary(current_original)
    equations_list = []

    for test_date in test_data.index:
        if test_date in data.index and target_col in data.columns:
            actual = data.loc[test_date, target_col]

            # forecasts, equations, _ = enhanced_forecast_single_step(
            #     current_original,
            #     current_stationary,
            #     transform,
            #     arima_order,
            #     seasonal_order,
            #     garch_params,
            #     has_hetero,
            #     test_date,
            # )
            # equations_list.append(equations)

            result_row = {
                "date": test_date,
                "actual": actual,
                # "arima_predicted": forecasts["arima"][0],
                # "arima_ci_lower": forecasts["arima"][1],
                # "arima_ci_upper": forecasts["arima"][2],
                # "sarimax_predicted": forecasts["sarimax"][0],
                # "sarimax_ci_lower": forecasts["sarimax"][1],
                # "sarimax_ci_upper": forecasts["sarimax"][2],
                # "garch_predicted": forecasts["garch"][0],
                # "garch_ci_lower": forecasts["garch"][1],
                # "garch_ci_upper": forecasts["garch"][2],
                # "garch_volatility": forecasts["garch"][3] if has_hetero else 0.02,
            }

            results.append(result_row)
            current_original = pd.concat(
                [current_original, pd.Series([actual], index=[test_date])]
            )

            try:
                if transform == "log_diff":
                    stationary_val = np.log(actual) - np.log(current_original.iloc[-2])
                elif transform == "diff":
                    stationary_val = actual - current_original.iloc[-2]
                else:
                    stationary_val = actual
                current_stationary = pd.concat(
                    [current_stationary, pd.Series([stationary_val], index=[test_date])]
                )
            except:
                pass

    results_df = pd.DataFrame(results)

    current_price = current_original.iloc[-1]
    # next_forecasts, _, arimareport = enhanced_forecast_single_step(
    #     current_original,
    #     current_stationary,
    #     transform,
    #     arima_order,
    #     seasonal_order,
    #     garch_params,
    #     has_hetero,
    #     pd.Timestamp.now().date(),
    # )

    equations_df = pd.DataFrame(equations_list)
    # logging.debug(f"\nEquations DataFrame saved with {len(equations_df)} predictions")

    # logging.debug(arimareport)

    next_business_date = results_df["date"].iloc[-1] + pd.offsets.BDay(1)

    next_row = {
        "date": next_business_date,
        "actual": np.nan,
        # "arima_predicted": next_forecasts["arima"][0],
        # "arima_ci_lower": next_forecasts["arima"][1],
        # "arima_ci_upper": next_forecasts["arima"][2],
        # "sarimax_predicted": next_forecasts["sarimax"][0],
        # "sarimax_ci_lower": next_forecasts["sarimax"][1],
        # "sarimax_ci_upper": next_forecasts["sarimax"][2],
        # "garch_predicted": next_forecasts["garch"][0],
        # "garch_ci_lower": next_forecasts["garch"][1],
        # "garch_ci_upper": next_forecasts["garch"][2],
        # "garch_volatility": next_forecasts["garch"][3] if has_hetero else 0.02,
    }

    results_df = pd.concat([results_df, pd.DataFrame([next_row])], ignore_index=True)

    return (
        results_df,
        (transform, arima_order, seasonal_order, has_hetero, garch_name, garch_params),
        equations_df,
    )


def get_unified_5day_forecasts(
    data, target_col="EUA_benchmark_settlement", model_config=None
):
    """
    Generate 5-day ahead forecasts using unified model approach.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset with target column
    target_col : str, optional
        Name of target column to forecast (default: 'EUA_benchmark_settlement')
    model_config : tuple, optional
        Pre-computed model configuration. If None, will be computed (default: None)

    Returns
    -------
    tuple
        (forecast_df, forecast_equations_df) containing 5-day forecasts and equations
    """
    full_ts = data[target_col].dropna()

    if model_config is None:
        seasonal_periods = detect_seasonality(full_ts, max_lag=30)
        transform, arima_order, seasonal_order, has_hetero, garch_name, garch_params = (
            get_optimal_models(full_ts, seasonal_periods)
        )
    else:
        transform, arima_order, seasonal_order, has_hetero, garch_name, garch_params = (
            model_config
        )

    stationary_ts, _ = make_stationary(full_ts)
    forecast_dates = pd.date_range(
        start=full_ts.index[-1] + pd.Timedelta(days=1), periods=5, freq="D"
    )
    forecasts = []

    current_original = full_ts.copy()
    current_stationary = stationary_ts.copy()
    equations_list = []

    for i, forecast_date in enumerate(forecast_dates):
        step_forecasts, equations, _ = enhanced_forecast_single_step(
            current_original,
            current_stationary,
            transform,
            arima_order,
            seasonal_order,
            garch_params,
            has_hetero,
            forecast_date,
        )
        equations_list.append(equations)

        forecast_row = {
            "date": forecast_date,
            # "arima_predicted": step_forecasts["arima"][0],
            # "arima_ci_lower": step_forecasts["arima"][1],
            # "arima_ci_upper": step_forecasts["arima"][2],
            # "sarimax_predicted": step_forecasts["sarimax"][0],
            # "sarimax_ci_lower": step_forecasts["sarimax"][1],
            # "sarimax_ci_upper": step_forecasts["sarimax"][2],
            # "garch_predicted": step_forecasts["garch"][0],
            # "garch_ci_lower": step_forecasts["garch"][1],
            # "garch_ci_upper": step_forecasts["garch"][2],
            # "garch_volatility": step_forecasts["garch"][3] if has_hetero else 0.02,
        }

        forecasts.append(forecast_row)

        predicted_value = step_forecasts["arima"][0]
        current_original = pd.concat(
            [current_original, pd.Series([predicted_value], index=[forecast_date])]
        )

        try:
            if transform == "log_diff":
                stationary_val = np.log(predicted_value) - np.log(
                    current_original.iloc[-2]
                )
            elif transform == "diff":
                stationary_val = predicted_value - current_original.iloc[-2]
            else:
                stationary_val = predicted_value
            current_stationary = pd.concat(
                [current_stationary, pd.Series([stationary_val], index=[forecast_date])]
            )
        except:
            pass

    forecast_df = pd.DataFrame(forecasts)
    forecast_equations_df = pd.DataFrame(equations_list)
    return forecast_df, forecast_equations_df


def select_eua_features(data):
    """
    Select best features for EUA prediction by testing transformations and RMSE.

    For each variable group, tests different transformations and selects the feature
    with the lowest RMSE when predicting the target.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing features and target column 'EUA_benchmark_settlement'

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: column_name, variable_group, transformation, rmse, best
    """

    def test_stationarity(series):
        series_clean = series.dropna()
        if len(series_clean) < 3 or series_clean.nunique() <= 1:
            return True
        return adfuller(series_clean)[1] < 0.05

    def log_diff(series):
        return np.log(series).diff()

    def first_diff(series):
        return series.diff()

    def weekly_pct_change(series):
        weekly = series.resample("W").last()
        return weekly.pct_change().reindex(series.index, method="ffill")

    def get_best_transformation(series, var_name):
        if test_stationarity(series):
            return series, "levels"

        if var_name.startswith("COT_"):
            transforms = {"weekly_pct": weekly_pct_change, "first_diff": first_diff}
        elif any(
            var_name.startswith(prefix)
            for prefix in ["TTF", "Coal", "German", "Brent", "EUA"]
        ):
            transforms = {"log_diff": log_diff, "first_diff": first_diff}
        else:
            transforms = {"log_diff": log_diff, "first_diff": first_diff}

        for transform_name, transform_func in transforms.items():
            try:
                transformed = transform_func(series)
                if test_stationarity(transformed):
                    return transformed, transform_name
            except:
                continue

        return first_diff(series), "first_diff"

    df_stationary = data.copy()
    transformations = {}

    for col in data.columns:
        if col != "EUA_benchmark_settlement":
            df_stationary[col], transformations[col] = get_best_transformation(
                data[col], col
            )

    target_stationary = log_diff(data["EUA_benchmark_settlement"])
    transformations["EUA_benchmark_settlement"] = "log_diff"

    target_t_plus_1 = target_stationary.shift(-1)
    target_t = target_stationary

    variable_groups = {
        "TTF": [col for col in data.columns if col.startswith("TTF")],
        "Coal": [col for col in data.columns if col.startswith("Coal")],
        "Power": [col for col in data.columns if col.startswith("German")],
        "Brent": [col for col in data.columns if col.startswith("Brent")],
        "Auction": [col for col in data.columns if col.startswith("Auction")],
        "Weather": [col for col in data.columns if col.startswith("weather")],
        "Options": [col for col in data.columns if col.startswith("op")],
        # 'EUA': [col for col in data.columns if col.startswith('EUA') and col != 'EUA_benchmark_settlement'],
        # 'COT': [col for col in data.columns if col.startswith('COT')],
        "COT": [
            col
            for col in data.columns
            if col.startswith("COT") and not col.startswith("COT_TTF_")
        ],
        "COT_TTF": [col for col in data.columns if col.startswith("COT_TTF")],
        # 'OFI': [col for col in data.columns if col.startswith('COT_OFI')],
        # 'CU': [col for col in data.columns if col.startswith('COT_CU')],
        # 'IFCI': [col for col in data.columns if col.startswith('COT_IFCI')],
        # 'OWCO': [col for col in data.columns if col.startswith('COT_OWCO')],
        # 'IF': [col for col in data.columns if col.startswith('COT_IF')],
        "Clean": [col for col in data.columns if col.startswith("clean")],
        "Switch": [col for col in data.columns if col.startswith("fuel")],
        "Wednesday": [col for col in data.columns if col.startswith("wednesday")],
    }

    results = []

    for group_name, variables in variable_groups.items():
        group_results = []

        for var in variables:
            if var in df_stationary.columns:
                temp_df = pd.DataFrame(
                    {
                        "target_t_plus_1": target_t_plus_1,
                        "target_t": target_t,
                        "feature_t": df_stationary[var],
                    }
                ).dropna()

                if len(temp_df) > 10:
                    # X = temp_df[['target_t', 'feature_t']]
                    X = temp_df[["feature_t"]]
                    y = temp_df["target_t_plus_1"]

                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    rmse = np.sqrt(mean_squared_error(y, y_pred))

                    if (
                        var == "TTF_front_settlement"
                        and "2025-01-03" in temp_df.index.astype(str)
                    ):
                        test_date = "2025-01-03"
                        test_row = temp_df.loc[test_date]
                        # predicted_transformed = model.intercept_ + model.coef_[0]*test_row['target_t'] + model.coef_[1]*test_row['feature_t']
                        predicted_transformed = (
                            model.intercept_ + model.coef_[0] * test_row["feature_t"]
                        )

                        current_log_eua = np.log(
                            data.loc[test_date, "EUA_benchmark_settlement"]
                        )
                        predicted_log_eua = current_log_eua + predicted_transformed
                        predicted_actual_eua = np.exp(predicted_log_eua)

                        # logging.debug(f"\n=== Verification for {var} on {test_date} ===")
                        # logging.debug(f"Original data: {data.loc[test_date, var]:.4f}")
                        # logging.debug(
                        #     f"Transformed data: {df_stationary.loc[test_date, var]:.4f}"
                        # )
                        # logging.debug(f"Transformation: {transformations[var]}")
                        # logging.debug(
                        #     f"Target original: {data.loc[test_date, 'EUA_benchmark_settlement']:.4f}"
                        # )
                        # logging.debug(
                        #     f"Target transformed: {target_stationary.loc[test_date]:.4f}"
                        # )
                        # logging.debug(f"Regression equation: target_t+1 = {model.intercept_:.6f} + {model.coef_[0]:.6f}*target_t + {model.coef_[1]:.6f}*{var}_t")
                        # logging.debug(
                        #     f"Regression equation: target_t+1 = {model.intercept_:.6f} + {model.coef_[0]:.6f}*{var}_t"
                        # )
                        # logging.debug(f"Predicted transformed: {predicted_transformed:.6f}")
                        # logging.debug(f"Predicted actual EUA: {predicted_actual_eua:.4f}")
                        # if transformations[var] == "log_diff":
                        #     logging.debug(
                        #         "To transform back: exp(cumsum(transformed_values)) * original_first_value"
                        #     )
                        # elif transformations[var] == "first_diff":
                        #     logging.debug(
                        #         "To transform back: cumsum(transformed_values) + original_first_value"
                        #     )
                        # logging.debug("=" * 50)

                    group_results.append(
                        {
                            "column_name": var,
                            "variable_group": group_name,
                            "transformation": transformations[var],
                            "rmse": rmse,
                        }
                    )

        if group_results:
            best_idx = min(
                range(len(group_results)), key=lambda i: group_results[i]["rmse"]
            )
            for i, result in enumerate(group_results):
                result["best"] = 1 if i == best_idx else 0
                results.append(result)

    return pd.DataFrame(results)


def apply_transformations_for_training(df, transformations):
    """
    Apply transformations to dataframe columns based on transformation dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe
    transformations : dict
        Dictionary mapping column names to transformation types:
        'log_diff', 'first_diff', 'weekly_pct', or 'levels'

    Returns
    -------
    pd.DataFrame
        Transformed dataframe with all transformations applied
    """
    df_transformed = df.copy()

    for col, transform in transformations.items():
        if col in df.columns:
            if transform == "log_diff":
                df_transformed[col] = np.log(np.maximum(df[col], 1e-10)).diff()
            elif transform == "first_diff":
                df_transformed[col] = df[col].diff()
            elif transform == "weekly_pct":
                weekly = df[col].resample("W").last()
                df_transformed[col] = weekly.pct_change().reindex(
                    df.index, method="ffill"
                )
            elif transform == "levels":
                df_transformed[col] = df[col]

    df_transformed = df_transformed.replace([np.inf, -np.inf], np.nan)
    df_transformed = df_transformed.clip(-1e10, 1e10)
    return df_transformed


def prepare_features_for_prediction(
    data, date, transformations, selected_features, debug=False
):
    """
    Prepare feature values for prediction at a specific date.

    Applies appropriate transformations based on the transformation dictionary
    to compute feature values using only data available before the prediction date.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset with datetime index
    date : pd.Timestamp
        Date for which to prepare features (features use data before this date)
    transformations : dict
        Dictionary mapping feature names to transformation types
    selected_features : list
        List of feature names to prepare
    debug : bool, optional
        Whether to logging.debug debug information (default: False)

    Returns
    -------
    np.ndarray
        Array of transformed feature values ready for model prediction

    Raises
    ------
    ValueError
        If not enough historical data is available
    """
    features = {}

    # if debug:
    # logging.debug(f"\n=== FEATURE PREPARATION FOR PREDICTING {date} ===")

    available_dates = data[data.index < date].index
    if len(available_dates) < 2:
        raise ValueError(f"Not enough historical data to predict {date}")

    last_date = available_dates[-1]
    second_last_date = available_dates[-2]

    # if debug:
    # logging.debug(f"Last available date: {last_date}")
    # logging.debug(f"Second last available date: {second_last_date}")

    for feature in selected_features:
        if feature in transformations:
            transform = transformations[feature]
            if transform in ["log_diff", "first_diff"]:
                current_val = data.loc[last_date, feature]
                prev_val = data.loc[second_last_date, feature]
                if transform == "log_diff":
                    features[feature] = np.log(np.maximum(current_val, 1e-10)) - np.log(
                        np.maximum(prev_val, 1e-10)
                    )
                else:
                    features[feature] = current_val - prev_val
            elif transform == "weekly_pct":
                weekly = data[feature].resample("W").last()
                weekly_pct = weekly.pct_change()
                features[feature] = weekly_pct.reindex(
                    [last_date], method="ffill"
                ).iloc[0]
            else:
                features[feature] = data.loc[last_date, feature]
        else:
            features[feature] = data.loc[last_date, feature]

        # if debug:
        #     logging.debug(f"{feature}: {features[feature]:.6f}")

    return np.array([features[f] for f in selected_features])


def inverse_transform_single(value, last_original, transform_type):
    """
    Inverse transform a single predicted value back to original scale.

    Parameters
    ----------
    value : float
        Transformed predicted value
    last_original : float
        Last value in original scale (needed for diff/log_diff transformations)
    transform_type : str
        Transformation type: 'log_diff', 'first_diff', or 'levels'

    Returns
    -------
    np.float64
        Predicted value in original scale
    """
    value = np.float64(value)
    last_original = np.float64(last_original)

    value = np.clip(value, -10, 10)

    if transform_type == "log_diff":
        return np.float64(last_original * np.exp(value))
    elif transform_type == "first_diff":
        return np.float64(last_original + value)
    else:
        return np.float64(value)


def select_best_alpha(X_train, y_train, model_type="elastic_high_l2", cv_folds=5):
    """
    Select optimal regularization parameter (alpha) using cross-validation.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix
    y_train : np.ndarray
        Training target vector
    model_type : str, optional
        Model type: 'elastic_high_l2', 'lasso', or 'elastic' (default: 'elastic_high_l2')
    cv_folds : int, optional
        Number of cross-validation folds (default: 5)

    Returns
    -------
    float
        Optimal alpha value for regularization
    """
    if model_type == "elastic_high_l2":
        alphas = np.logspace(-4, 0, 20)
        best_alpha = None
        best_score = -np.inf

        for alpha in alphas:
            try:
                model = ElasticNet(alpha=alpha, l1_ratio=0.1, max_iter=5000, tol=1e-4)
                scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=min(cv_folds, len(X_train) // 2),
                    scoring="neg_mean_squared_error",
                )
                mean_score = scores.mean()

                if np.isfinite(mean_score) and mean_score > best_score:
                    best_score = mean_score
                    best_alpha = alpha
            except:
                continue

        return best_alpha if best_alpha is not None else 0.01

    else:
        alphas = np.logspace(-3, 1, 30)
        best_alpha = None
        best_score = -np.inf

        for alpha in alphas:
            try:
                if model_type == "lasso":
                    model = Lasso(alpha=alpha, max_iter=5000, tol=1e-4)
                elif model_type == "elastic":
                    model = ElasticNet(
                        alpha=alpha, l1_ratio=0.5, max_iter=5000, tol=1e-4
                    )

                scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=min(cv_folds, len(X_train) // 2),
                    scoring="neg_mean_squared_error",
                )
                mean_score = scores.mean()

                if np.isfinite(mean_score) and mean_score > best_score:
                    best_score = mean_score
                    best_alpha = alpha
            except:
                continue

        return best_alpha if best_alpha is not None else 0.1


def rolling_lasso_regression_with_contributions(
    data,
    selected_features,
    transformations_dict,
    test_start_date="2025-01-01",
    window_days=None,
    predict_next_day=False,
):
    """
    Perform rolling window Lasso regression with feature contribution tracking.

    Fits Lasso models on rolling windows and tracks individual feature contributions
    to predictions. Can optionally generate day-ahead forecast with detailed report.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset with datetime index
    selected_features : list
        List of feature names to use in model
    transformations_dict : dict
        Dictionary mapping column names to transformation types
    test_start_date : str, optional
        Start date for test period (default: '2025-01-01')
    window_days : int, optional
        Rolling window size in days. If None, uses all available data (default: None)
    predict_next_day : bool, optional
        Whether to generate day-ahead forecast with detailed report (default: False)

    Returns
    -------
    tuple
        (results_df, contributions_df, day_ahead_forecast) where:
        - results_df: predictions and confidence intervals
        - contributions_df: feature contributions for each prediction
        - day_ahead_forecast: dict with next day forecast and calculation report
    """
    df_transformed = apply_transformations_for_training(data, transformations_dict)
    target = "EUA_benchmark_settlement"

    df_transformed[target] = df_transformed[target].shift(-1)

    test_dates = data[data.index >= test_start_date].index
    results = []
    contributions_data = []

    last_calculation_details = {}
    day_ahead_forecast = None

    for i, test_date in enumerate(test_dates):
        if window_days is None:
            train_data = df_transformed[df_transformed.index < test_date].dropna()
        else:
            start_date = test_date - pd.Timedelta(days=window_days)
            train_data = df_transformed[
                (df_transformed.index >= start_date)
                & (df_transformed.index < test_date)
            ].dropna()

        if len(train_data) < len(selected_features) + 2:
            continue

        X_train = train_data[selected_features].values.astype(np.float64)
        y_train = train_data[target].values.astype(np.float64)

        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0)
        X_std = np.where(X_std < 1e-6, 1.0, X_std)
        X_train_scaled = (X_train - X_mean) / X_std

        best_alpha = select_best_alpha(X_train_scaled, y_train, "lasso")
        model = Lasso(alpha=best_alpha, max_iter=5000, tol=1e-4)
        model.fit(X_train_scaled, y_train)

        try:
            # X_test = prepare_features_for_prediction(data, test_date, transformations_dict, selected_features, debug=(i < 2))
            X_test = prepare_features_for_prediction(
                data, test_date, transformations_dict, selected_features
            )
            if np.isnan(X_test).any():
                continue

            X_test_scaled = (X_test - X_mean) / X_std
            X_test_scaled = X_test_scaled.reshape(1, -1).astype(np.float64)
            prediction = model.predict(X_test_scaled)[0]

            if not np.isfinite(prediction):
                continue

            if i == len(test_dates) - 1 or test_date == test_dates[-1]:
                last_calculation_details = {
                    "test_date": test_date,
                    "train_start": start_date if window_days else train_data.index[0],
                    "train_end": train_data.index[-1],
                    "coefficients": model.coef_,
                    "intercept": model.intercept_,
                    "features": selected_features,
                    "X_test_raw": X_test,
                    "X_test_scaled": X_test_scaled[0],
                    "X_mean": X_mean,
                    "X_std": X_std,
                    "prediction_transformed": prediction,
                    "last_original_value": data.loc[
                        data[data.index < test_date].index[-1], target
                    ],
                    "alpha": best_alpha,
                    "model": model,
                    "X_train_scaled": X_train_scaled,
                    "y_train": y_train,
                }

            intercept_contrib = model.intercept_
            feature_contribs = {}
            for j, feature in enumerate(selected_features):
                coef = model.coef_[j]
                feature_val = X_test_scaled[0][j]
                contrib = coef * feature_val
                feature_contribs[feature] = contrib

            contrib_row = {
                "date": test_date,
                "intercept": intercept_contrib,
                "prediction_total": prediction,
            }
            contrib_row.update(feature_contribs)
            contributions_data.append(contrib_row)

            prev_date = data[data.index < test_date].index[-1]
            last_original = data.loc[prev_date, target]

            y_pred_train = model.predict(X_train_scaled)
            mse = np.mean((y_train - y_pred_train) ** 2)

            var_pred = mse * (1 + 1 / len(X_train))
            t_val = stats.t.ppf(
                0.975, len(X_train) - np.sum(np.abs(model.coef_) > 1e-6) - 1
            )
            margin_error = t_val * np.sqrt(var_pred)

            target_transform = transformations_dict.get(target, "levels")
            predicted_original = inverse_transform_single(
                prediction, last_original, target_transform
            )
            ci_lower_original = inverse_transform_single(
                prediction - margin_error, last_original, target_transform
            )
            ci_upper_original = inverse_transform_single(
                prediction + margin_error, last_original, target_transform
            )

            results.append(
                {
                    "date": test_date,
                    "actual_original": np.float64(data.loc[test_date, target]),
                    "predicted_original": predicted_original,
                    "ci_lower_original": ci_lower_original,
                    "ci_upper_original": ci_upper_original,
                    "alpha_used": best_alpha,
                    "active_features": np.sum(np.abs(model.coef_) > 1e-6),
                }
            )

        except Exception as e:
            logging.debug(f"Error processing {test_date}: {e}")
            continue

    if predict_next_day and last_calculation_details:
        next_date = data.index[-1] + pd.Timedelta(days=1)

        # Initialize report string
        report = ""
        report += "\n" + "=" * 80 + "\n"
        report += "DETAILED LASSO CALCULATION FOR DAY-AHEAD FORECAST\n"
        report += "=" * 80 + "\n"

        # Use latest data including today
        if window_days is None:
            train_data = df_transformed.dropna()
        else:
            start_date = data.index[-1] - pd.Timedelta(days=window_days - 1)
            train_data = df_transformed[(df_transformed.index >= start_date)].dropna()

        report += f"\nPrediction Date: {next_date.date()} (t+1)\n"
        report += f"Current Date: {data.index[-1].date()} (t)\n"
        report += f"Training Window: {train_data.index[0].date()} to {train_data.index[-1].date()}\n"
        report += f"Alpha (regularization): {last_calculation_details['alpha']:.6f}\n"

        # Check transformations for features
        report += "\n" + "-" * 60 + "\n"
        report += "TRANSFORMATIONS APPLIED:\n"
        report += "-" * 60 + "\n"
        for feature in selected_features:
            transform = transformations_dict.get(feature, "levels")
            report += f"{feature:30s}: {transform}\n"

        # Prepare features for next day prediction
        features = {}
        report += "\n" + "-" * 60 + "\n"
        report += "RAW FEATURE VALUES (before transformation):\n"
        report += "-" * 60 + "\n"

        for feature in selected_features:
            transform = transformations_dict.get(feature, "levels")

            if transform in ["log_diff", "first_diff"]:
                current_val = data.iloc[-1][feature]
                prev_val = data.iloc[-2][feature]
                if transform == "log_diff":
                    features[feature] = np.log(np.maximum(current_val, 1e-10)) - np.log(
                        np.maximum(prev_val, 1e-10)
                    )
                    report += f"{feature:30s}: {features[feature]:15.6f} (t={current_val:.4f}, t-1={prev_val:.4f})\n"
                else:
                    features[feature] = current_val - prev_val
                    report += f"{feature:30s}: {features[feature]:15.6f} (t={current_val:.4f}, t-1={prev_val:.4f})\n"
            elif transform == "weekly_pct":
                weekly = data[feature].resample("W").last()
                weekly_pct = weekly.pct_change()
                features[feature] = weekly_pct.reindex(
                    [data.index[-1]], method="ffill"
                ).iloc[0]
                current_val = data.iloc[-1][feature]
                report += f"{feature:30s}: {features[feature]:15.6f} (weekly %, current={current_val:.4f})\n"
            else:
                features[feature] = data.iloc[-1][feature]
                report += f"{feature:30s}: {features[feature]:15.6f} (levels)\n"

        X_test_next = np.array([features[f] for f in selected_features])

        report += "\n" + "-" * 60 + "\n"
        report += "SCALING PARAMETERS:\n"
        report += "-" * 60 + "\n"
        for i, feature in enumerate(selected_features):
            report += f"{feature:30s}: mean={last_calculation_details['X_mean'][i]:10.6f}, std={last_calculation_details['X_std'][i]:10.6f}\n"

        # Scale features
        X_test_next_scaled = (
            X_test_next - last_calculation_details["X_mean"]
        ) / last_calculation_details["X_std"]

        report += "\n" + "-" * 60 + "\n"
        report += "SCALED FEATURE VALUES:\n"
        report += "-" * 60 + "\n"
        for i, feature in enumerate(selected_features):
            report += f"{feature:30s}: {X_test_next_scaled[i]:15.6f}\n"

        # Get model
        model = last_calculation_details["model"]

        report += "\n" + "-" * 60 + "\n"
        report += "MODEL COEFFICIENTS:\n"
        report += "-" * 60 + "\n"
        for i, feature in enumerate(selected_features):
            coef = model.coef_[i]
            if abs(coef) > 1e-10:
                report += f"{feature:30s}: {coef:15.6f}\n"
            else:
                report += f"{feature:30s}: {coef:15.6f} (inactive)\n"
        report += f"{'Intercept':30s}: {model.intercept_:15.6f}\n"

        # Predict
        prediction_next = model.predict(X_test_next_scaled.reshape(1, -1))[0]

        report += "\n" + "-" * 60 + "\n"
        report += "CALCULATION (coef * scaled_value):\n"
        report += "-" * 60 + "\n"
        total = 0
        for i, feature in enumerate(selected_features):
            coef = model.coef_[i]
            scaled_val = X_test_next_scaled[i]
            contrib = coef * scaled_val
            total += contrib
            if abs(coef) > 1e-10:
                report += f"{feature:30s}: {coef:10.6f} * {scaled_val:10.6f} = {contrib:10.6f}\n"

        report += f"{'Intercept':30s}: {model.intercept_:35.6f}\n"
        total += model.intercept_
        report += "-" * 60 + "\n"
        report += f"{'TOTAL (log difference)':30s}: {total:35.6f}\n"
        report += f"{'Predicted (should match)':30s}: {prediction_next:35.6f}\n"

        # Calculate confidence intervals for day-ahead forecast
        X_train_scaled = last_calculation_details["X_train_scaled"]
        y_train = last_calculation_details["y_train"]
        y_pred_train = model.predict(X_train_scaled)
        mse = np.mean((y_train - y_pred_train) ** 2)

        var_pred = mse * (1 + 1 / len(X_train_scaled))
        t_val = stats.t.ppf(
            0.975, len(X_train_scaled) - np.sum(np.abs(model.coef_) > 1e-10) - 1
        )
        margin_error = t_val * np.sqrt(var_pred)

        report += "\n" + "-" * 60 + "\n"
        report += "CONFIDENCE INTERVAL CALCULATION:\n"
        report += "-" * 60 + "\n"
        report += f"MSE from training: {mse:.6f}\n"
        report += f"Prediction variance: {var_pred:.6f}\n"
        report += f"t-value (95% CI): {t_val:.6f}\n"
        report += f"Margin of error: {margin_error:.6f}\n"

        report += "\n" + "-" * 60 + "\n"
        report += "INVERSE TRANSFORMATION:\n"
        report += "-" * 60 + "\n"
        last_eua = data.iloc[-1][target]
        final_prediction_next = last_eua * np.exp(prediction_next)
        ci_lower_next = last_eua * np.exp(prediction_next - margin_error)
        ci_upper_next = last_eua * np.exp(prediction_next + margin_error)

        report += f"Last EUA value (t): {last_eua:.6f}\n"
        report += f"Predicted log difference: {prediction_next:.6f}\n"
        report += f"Formula: {last_eua:.6f} * exp({prediction_next:.6f}) = {final_prediction_next:.6f}\n"
        report += f"Final EUA prediction (t+1): {final_prediction_next:.6f}\n"
        report += f"95% CI Lower: {ci_lower_next:.6f}\n"
        report += f"95% CI Upper: {ci_upper_next:.6f}\n"

        report += "\n" + "-" * 60 + "\n"
        report += "EXCEL REPLICATION FORMULA:\n"
        report += "-" * 60 + "\n"
        report += "\nFor scaled values (Xi_scaled = (Xi_raw - mean_i) / std_i):\n"
        excel_formula = "="
        for i, feature in enumerate(selected_features):
            coef = model.coef_[i]
            if abs(coef) > 1e-10:
                excel_formula += f"{coef:.6f}*((X_{i + 1}_raw-{last_calculation_details['X_mean'][i]:.6f})/{last_calculation_details['X_std'][i]:.6f})+"
        excel_formula += f"{model.intercept_:.6f}"
        report += excel_formula + "\n"

        report += "\nFinal EUA = Last_EUA * EXP(above_result)\n"
        report += f"\nFINAL DAY-AHEAD FORECAST: {final_prediction_next:.2f} EUR\n"
        report += (
            f"95% CONFIDENCE INTERVAL: [{ci_lower_next:.2f}, {ci_upper_next:.2f}] EUR\n"
        )
        report += "=" * 80 + "\n"

        day_ahead_forecast = {
            "date": next_date,
            "predicted_original": final_prediction_next,
            "ci_lower_original": ci_lower_next,
            "ci_upper_original": ci_upper_next,
            "calculation_report": report,  # Add this line
        }

    results_df = pd.DataFrame(results).set_index("date")
    contributions_df = pd.DataFrame(contributions_data).set_index("date")
    return results_df, contributions_df, day_ahead_forecast


def plot_lasso_predictions_with_ci(
    results_df, title="Lasso Predictions vs Actual", day_ahead_forecast=None
):
    """
    Plot Lasso predictions with confidence intervals and day-ahead forecast.

    Creates two subplots: full time series and zoomed last 30 days view.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with predictions, actuals, and confidence intervals
    title : str, optional
        Plot title (default: 'Lasso Predictions vs Actual')
    day_ahead_forecast : dict, optional
        Dictionary with day-ahead forecast info including 'date', 'predicted_original',
        'ci_lower_original', 'ci_upper_original' (default: None)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Full time series
    ax1.plot(
        results_df.index, results_df["actual_original"], "b-", label="Actual", alpha=0.7
    )
    ax1.plot(
        results_df.index,
        results_df["predicted_original"],
        "r--",
        label="Predicted",
        alpha=0.7,
    )
    ax1.fill_between(
        results_df.index,
        results_df["ci_lower_original"],
        results_df["ci_upper_original"],
        alpha=0.2,
        color="red",
        label="95% CI",
    )
    ax1.set_xlabel("Date")
    ax1.set_ylabel("EUA Price (EUR)")
    ax1.set_title(f"{title} - Full Period")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Last 30 days zoomed
    last_30 = results_df.tail(30)
    ax2.plot(
        last_30.index,
        last_30["actual_original"],
        "b-",
        label="Actual",
        marker="o",
        markersize=4,
    )
    ax2.plot(
        last_30.index,
        last_30["predicted_original"],
        "r--",
        label="Predicted",
        marker="s",
        markersize=4,
    )
    ax2.fill_between(
        last_30.index,
        last_30["ci_lower_original"],
        last_30["ci_upper_original"],
        alpha=0.2,
        color="red",
        label="95% CI",
    )

    # Add day-ahead forecast point with confidence intervals
    if day_ahead_forecast:
        next_date = day_ahead_forecast["date"]
        next_pred = day_ahead_forecast["predicted_original"]
        ax2.plot(next_date, next_pred, "go", markersize=8, label="Day-ahead forecast")

        # Add error bars for confidence interval
        ci_lower = day_ahead_forecast["ci_lower_original"]
        ci_upper = day_ahead_forecast["ci_upper_original"]
        ax2.errorbar(
            next_date,
            next_pred,
            yerr=[[next_pred - ci_lower], [ci_upper - next_pred]],
            fmt="go",
            capsize=5,
            capthick=2,
            alpha=0.7,
        )

        ax2.axvline(x=next_date, color="green", linestyle=":", alpha=0.5)
    else:
        next_date = last_30.index[-1] + pd.Timedelta(days=1)
        ax2.axvline(
            x=next_date, color="green", linestyle=":", alpha=0.5, label="Day-ahead"
        )

    ax2.set_xlabel("Date")
    ax2.set_ylabel("EUA Price (EUR)")
    ax2.set_title(f"{title} - Last 30 Days")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    # plt.show()


def rolling_elastic_regression_with_contributions(
    data,
    selected_features,
    transformations_dict,
    test_start_date="2025-01-01",
    window_days=None,
):
    """
    Perform rolling window ElasticNet regression with feature contribution tracking.

    Similar to rolling_lasso_regression but uses ElasticNet with high L2 regularization.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset with datetime index
    selected_features : list
        List of feature names to use in model
    transformations_dict : dict
        Dictionary mapping column names to transformation types
    test_start_date : str, optional
        Start date for test period (default: '2025-01-01')
    window_days : int, optional
        Rolling window size in days. If None, uses all available data (default: None)

    Returns
    -------
    tuple
        (results_df, contributions_df) containing predictions and feature contributions
    """
    df_transformed = apply_transformations_for_training(data, transformations_dict)
    target = "EUA_benchmark_settlement"

    df_transformed[target] = df_transformed[target].shift(-1)

    test_dates = data[data.index >= test_start_date].index
    results = []
    contributions_data = []

    for i, test_date in enumerate(test_dates):
        if window_days is None:
            train_data = df_transformed[df_transformed.index < test_date].dropna()
        else:
            start_date = test_date - pd.Timedelta(days=window_days)
            train_data = df_transformed[
                (df_transformed.index >= start_date)
                & (df_transformed.index < test_date)
            ].dropna()

        if len(train_data) < len(selected_features) + 2:
            continue

        X_train = train_data[selected_features].values.astype(np.float64)
        y_train = train_data[target].values.astype(np.float64)

        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0)
        X_std = np.where(X_std < 1e-6, 1.0, X_std)
        X_train_scaled = (X_train - X_mean) / X_std

        best_alpha = select_best_alpha(X_train_scaled, y_train, "elastic_high_l2")
        model = ElasticNet(alpha=best_alpha, l1_ratio=0.1, max_iter=5000, tol=1e-4)
        model.fit(X_train_scaled, y_train)

        try:
            # X_test = prepare_features_for_prediction(data, test_date, transformations_dict, selected_features, debug=(i < 2))
            X_test = prepare_features_for_prediction(
                data, test_date, transformations_dict, selected_features
            )
            if np.isnan(X_test).any():
                continue

            X_test_scaled = (X_test - X_mean) / X_std
            X_test_scaled = X_test_scaled.reshape(1, -1).astype(np.float64)
            prediction = model.predict(X_test_scaled)[0]

            if not np.isfinite(prediction):
                continue

            intercept_contrib = model.intercept_
            feature_contribs = {}
            for j, feature in enumerate(selected_features):
                coef = model.coef_[j]
                feature_val = X_test_scaled[0][j]
                contrib = coef * feature_val
                feature_contribs[feature] = contrib

            contrib_row = {
                "date": test_date,
                "intercept": intercept_contrib,
                "prediction_total": prediction,
            }
            contrib_row.update(feature_contribs)
            contributions_data.append(contrib_row)

            prev_date = data[data.index < test_date].index[-1]
            last_original = data.loc[prev_date, target]

            y_pred_train = model.predict(X_train_scaled)
            mse = np.mean((y_train - y_pred_train) ** 2)

            var_pred = mse * (1 + 1 / len(X_train))
            t_val = stats.t.ppf(
                0.975, len(X_train) - np.sum(np.abs(model.coef_) > 1e-6) - 1
            )
            margin_error = t_val * np.sqrt(var_pred)

            target_transform = transformations_dict.get(target, "levels")
            predicted_original = inverse_transform_single(
                prediction, last_original, target_transform
            )
            ci_lower_original = inverse_transform_single(
                prediction - margin_error, last_original, target_transform
            )
            ci_upper_original = inverse_transform_single(
                prediction + margin_error, last_original, target_transform
            )

            results.append(
                {
                    "date": test_date,
                    "actual_original": np.float64(data.loc[test_date, target]),
                    "predicted_original": predicted_original,
                    "ci_lower_original": ci_lower_original,
                    "ci_upper_original": ci_upper_original,
                    "alpha_used": best_alpha,
                    "active_features": np.sum(np.abs(model.coef_) > 1e-6),
                }
            )

        except Exception as e:
            logging.debug(f"Error processing {test_date}: {e}")
            continue

    results_df = pd.DataFrame(results).set_index("date")
    contributions_df = pd.DataFrame(contributions_data).set_index("date")
    return results_df, contributions_df


def convert_regularized_contributions_to_original(
    contributions_df, data, transformations_dict, target="EUA_benchmark_settlement"
):
    """
    Convert feature contributions from transformed scale to original price scale.

    Parameters
    ----------
    contributions_df : pd.DataFrame
        DataFrame with contributions in transformed scale
    data : pd.DataFrame
        Original dataset with target column
    transformations_dict : dict
        Dictionary mapping column names to transformation types
    target : str, optional
        Name of target column (default: 'EUA_benchmark_settlement')

    Returns
    -------
    pd.DataFrame
        DataFrame with contributions converted to original price scale (EUR)
    """
    original_contributions = contributions_df.copy()
    target_transform = transformations_dict.get(target, "levels")

    for date in contributions_df.index:
        prev_date = data[data.index < date].index[-1]
        last_original = data.loc[prev_date, target]

        prediction_transformed = contributions_df.loc[date, "prediction_total"]

        if target_transform == "log_diff":
            final_prediction = last_original * np.exp(prediction_transformed)
        elif target_transform == "first_diff":
            final_prediction = last_original + prediction_transformed
        else:
            final_prediction = prediction_transformed

        total_change = final_prediction - last_original

        feature_cols = [
            col
            for col in contributions_df.columns
            if col not in ["intercept", "prediction_total"]
        ]
        total_transformed_contrib = (
            contributions_df.loc[date, feature_cols].sum()
            + contributions_df.loc[date, "intercept"]
        )

        for col in contributions_df.columns:
            if col == "prediction_total":
                original_contributions.loc[date, col] = final_prediction
            elif col in feature_cols or col == "intercept":
                if abs(total_transformed_contrib) > 1e-10:
                    proportion = (
                        contributions_df.loc[date, col] / total_transformed_contrib
                    )
                    original_contributions.loc[date, col] = total_change * proportion
                else:
                    original_contributions.loc[date, col] = 0

    return original_contributions


def plot_regularized_multiple_waterfalls(
    contributions_df,
    data,
    transformations_dict,
    model_name,
    start_date,
    end_date,
    max_charts=6,
    target="EUA_benchmark_settlement",
):
    """
    Plot multiple waterfall charts showing feature contributions over time period.

    Parameters
    ----------
    contributions_df : pd.DataFrame
        DataFrame with feature contributions
    data : pd.DataFrame
        Original dataset
    transformations_dict : dict
        Dictionary mapping column names to transformation types
    model_name : str
        Name of the model (for plot title)
    start_date : str
        Start date for the period to plot
    end_date : str
        End date for the period to plot
    max_charts : int, optional
        Maximum number of waterfall charts to display (default: 6)
    target : str, optional
        Name of target column (default: 'EUA_benchmark_settlement')
    """
    original_contribs = convert_regularized_contributions_to_original(
        contributions_df, data, transformations_dict, target
    )

    date_mask = (original_contribs.index >= start_date) & (
        original_contribs.index <= end_date
    )
    plot_dates = original_contribs[date_mask].index

    if len(plot_dates) == 0:
        logging.debug(f"No data found for date range {start_date} to {end_date}")
        return

    selected_dates = plot_dates[:: max(1, len(plot_dates) // max_charts)][:max_charts]

    n_charts = len(selected_dates)
    cols = min(3, n_charts)
    rows = (n_charts + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_charts == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for i, date in enumerate(selected_dates):
        ax = axes[i]
        prev_date = data[data.index < date].index[-1]
        starting_price = data.loc[prev_date, target]
        final_prediction = original_contribs.loc[date, "prediction_total"]

        feature_cols = [
            col
            for col in original_contribs.columns
            if col not in ["intercept", "prediction_total"]
        ]
        contribs = original_contribs.loc[date, ["intercept"] + feature_cols]

        sorted_contribs = contribs[contribs != 0].sort_values(key=abs, ascending=False)[
            :8
        ]

        start_label = f"EUA_{prev_date.strftime('%m%d')}"
        final_label = f"EUA_{date.strftime('%m%d')}"
        labels = [start_label] + list(sorted_contribs.index) + [final_label]

        cumulative = starting_price
        x_pos = range(len(labels))

        ax.bar(0, starting_price, color="lightblue", alpha=0.7, width=0.6)

        for j, (feature, contrib) in enumerate(sorted_contribs.items(), 1):
            if contrib >= 0:
                ax.bar(
                    j, contrib, bottom=cumulative, color="green", alpha=0.7, width=0.6
                )
            else:
                ax.bar(
                    j,
                    abs(contrib),
                    bottom=cumulative + contrib,
                    color="red",
                    alpha=0.7,
                    width=0.6,
                )
            cumulative += contrib

        ax.bar(len(labels) - 1, final_prediction, color="navy", alpha=0.7, width=0.6)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{date.strftime('%m-%d')}", fontsize=10)
        ax.grid(True, alpha=0.3)

        total_change = final_prediction - starting_price
        ax.text(
            0.02,
            0.98,
            f"{total_change:+.1f}€",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(
        f"{model_name} Waterfall Charts ({start_date} to {end_date})", fontsize=14
    )
    plt.tight_layout()
    # plt.show()


def plot_regularized_contribution_heatmap(
    lasso_contrib,
    elastic_contrib,
    data,
    transformations_dict,
    start_date,
    end_date,
    target="EUA_benchmark_settlement",
    models="both",
):
    """
    Plot heatmap showing feature contributions over time for Lasso and/or ElasticNet.

    Parameters
    ----------
    lasso_contrib : pd.DataFrame
        Lasso model contributions dataframe
    elastic_contrib : pd.DataFrame
        ElasticNet model contributions dataframe
    data : pd.DataFrame
        Original dataset
    transformations_dict : dict
        Dictionary mapping column names to transformation types
    start_date : str
        Start date for the period to plot
    end_date : str
        End date for the period to plot
    target : str, optional
        Name of target column (default: 'EUA_benchmark_settlement')
    models : str or list, optional
        Which models to plot: 'both', 'lasso', 'elastic', or list of model names (default: 'both')
    """
    if isinstance(models, str):
        models = models.lower()
        if models == "both":
            models_to_plot = ["lasso", "elastic"]
        elif models in ["lasso", "elastic", "elasticnet"]:
            models_to_plot = [models.replace("elasticnet", "elastic")]
        else:
            raise ValueError(
                "models must be 'lasso', 'elastic', 'elasticnet', 'both', or a list"
            )
    else:
        models_to_plot = [m.lower().replace("elasticnet", "elastic") for m in models]

    contribs_dict = {}
    if "lasso" in models_to_plot:
        contribs_dict["Lasso"] = convert_regularized_contributions_to_original(
            lasso_contrib, data, transformations_dict, target
        )
    if "elastic" in models_to_plot:
        contribs_dict["ElasticNet"] = convert_regularized_contributions_to_original(
            elastic_contrib, data, transformations_dict, target
        )

    n_models = len(contribs_dict)
    fig, axes = plt.subplots(n_models, 1, figsize=(15, 5 * n_models))

    if n_models == 1:
        axes = [axes]

    for ax, (title, contribs) in zip(axes, contribs_dict.items()):
        date_mask = (contribs.index >= start_date) & (contribs.index <= end_date)
        plot_data = contribs[date_mask].copy()

        if plot_data.empty:
            continue

        feature_cols = [
            col
            for col in plot_data.columns
            if col not in ["intercept", "prediction_total"]
        ]
        heatmap_data = plot_data[["intercept"] + feature_cols].T

        vmax = heatmap_data.abs().max().max()
        im = ax.imshow(
            heatmap_data.values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax
        )

        ax.set_xticks(range(len(heatmap_data.columns)))
        ax.set_xticklabels(
            [d.strftime("%m-%d") for d in heatmap_data.columns], rotation=45, fontsize=8
        )
        ax.set_yticks(range(len(heatmap_data.index)))
        ax.set_yticklabels(heatmap_data.index, fontsize=9)

        ax.set_xlabel("Date")
        ax.set_ylabel("Features")
        ax.set_title(
            f"{title} Feature Contributions Heatmap ({start_date} to {end_date})"
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Contribution (€)", rotation=270, labelpad=20)

    plt.tight_layout()
    # plt.show()


def plot_regularized_average_waterfall(
    lasso_contrib,
    elastic_contrib,
    data,
    transformations_dict,
    start_date,
    end_date,
    target="EUA_benchmark_settlement",
):
    """
    Plot average waterfall charts for Lasso and ElasticNet models.

    Shows average feature contributions over the specified time period.

    Parameters
    ----------
    lasso_contrib : pd.DataFrame
        Lasso model contributions dataframe
    elastic_contrib : pd.DataFrame
        ElasticNet model contributions dataframe
    data : pd.DataFrame
        Original dataset
    transformations_dict : dict
        Dictionary mapping column names to transformation types
    start_date : str
        Start date for the period to average over
    end_date : str
        End date for the period to average over
    target : str, optional
        Name of target column (default: 'EUA_benchmark_settlement')
    """
    lasso_original = convert_regularized_contributions_to_original(
        lasso_contrib, data, transformations_dict, target
    )
    elastic_original = convert_regularized_contributions_to_original(
        elastic_contrib, data, transformations_dict, target
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, contribs, title in zip(
        axes, [lasso_original, elastic_original], ["Lasso", "ElasticNet"]
    ):
        date_mask = (contribs.index >= start_date) & (contribs.index <= end_date)
        plot_data = contribs[date_mask]

        if plot_data.empty:
            continue

        feature_cols = [
            col
            for col in contribs.columns
            if col not in ["intercept", "prediction_total"]
        ]
        avg_contribs = plot_data[["intercept"] + feature_cols].mean()

        avg_contribs_nonzero = avg_contribs[avg_contribs != 0]
        sorted_contribs = avg_contribs_nonzero.sort_values(key=abs, ascending=False)

        date_range = (plot_data.index >= start_date) & (plot_data.index <= end_date)
        prev_dates = [data[data.index < d].index[-1] for d in plot_data.index]
        avg_starting_price = data.loc[prev_dates, target].mean()
        avg_final_prediction = plot_data["prediction_total"].mean()

        labels = ["Avg Start"] + list(sorted_contribs.index) + ["Avg Final"]

        cumulative = avg_starting_price
        x_pos = range(len(labels))

        ax.bar(
            0,
            avg_starting_price,
            color="lightblue",
            alpha=0.7,
            label="Avg Starting Price",
        )

        for i, (feature, contrib) in enumerate(sorted_contribs.items(), 1):
            if contrib >= 0:
                ax.bar(i, contrib, bottom=cumulative, color="green", alpha=0.7)
                ax.annotate(
                    f"+{contrib:.2f}",
                    xy=(i, cumulative + contrib / 2),
                    ha="center",
                    va="center",
                    fontsize=8,
                )
            else:
                ax.bar(
                    i, abs(contrib), bottom=cumulative + contrib, color="red", alpha=0.7
                )
                ax.annotate(
                    f"{contrib:.2f}",
                    xy=(i, cumulative + contrib / 2),
                    ha="center",
                    va="center",
                    fontsize=8,
                )
            cumulative += contrib

        ax.bar(
            len(labels) - 1,
            avg_final_prediction,
            color="navy",
            alpha=0.7,
            label="Avg Final Prediction",
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("EUA Price (€)")
        ax.set_title(f"{title} Average Waterfall ({start_date} to {end_date})")
        ax.grid(True, alpha=0.3)

        total_change = avg_final_prediction - avg_starting_price
        ax.text(
            0.02,
            0.98,
            f"Avg Change: {total_change:+.2f}€",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout()
    # plt.show()


def compare_model_contributions(
    lasso_contrib,
    elastic_contrib,
    data,
    transformations_dict,
    start_date,
    end_date,
    target="EUA_benchmark_settlement",
):
    """
    Compare average feature contributions between Lasso and ElasticNet models.

    Parameters
    ----------
    lasso_contrib : pd.DataFrame
        Lasso model contributions dataframe
    elastic_contrib : pd.DataFrame
        ElasticNet model contributions dataframe
    data : pd.DataFrame
        Original dataset
    transformations_dict : dict
        Dictionary mapping column names to transformation types
    start_date : str
        Start date for the comparison period
    end_date : str
        End date for the comparison period
    target : str, optional
        Name of target column (default: 'EUA_benchmark_settlement')

    Returns
    -------
    pd.DataFrame
        DataFrame comparing average contributions between models
    """
    lasso_original = convert_regularized_contributions_to_original(
        lasso_contrib, data, transformations_dict, target
    )
    elastic_original = convert_regularized_contributions_to_original(
        elastic_contrib, data, transformations_dict, target
    )

    date_mask = (lasso_original.index >= start_date) & (
        lasso_original.index <= end_date
    )
    lasso_period = lasso_original[date_mask]
    elastic_period = elastic_original[date_mask]

    feature_cols = [
        col
        for col in lasso_period.columns
        if col not in ["intercept", "prediction_total"]
    ]

    lasso_avg = lasso_period[["intercept"] + feature_cols].mean()
    elastic_avg = elastic_period[["intercept"] + feature_cols].mean()

    comparison_df = pd.DataFrame(
        {
            "Lasso": lasso_avg,
            "ElasticNet": elastic_avg,
            "Difference": elastic_avg - lasso_avg,
        }
    )

    comparison_df = comparison_df.sort_values("Difference", key=abs, ascending=False)

    # logging.debug(f"\nAverage Feature Contributions Comparison ({start_date} to {end_date}):")
    # logging.debug(comparison_df.round(3))

    return comparison_df


def calculate_directional_trading_returns_filtered_trading_cost(
    results_df,
    transaction_cost=0.2,
    window_days=30,
    high_range_col="actual_high_7_mean",
    low_range_col="actual_low_7_mean",
    start_date=None,
):
    """
    Calculate directional trading returns with filtering and transaction costs.

    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing actual and predicted values
    transaction_cost : float
        Transaction cost per trade (default: 0.2)
    window_days : int
        Number of days to analyze (default: 30)
    high_range_col : str
        Column name for high range values (default: "actual_high_7_mean")
    low_range_col : str
        Column name for low range values (default: "actual_low_7_mean")
    start_date : str or pd.Timestamp, optional
        End date for the analysis window. If None, uses the last available date.
        Goes back window_days from this date.
    """
    models = {
        # 'ARIMA': 'arima_predicted',
        # 'Linear Regression': 'lr_predicted',
        "Lasso": "lasso_predicted_original",
    }

    results_df = results_df.copy()
    results_df["date"] = pd.to_datetime(results_df["date"])

    # Determine the end date for the analysis window
    if start_date is None:
        # Use the last available date
        end_date = results_df["date"].max()
    else:
        # Convert start_date to datetime if it's a string
        end_date = pd.to_datetime(start_date)

    # Filter data up to and including the end_date
    results_df = (
        results_df[results_df["date"] <= end_date]
        .sort_values("date")
        .reset_index(drop=True)
    )

    all_trading_results = {}

    for model_name, pred_col in models.items():
        daily_results = []

        for i in range(1, len(results_df)):
            actual_open = results_df.iloc[i]["actual_open"]
            current_prediction = results_df.iloc[i][pred_col]
            current_actual = results_df.iloc[i]["actual"]
            high_range = results_df.iloc[i][high_range_col]
            low_range = results_df.iloc[i][low_range_col]

            if pd.isna(high_range) or pd.isna(low_range) or pd.isna(current_prediction):
                continue

            predicted_profit = abs(current_prediction - actual_open)

            if (
                low_range <= current_prediction <= high_range
                and predicted_profit > transaction_cost
            ):
                signal = "BUY" if current_prediction > actual_open else "SELL"
                pnl = (
                    (current_actual - actual_open - transaction_cost)
                    if signal == "BUY"
                    else (actual_open - current_actual - transaction_cost)
                )
                percent_return = (pnl / actual_open) * 100
                trade_executed = True
            else:
                signal = "NO_TRADE"
                pnl = 0
                percent_return = 0
                trade_executed = False

            daily_results.append(
                {
                    "date": results_df.iloc[i]["date"],
                    "signal": signal,
                    "daily_pnl": pnl,
                    "daily_percent": percent_return,
                    "actual_open": actual_open,
                    "prediction": current_prediction,
                    "current_actual": current_actual,
                    "high_range": high_range,
                    "low_range": low_range,
                    "predicted_profit": predicted_profit,
                    "trade_executed": trade_executed,
                }
            )

        df_results = pd.DataFrame(daily_results)
        df_results["cumulative_pnl"] = df_results["daily_pnl"].cumsum()
        df_results["cumulative_percent"] = df_results["daily_percent"].cumsum()

        if window_days:
            df_results = df_results.tail(window_days)
            df_results["cumulative_pnl"] = df_results["daily_pnl"].cumsum()
            df_results["cumulative_percent"] = df_results["daily_percent"].cumsum()

        all_trading_results[model_name] = df_results

    summary_stats = []
    for model_name, df in all_trading_results.items():
        total_pnl = df["daily_pnl"].sum()
        total_percent = df["daily_percent"].sum()
        executed_trades = df[df["trade_executed"] == True]
        win_rate = (
            (executed_trades["daily_pnl"] > 0).mean() * 100
            if len(executed_trades) > 0
            else 0
        )
        trade_execution_rate = df["trade_executed"].mean() * 100

        summary_stats.append(
            {
                "Model": model_name,
                "Total_PnL": total_pnl,
                "Total_Percent": total_percent,
                "Win_Rate": win_rate,
                "Total_Days": len(df),
                "Executed_Trades": len(executed_trades),
                "Execution_Rate": trade_execution_rate,
            }
        )

    summary_df = pd.DataFrame(summary_stats)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Cumulative Absolute Returns (Filtered + TC)",
            "Cumulative Percentage Returns (Filtered + TC)",
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]],
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, (model_name, df) in enumerate(all_trading_results.items()):
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["cumulative_pnl"],
                mode="lines",
                name=model_name,
                line=dict(color=colors[i], width=3),
                hovertemplate="<b>%{fullData.name}</b><br>Date: %{x}<br>Cumulative P&L: %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["cumulative_percent"],
                mode="lines",
                name=model_name,
                line=dict(color=colors[i], width=3),
                showlegend=False,
                hovertemplate="<b>%{fullData.name}</b><br>Date: %{x}<br>Cumulative %: %{y:.2f}%<extra></extra>",
            ),
            row=1,
            col=2,
        )

    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=1, col=2)

    range_label = f"{high_range_col.replace('actual_', '').replace('_mean', '').upper()}/{low_range_col.replace('actual_', '').replace('_mean', '').upper()}"

    fig.update_layout(
        title=f"Filtered Directional Trading - Last {window_days} Days ({range_label} Range, TC: {transaction_cost})",
        height=500,
        showlegend=True,
        hovermode="x unified",
    )

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="Cumulative P&L", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative % Return", row=1, col=2)

    # fig.show()

    logging.debug("=" * 95)
    logging.debug(
        f"FILTERED DIRECTIONAL TRADING - Last {window_days} Days ({range_label} Range, TC: {transaction_cost})"
    )
    logging.debug("=" * 95)
    logging.debug(
        f"{'Model':<18} {'Total P&L':<12} {'Total %':<12} {'Win Rate':<12} {'Exec Trades':<12} {'Exec Rate %':<12}"
    )
    logging.debug("-" * 95)

    for _, row in summary_df.iterrows():
        logging.debug(
            f"{row['Model']:<18} {row['Total_PnL']:<12.3f} {row['Total_Percent']:<12.2f} {row['Win_Rate']:<12.1f} {row['Executed_Trades']:<12} {row['Execution_Rate']:<12.1f}"
        )

    return all_trading_results, summary_df


def calculate_directional_accuracy(
    results_df,
    window_days=90,
    model_col="lasso_predicted_original",
    start_date=None,
):
    """
    Calculate directional accuracy for Lasso predictions.

    Compares predicted direction (sign(predicted_{t+1} - actual_t))
    with actual direction (sign(actual_{t+1} - actual_t)).

    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing actual and predicted values
    window_days : int
        Number of days to analyze (default: 90)
    model_col : str
        Column name for predictions (default: "lasso_predicted_original")
    start_date : str or pd.Timestamp, optional
        End date for the analysis window. If None, uses the last available date.
        Goes back window_days from this date.

    Returns:
    --------
    None (logging.debugs accuracy percentage)
    """
    results_df = results_df.copy()
    results_df["date"] = pd.to_datetime(results_df["date"])
    results_df = results_df.sort_values("date").reset_index(drop=True)

    # Determine the end date for the analysis window
    if start_date is None:
        # Use the last available date
        end_date = results_df["date"].max()
    else:
        # Convert start_date to datetime if it's a string
        end_date = pd.to_datetime(start_date)

    # Filter data up to and including the end_date, then get valid data
    filtered_df = results_df[results_df["date"] <= end_date]
    valid_data = filtered_df[
        filtered_df["actual"].notna() & filtered_df[model_col].notna()
    ].tail(window_days + 1)  # Need +1 to get actual_{t+1}

    if len(valid_data) < 2:
        logging.debug(
            "Not enough data for directional accuracy analysis (need at least 2 days)"
        )
        return

    correct_predictions = 0
    total_predictions = 0

    # Iterate through each day (except the last one, since we need actual_{t+1})
    for i in range(len(valid_data) - 1):
        actual_t = valid_data.iloc[i]["actual"]
        predicted_t_plus_1 = valid_data.iloc[i][model_col]
        actual_t_plus_1 = valid_data.iloc[i + 1]["actual"]

        # Skip if any value is NaN
        if pd.isna(actual_t) or pd.isna(predicted_t_plus_1) or pd.isna(actual_t_plus_1):
            continue

        # Calculate predicted direction: sign(predicted_{t+1} - actual_t)
        predicted_direction = np.sign(predicted_t_plus_1 - actual_t)

        # Calculate actual direction: sign(actual_{t+1} - actual_t)
        actual_direction = np.sign(actual_t_plus_1 - actual_t)

        # Count as correct if signs match
        if predicted_direction == actual_direction:
            correct_predictions += 1

        total_predictions += 1

    if total_predictions == 0:
        logging.debug("Not enough valid predictions for directional accuracy analysis")
        return

    accuracy = (correct_predictions / total_predictions) * 100

    # Get date range for display
    start_date_str = valid_data.iloc[0]["date"].strftime("%Y-%m-%d")
    end_date_str = valid_data.iloc[-2]["date"].strftime(
        "%Y-%m-%d"
    )  # -2 because we need actual_{t+1}

    logging.debug(
        f"Accuracy on {window_days} days is {accuracy:.2f}% "
        f"(from {start_date_str} to {end_date_str})"
    )


def select_eua_features_sfs(
    data, feature_columns, k_features="best", forward=True, floating=False, cv_folds=5
):
    """
    Select features using Sequential Feature Selection (SFS) with time series cross-validation.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with features and target
    feature_columns : list
        List of candidate feature column names
    k_features : int or str, optional
        Number of features to select or 'best' for automatic selection (default: 'best')
    forward : bool, optional
        If True, use forward selection; if False, use backward elimination (default: True)
    floating : bool, optional
        Whether to use floating selection (can add/remove features) (default: False)
    cv_folds : int, optional
        Number of cross-validation folds (default: 5)

    Returns
    -------
    dict
        Dictionary containing selected_features, sfs_model, scaler, optimal_n_features,
        cv_scores, transformations, results_df, and analysis_data
    """

    def test_stationarity(series):
        series_clean = series.dropna()
        if len(series_clean) < 3 or series_clean.nunique() <= 1:
            return True
        return adfuller(series_clean)[1] < 0.05

    def log_diff(series):
        return np.log(series).diff()

    def first_diff(series):
        return series.diff()

    def weekly_pct_change(series):
        weekly = series.resample("W").last()
        return weekly.pct_change().reindex(series.index, method="ffill")

    def get_best_transformation(series, var_name):
        if test_stationarity(series):
            return series, "levels"

        if var_name.startswith("COT_"):
            transforms = {"weekly_pct": weekly_pct_change, "first_diff": first_diff}
        elif any(
            var_name.startswith(prefix)
            for prefix in ["TTF", "Coal", "German", "Brent", "EUA"]
        ):
            transforms = {"log_diff": log_diff, "first_diff": first_diff}
        else:
            transforms = {"log_diff": log_diff, "first_diff": first_diff}

        for transform_name, transform_func in transforms.items():
            try:
                transformed = transform_func(series)
                if test_stationarity(transformed):
                    return transformed, transform_name
            except:
                continue

        return first_diff(series), "first_diff"

    df_stationary = data.copy()
    transformations = {}

    for col in data.columns:
        if col != "EUA_benchmark_settlement":
            df_stationary[col], transformations[col] = get_best_transformation(
                data[col], col
            )

    target_stationary = log_diff(data["EUA_benchmark_settlement"])
    transformations["EUA_benchmark_settlement"] = "log_diff"
    target_t_plus_1 = target_stationary.shift(-1)

    feature_cols = [
        col
        for col in feature_columns
        if col in df_stationary.columns and col != "EUA_benchmark_settlement"
    ]

    analysis_df = pd.DataFrame(
        {"target": target_t_plus_1, **{col: df_stationary[col] for col in feature_cols}}
    ).dropna()

    X = analysis_df[feature_cols]
    y = analysis_df["target"]

    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns, index=X.index)

    estimator = LinearRegression()
    tscv = TimeSeriesSplit(n_splits=cv_folds)

    sfs = SFS(
        estimator=estimator,
        k_features=k_features,
        forward=forward,
        floating=floating,
        scoring="neg_mean_squared_error",  #'r2','neg_mean_squared_error', 'neg_mean_absolute_error'
        cv=tscv,
        n_jobs=-1,
    )

    sfs.fit(X_scaled, y)

    selected_features = list(sfs.k_feature_names_)

    results_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "selected": [col in selected_features for col in feature_cols],
            "transformation": [transformations[col] for col in feature_cols],
        }
    )

    selected_results = results_df[results_df["selected"]].copy()
    non_selected_results = results_df[~results_df["selected"]].copy()
    results_df = pd.concat([selected_results, non_selected_results]).reset_index(
        drop=True
    )

    return {
        "selected_features": selected_features,
        "sfs_model": sfs,
        "scaler": scaler,
        "optimal_n_features": len(selected_features),
        "cv_scores": sfs.subsets_,
        "transformations": transformations,
        "results_df": results_df,
        "analysis_data": analysis_df,
    }


def select_eua_features_rfecv(data, feature_columns, cv_folds=5, min_features=1):
    """
    Select features using Recursive Feature Elimination with Cross-Validation (RFECV).

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with features and target
    feature_columns : list
        List of candidate feature column names
    cv_folds : int, optional
        Number of cross-validation folds (default: 5)
    min_features : int, optional
        Minimum number of features to select (default: 1)

    Returns
    -------
    dict
        Dictionary containing selected_features, rfecv_model, scaler, optimal_n_features,
        cv_scores, transformations, results_df, and analysis_data
    """

    def test_stationarity(series):
        series_clean = series.dropna()
        if len(series_clean) < 3 or series_clean.nunique() <= 1:
            return True
        return adfuller(series_clean)[1] < 0.05

    def log_diff(series):
        return np.log(series).diff()

    def first_diff(series):
        return series.diff()

    def weekly_pct_change(series):
        weekly = series.resample("W").last()
        return weekly.pct_change().reindex(series.index, method="ffill")

    def get_best_transformation(series, var_name):
        if test_stationarity(series):
            return series, "levels"

        if var_name.startswith("COT_"):
            transforms = {"weekly_pct": weekly_pct_change, "first_diff": first_diff}
        elif any(
            var_name.startswith(prefix)
            for prefix in ["TTF", "Coal", "German", "Brent", "EUA"]
        ):
            transforms = {"log_diff": log_diff, "first_diff": first_diff}
        else:
            transforms = {"log_diff": log_diff, "first_diff": first_diff}

        for transform_name, transform_func in transforms.items():
            try:
                transformed = transform_func(series)
                if test_stationarity(transformed):
                    return transformed, transform_name
            except:
                continue

        return first_diff(series), "first_diff"

    df_stationary = data.copy()
    transformations = {}

    for col in data.columns:
        if col != "EUA_benchmark_settlement":
            df_stationary[col], transformations[col] = get_best_transformation(
                data[col], col
            )

    target_stationary = log_diff(data["EUA_benchmark_settlement"])
    transformations["EUA_benchmark_settlement"] = "log_diff"
    target_t_plus_1 = target_stationary.shift(-1)

    feature_cols = [
        col
        for col in feature_columns
        if col in df_stationary.columns and col != "EUA_benchmark_settlement"
    ]

    analysis_df = pd.DataFrame(
        {"target": target_t_plus_1, **{col: df_stationary[col] for col in feature_cols}}
    ).dropna()

    X = analysis_df[feature_cols]
    y = analysis_df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    estimator = LinearRegression()
    tscv = TimeSeriesSplit(n_splits=cv_folds)

    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=tscv,
        scoring="neg_mean_squared_error",
        min_features_to_select=min_features,
        n_jobs=-1,
    )

    rfecv.fit(X_scaled, y)

    selected_features = X.columns[rfecv.support_].tolist()
    feature_rankings = dict(zip(X.columns, rfecv.ranking_))

    results_df = pd.DataFrame(
        {
            "feature": X.columns,
            "selected": rfecv.support_,
            "ranking": rfecv.ranking_,
            "transformation": [transformations[col] for col in X.columns],
        }
    ).sort_values("ranking")

    return {
        "selected_features": selected_features,
        "rfecv_model": rfecv,
        "scaler": scaler,
        "optimal_n_features": rfecv.n_features_,
        "cv_scores": rfecv.cv_results_,
        "transformations": transformations,
        "results_df": results_df,
        "analysis_data": analysis_df,
    }


def save_forecast_comparison_csv(
    results_df,
    mean_trading_results=None,
    maxmin_trading_results=None,
    output_path=None,
    predictions_folder=None,
    filename="forecast_comparison.csv",
):
    """
    Create and save a CSV file with Date, Forecast Price, Actual Open, Actual Settlement,
    PnL for Mean strategy, PnL for Max/Min strategy, and Cumulative PnL for each strategy.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing date, predictions, and actual values
        Expected columns: 'date', 'lasso_predicted_original' (or similar),
        'actual_open', 'actual' (settlement)
    mean_trading_results : dict, optional
        Dictionary returned from calculate_directional_trading_returns_filtered_trading_cost
        for Mean strategy. Should contain 'Lasso' key with DataFrame containing 'date', 'daily_pnl', 'cumulative_pnl'
    maxmin_trading_results : dict, optional
        Dictionary returned from calculate_directional_trading_returns_filtered_trading_cost
        for Max/Min strategy. Should contain 'Lasso' key with DataFrame containing 'date', 'daily_pnl', 'cumulative_pnl'
    output_path : str, optional
        Full GCS path to save the file (e.g., 'predictions/forecast_comparison.csv')
        If None, will use predictions_folder + filename
    predictions_folder : str, optional
        Folder path within GCS bucket (e.g., 'predictions/predictions_2025-11-13')
        Required if output_path is None
    filename : str, optional
        Name of the CSV file (default: 'forecast_comparison.csv')

    Returns
    -------
    str
        Path where the file was saved
    """
    # Create a copy to avoid modifying original
    df = results_df.copy()

    # Ensure date is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])

    # Sort by date first
    df = df.sort_values("date").reset_index(drop=True)

    # Identify forecast column (prefer lasso_predicted_original, fallback to others)
    forecast_col = None
    for col in ["lasso_predicted_original", "predicted_original", "arima_predicted"]:
        if col in df.columns:
            forecast_col = col
            break

    if forecast_col is None:
        raise ValueError(
            "No forecast column found. Expected one of: 'lasso_predicted_original', "
            "'predicted_original', 'arima_predicted'"
        )

    # Identify actual settlement column
    settlement_col = None
    for col in ["actual", "actual_original", "EUA_benchmark_settlement"]:
        if col in df.columns:
            settlement_col = col
            break

    if settlement_col is None:
        raise ValueError(
            "No settlement column found. Expected one of: 'actual', "
            "'actual_original', 'EUA_benchmark_settlement'"
        )

    # Identify actual open column
    open_col = None
    for col in ["actual_open", "EUA_benchmark_open"]:
        if col in df.columns:
            open_col = col
            break

    if open_col is None:
        # Try to get from data if available
        log_message(
            "Warning: 'actual_open' not found in results_df. Will use NaN for Actual Open.",
            "warning",
        )
        df["actual_open"] = np.nan
        open_col = "actual_open"

    # Extract PnL data from trading results if provided
    mean_pnl_data = None
    maxmin_pnl_data = None

    # Handle tuple return (all_trading_results, summary_df) or just dict
    if mean_trading_results is not None:
        if isinstance(mean_trading_results, tuple):
            mean_trading_results = mean_trading_results[
                0
            ]  # Extract the dict from tuple

    if maxmin_trading_results is not None:
        if isinstance(maxmin_trading_results, tuple):
            maxmin_trading_results = maxmin_trading_results[
                0
            ]  # Extract the dict from tuple

    if (
        mean_trading_results is not None
        and isinstance(mean_trading_results, dict)
        and "Lasso" in mean_trading_results
    ):
        mean_pnl_data = mean_trading_results["Lasso"][
            ["date", "daily_pnl", "cumulative_pnl"]
        ].copy()
        mean_pnl_data["date"] = pd.to_datetime(mean_pnl_data["date"])
        mean_pnl_data = mean_pnl_data.rename(
            columns={"daily_pnl": "mean_pnl", "cumulative_pnl": "mean_cumulative_pnl"}
        )

    if (
        maxmin_trading_results is not None
        and isinstance(maxmin_trading_results, dict)
        and "Lasso" in maxmin_trading_results
    ):
        maxmin_pnl_data = maxmin_trading_results["Lasso"][
            ["date", "daily_pnl", "cumulative_pnl"]
        ].copy()
        maxmin_pnl_data["date"] = pd.to_datetime(maxmin_pnl_data["date"])
        maxmin_pnl_data = maxmin_pnl_data.rename(
            columns={
                "daily_pnl": "maxmin_pnl",
                "cumulative_pnl": "maxmin_cumulative_pnl",
            }
        )

    # Merge PnL data with main dataframe
    if mean_pnl_data is not None:
        df = df.merge(mean_pnl_data, on="date", how="left")
    else:
        df["mean_pnl"] = np.nan
        df["mean_cumulative_pnl"] = 0.0

    if maxmin_pnl_data is not None:
        df = df.merge(maxmin_pnl_data, on="date", how="left")
    else:
        df["maxmin_pnl"] = np.nan
        df["maxmin_cumulative_pnl"] = 0.0

    # Create the comparison dataframe
    comparison_df = pd.DataFrame(
        {
            "Date": df["date"].dt.strftime("%Y-%m-%d"),
            "Forecast Price": df[forecast_col],
            "Actual Open": df[open_col],
            "Actual Settlement": df[settlement_col],
            "Mean Strategy PnL": df["mean_pnl"],
            "Max/Min Strategy PnL": df["maxmin_pnl"],
            "Mean Strategy Cumulative PnL": df["mean_cumulative_pnl"],
            "Max/Min Strategy Cumulative PnL": df["maxmin_cumulative_pnl"],
        }
    )

    # Remove rows where all values except Date are NaN
    comparison_df = comparison_df.dropna(
        subset=["Forecast Price", "Actual Open", "Actual Settlement"], how="all"
    )

    # Sort by date
    comparison_df = comparison_df.sort_values("Date").reset_index(drop=True)

    # Determine output path
    if output_path is None:
        if predictions_folder is None:
            # Use today's date as default
            date_today = datetime.now().strftime("%Y-%m-%d")
            predictions_folder = f"predictions/predictions_{date_today}"
        output_path = f"{predictions_folder}/{filename}"

    # Save to GCS
    success = gcs_manager.upload_dataframe(comparison_df, output_path, index=False)

    if success:
        log_message(
            f"Saved forecast comparison CSV to gs://dashboard_data_ge/{output_path}",
            "info",
        )
        logging.debug(f"Forecast comparison CSV saved: {len(comparison_df)} rows")
        logging.debug(
            f"Date range: {comparison_df['Date'].min()} to {comparison_df['Date'].max()}"
        )
        if "Mean Strategy PnL" in comparison_df.columns:
            total_mean = comparison_df["Mean Strategy PnL"].fillna(0).sum()
            total_maxmin = comparison_df["Max/Min Strategy PnL"].fillna(0).sum()
            final_mean_cumulative = (
                comparison_df["Mean Strategy Cumulative PnL"].iloc[-1]
                if len(comparison_df) > 0
                else 0
            )
            final_maxmin_cumulative = (
                comparison_df["Max/Min Strategy Cumulative PnL"].iloc[-1]
                if len(comparison_df) > 0
                else 0
            )
            logging.debug(f"Total Mean Strategy PnL: {total_mean:.2f}")
            logging.debug(f"Total Max/Min Strategy PnL: {total_maxmin:.2f}")
            logging.debug(f"Final Mean Strategy Cumulative PnL: {final_mean_cumulative:.2f}")
            logging.debug(
                f"Final Max/Min Strategy Cumulative PnL: {final_maxmin_cumulative:.2f}"
            )
    else:
        log_message(
            f"Failed to save forecast comparison CSV to gs://dashboard_data_ge/{output_path}",
            "error",
        )

    return output_path


def plot_forecasts_interactive(df, show_lr=False, last_days=None):
    """
    Create interactive Plotly chart showing forecasts vs actuals.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: date, actual, lasso_predicted_original,
        lasso_ci_lower_original, lasso_ci_upper_original, and optionally lr_predicted
    show_lr : bool, optional
        Whether to show Linear Regression forecasts (default: False)
    last_days : int, optional
        If specified, only show the last N days (default: None)

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure
    """
    if last_days:
        dates_temp = pd.to_datetime(df["date"])
        cutoff_date = dates_temp.max() - pd.Timedelta(days=last_days)
        df = df[dates_temp >= cutoff_date].reset_index(drop=True)

    fig = go.Figure()

    dates = pd.to_datetime(df["date"])
    actual_mask = ~df["actual"].isna()
    forecast_mask = df["actual"].isna()
    lasso_mask = ~df["lasso_predicted_original"].isna()

    fig.add_trace(
        go.Scatter(
            x=dates[actual_mask],
            y=df["actual"][actual_mask],
            mode="lines+markers",
            name="Actual",
            line=dict(color="black", width=2),
            hovertemplate="<b>Actual</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>",
        )
    )

    historical_mask = lasso_mask & actual_mask
    if historical_mask.any():
        fig.add_trace(
            go.Scatter(
                x=dates[historical_mask],
                y=df["lasso_ci_upper_original"][historical_mask],
                fill=None,
                mode="lines",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=dates[historical_mask],
                y=df["lasso_ci_lower_original"][historical_mask],
                fill="tonexty",
                mode="lines",
                name="Lasso CI (Historical)",
                line=dict(color="rgba(0,0,0,0)"),
                fillcolor="rgba(255,0,0,0.2)",
                hovertemplate="<b>Lasso CI</b><br>Date: %{x}<br>Lower: %{y:.2f}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=dates[historical_mask],
                y=df["lasso_predicted_original"][historical_mask],
                mode="lines+markers",
                name="Lasso (Historical)",
                line=dict(color="red", width=2, dash="dash"),
                hovertemplate="<b>Lasso</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>",
            )
        )

    forecast_lasso_mask = lasso_mask & forecast_mask
    if forecast_lasso_mask.any():
        fig.add_trace(
            go.Scatter(
                x=dates[forecast_lasso_mask],
                y=df["lasso_ci_upper_original"][forecast_lasso_mask],
                mode="lines",
                name="Lasso CI Upper (Forecast)",
                line=dict(color="rgba(34,139,34,0.3)", width=1),
                hovertemplate="<b>Lasso CI Upper</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=dates[forecast_lasso_mask],
                y=df["lasso_ci_lower_original"][forecast_lasso_mask],
                fill="tonexty",
                mode="lines",
                name="Lasso CI (Forecast)",
                line=dict(color="rgba(34,139,34,0.3)", width=1),
                fillcolor="rgba(34,139,34,0.3)",
                hovertemplate="<b>Lasso CI Lower</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=dates[forecast_lasso_mask],
                y=df["lasso_predicted_original"][forecast_lasso_mask],
                mode="lines+markers",
                name="Lasso (Forecast)",
                line=dict(color="forestgreen", width=3),
                marker=dict(size=8, color="forestgreen"),
                hovertemplate="<b>Lasso Forecast</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>",
            )
        )

    if show_lr:
        lr_mask = ~df["lr_predicted"].isna()
        historical_lr_mask = lr_mask & actual_mask
        forecast_lr_mask = lr_mask & forecast_mask

        if historical_lr_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=dates[historical_lr_mask],
                    y=df["lr_ci_upper"][historical_lr_mask],
                    fill=None,
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=dates[historical_lr_mask],
                    y=df["lr_ci_lower"][historical_lr_mask],
                    fill="tonexty",
                    mode="lines",
                    name="LR CI (Historical)",
                    line=dict(color="rgba(0,0,0,0)"),
                    fillcolor="rgba(0,0,255,0.15)",
                    hovertemplate="<b>LR CI</b><br>Date: %{x}<br>Lower: %{y:.2f}<extra></extra>",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=dates[historical_lr_mask],
                    y=df["lr_predicted"][historical_lr_mask],
                    mode="lines",
                    name="Linear Reg (Historical)",
                    line=dict(color="blue", width=2, dash="dot"),
                    hovertemplate="<b>Linear Reg</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>",
                )
            )

        if forecast_lr_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=dates[forecast_lr_mask],
                    y=df["lr_ci_upper"][forecast_lr_mask],
                    mode="lines",
                    name="LR CI Upper (Forecast)",
                    line=dict(color="rgba(128,0,128,0.4)", width=1),
                    hovertemplate="<b>LR CI Upper</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=dates[forecast_lr_mask],
                    y=df["lr_ci_lower"][forecast_lr_mask],
                    fill="tonexty",
                    mode="lines",
                    name="LR CI (Forecast)",
                    line=dict(color="rgba(128,0,128,0.4)", width=1),
                    fillcolor="rgba(128,0,128,0.25)",
                    hovertemplate="<b>LR CI Lower</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=dates[forecast_lr_mask],
                    y=df["lr_predicted"][forecast_lr_mask],
                    mode="lines+markers",
                    name="Linear Reg (Forecast)",
                    line=dict(color="purple", width=3),
                    marker=dict(size=8, color="purple"),
                    hovertemplate="<b>LR Forecast</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>",
                )
            )

    fig.update_layout(height=500, hovermode="x unified")
    return fig


if __name__ == "__main__":
    setup_output_directories()
    visualizer = Visualizer(OUTPUT_CONFIG["visualization_path"])

    # Create predictions folder for today (GCS path)
    date_today = datetime.now().strftime("%Y-%m-%d")
    predictions_folder = f"predictions/predictions_{date_today}"
    log_message(
        f"Using predictions folder: gs://dashboard_data_ge/{predictions_folder}", "info"
    )

    # Delete existing predictions folder to start fresh
    log_message(
        f"Deleting existing folder: gs://dashboard_data_ge/{predictions_folder}", "info"
    )
    gcs_manager.delete_folder(predictions_folder)

    # Check for existing data_after_preprocessor.csv in GCS
    data_after_preprocessor_path = f"{predictions_folder}/data_after_preprocessor.csv"

    # Always process new data (removed if condition)
    log_message("Step 1: Data Processing", "info")
    processor = DataProcessor()

    data = processor.load_and_process()
    logging.debug(f"Data loaded and cleaned. Shape: {data.shape}")
    logging.debug(f"Date range: {data.index.min()} to {data.index.max()}")

    summary = processor.get_summary()
    data = data.groupby(data.index).last()

    # Ensure index has a name for consistency when saving
    if data.index.name is None:
        data.index.name = "date"

    gcs_manager.upload_dataframe(
        data.reset_index(), data_after_preprocessor_path, index=False
    )
    log_message(
        f"Saved data_after_preprocessor.csv to gs://dashboard_data_ge/{predictions_folder}",
        "info",
    )

    # Data visualization
    target_column = (
        processor.target.replace("_log", "")
        if processor.log_transformed
        else processor.target
    )

    warnings.filterwarnings("ignore")

    # Check for existing results_df.csv in GCS
    results_df_path = f"{predictions_folder}/results_df.csv"

    if gcs_manager.file_exists(results_df_path):
        log_message(
            "Loading existing results_df.csv from GCS, skipping time series analysis",
            "info",
        )
        results_df = gcs_manager.download_dataframe(results_df_path)
        if "date" in results_df.columns:
            results_df["date"] = pd.to_datetime(results_df["date"])
        logging.debug(f"Loaded results with shape: {results_df.shape}")
        ts_day_ahead = results_df.loc[results_df["date"].idxmax()]
        results_df = results_df.drop(results_df["date"].idxmax())
        # Set dummy model_config and other variables to avoid errors
        model_config = None
        test_eq_df = None
        final_forecast_df = None
        eq_df = None
    else:
        results_df, model_config, test_eq_df = unified_model_comparison(
            data, "EUA_benchmark_settlement", "2025-01-01"
        )
        final_forecast_df, eq_df = get_unified_5day_forecasts(
            data, "EUA_benchmark_settlement", model_config
        )

        gcs_manager.upload_dataframe(results_df, results_df_path, index=False)
        log_message(
            f"Saved results_df.csv to gs://dashboard_data_ge/{predictions_folder}",
            "info",
        )

        ########################################################
        # Load results and prepare for prediction
        ########################################################
        results_df = gcs_manager.download_dataframe(results_df_path)
        if "date" in results_df.columns:
            results_df["date"] = pd.to_datetime(results_df["date"])
        logging.debug(f"Loaded results with shape: {results_df.shape}")
        results_df.head()
        ts_day_ahead = results_df.loc[results_df["date"].idxmax()]
        results_df = results_df.drop(results_df["date"].idxmax())

    data[data.index >= "2025-01-01"].head(10)
    results_df["actual_open"] = results_df["date"].map(data["EUA_benchmark_open"])
    results_df["actual_high"] = results_df["date"].map(data["EUA_benchmark_high"])
    results_df["actual_low"] = results_df["date"].map(data["EUA_benchmark_low"])
    # NOTE: here results_df has actual, actual_open, actual_high, actual_low columns

    data = data.drop("EUA_benchmark_open", axis=1)
    data = data.drop("EUA_benchmark_high", axis=1)
    data = data.drop("EUA_benchmark_low", axis=1)

    logging.debug(f"Last date in data: {max(data.index)}")

    # Check for existing eua_feature_selection_results.csv in GCS
    feature_selection_path = f"{predictions_folder}/eua_feature_selection_results.csv"

    if gcs_manager.file_exists(feature_selection_path):
        log_message(
            "Loading existing eua_feature_selection_results.csv from GCS, skipping feature selection",
            "info",
        )
        output_df = gcs_manager.download_dataframe(feature_selection_path)
    else:
        output_df = select_eua_features(data)
        gcs_manager.upload_dataframe(output_df, feature_selection_path, index=False)
        log_message(
            f"Saved eua_feature_selection_results.csv to gs://dashboard_data_ge/{predictions_folder}",
            "info",
        )

    selected_cols = [
        "TTF_front_low",
        "Coal_front_open_interest",
        "German_Power_front_settlement",
        "Brent_Crude_benchmark_low",
        "Auction_Total_Allowances_Bid",
        "weather_cdd_min",
        "op_call_bm_atm_far_vol",
        # 'COT_IF_Positions_short_total',
        "clean_dark_value",
        "fuel_switch_value",
        "wednesday",
        "EUA_benchmark_volume",
        "EUA_benchmark_open_interest",
        "COT_CU_Positions_long_total",
        "COT_CU_Positions_short_total",
        "COT_IF_Positions_long_total",
        "COT_IF_Positions_short_total",
        "COT_IFCI_Positions_long_total",
        "COT_IFCI_Positions_short_total",
        "COT_OWCO_Positions_long_total",
        "COT_OWCO_Positions_short_total",
        "COT_OFI_Positions_long_total",
        "COT_OFI_Positions_short_total",
        "COT_TTF_IF_Positions_long_total",
        "COT_TTF_IF_Positions_short_total",
        "COT_TTF_CU_Positions_long_total",
        "COT_TTF_CU_Positions_short_total",
        "COT_TTF_IFCI_Positions_long_total",
        "COT_TTF_IFCI_Positions_short_total",
        "COT_TTF_OFI_Positions_long_total",
        "COT_TTF_OFI_Positions_short_total",
    ]

    results = select_eua_features_rfecv(data, selected_cols, cv_folds=5, min_features=5)

    selected_cols = [
        "TTF_front_low",
        "Coal_front_open_interest",
        "German_Power_front_settlement",
        "Brent_Crude_benchmark_low",
        "Auction_Total_Allowances_Bid",
        "weather_cdd_min",
        "op_call_bm_atm_far_vol",
        # 'COT_IF_Positions_short_total',
        "clean_dark_value",
        "fuel_switch_value",
        "wednesday",
        "EUA_benchmark_volume",
        "EUA_benchmark_open_interest",
        "COT_CU_Positions_long_total",
        "COT_CU_Positions_short_total",
        "COT_IF_Positions_long_total",
        "COT_IF_Positions_short_total",
        "COT_IFCI_Positions_long_total",
        "COT_IFCI_Positions_short_total",
        "COT_OWCO_Positions_long_total",
        "COT_OWCO_Positions_short_total",
        "COT_OFI_Positions_long_total",
        "COT_OFI_Positions_short_total",
        "COT_TTF_IF_Positions_long_total",
        "COT_TTF_IF_Positions_short_total",
        "COT_TTF_CU_Positions_long_total",
        "COT_TTF_CU_Positions_short_total",
        "COT_TTF_IFCI_Positions_long_total",
        "COT_TTF_IFCI_Positions_short_total",
        "COT_TTF_OFI_Positions_long_total",
        "COT_TTF_OFI_Positions_short_total",
    ]

    sfs_results = select_eua_features_sfs(
        data, selected_cols, k_features=5, forward=False
    )

    # Main execution
    transformations_dict = dict(
        zip(output_df["column_name"], output_df["transformation"])
    )
    transformations_dict["EUA_benchmark_settlement"] = "log_diff"

    features_list = [
        "TTF_front_low",
        "weather_cdd_min",
        "op_call_bm_atm_far_vol",
        "fuel_switch_value",
        "COT_CU_Positions_long_total",
        "EUA_benchmark_volume",
        "COT_TTF_CU_Positions_long_total",
    ]

    # Check for existing eua_results.csv in GCS
    eua_results_path = f"{predictions_folder}/eua_results.csv"

    if gcs_manager.file_exists(eua_results_path):
        log_message(
            "Loading existing eua_results.csv from GCS, skipping lasso/elastic regression",
            "info",
        )
        results_df_with_regression = gcs_manager.download_dataframe(eua_results_path)
        if "date" in results_df_with_regression.columns:
            results_df_with_regression["date"] = pd.to_datetime(
                results_df_with_regression["date"]
            )
        # Set dummy variables to avoid errors
        lasso_results = None
        lasso_contrib = None
        day_ahead_prediction = None
        elastic_results = None
        elastic_contrib = None
        # Need to extract day_ahead_prediction from the loaded results
        latest_date = results_df_with_regression["date"].max()
        next_working_day = pd.bdate_range(start=latest_date, periods=2)[1]
        last_row_idx = (
            results_df_with_regression[
                results_df_with_regression["date"] == latest_date
            ].index[0]
            if len(
                results_df_with_regression[
                    results_df_with_regression["date"] == latest_date
                ]
            )
            > 0
            else results_df_with_regression.index[-1]
        )
        day_ahead_prediction = {
            "date": next_working_day,
            "predicted_original": results_df_with_regression.loc[
                last_row_idx, "lasso_predicted_original"
            ]
            if "lasso_predicted_original" in results_df_with_regression.columns
            else None,
            "ci_lower_original": results_df_with_regression.loc[
                last_row_idx, "lasso_ci_lower_original"
            ]
            if "lasso_ci_lower_original" in results_df_with_regression.columns
            else None,
            "ci_upper_original": results_df_with_regression.loc[
                last_row_idx, "lasso_ci_upper_original"
            ]
            if "lasso_ci_upper_original" in results_df_with_regression.columns
            else None,
            "calculation_report": "Loaded from existing file",
        }
        # Let us save the day ahead prediction to a csv file
        day_ahead_prediction_path = f"{predictions_folder}/day_ahead_prediction.csv"
        day_ahead_df = pd.DataFrame(
            {
                "Date": [datetime.today().date()],
                "Price": [day_ahead_prediction["predicted_original"]],
            }
        )
        gcs_manager.upload_dataframe(
            day_ahead_df, day_ahead_prediction_path, index=False
        )
        log_message(
            f"Saved day ahead prediction to gs://dashboard_data_ge/{predictions_folder}",
            "info",
        )
        # Extract results_df (without the day-ahead row) and store full version for later
        results_df = results_df_with_regression[
            results_df_with_regression["date"] != latest_date
        ].copy()
        # Store the full version for results_dashboard creation later
        results_df_loaded = results_df_with_regression.copy()
    else:
        lasso_results, lasso_contrib, day_ahead_prediction = (
            rolling_lasso_regression_with_contributions(
                data,
                features_list,
                transformations_dict,
                window_days=60,
                predict_next_day=True,
            )
        )
        elastic_results, elastic_contrib = (
            rolling_elastic_regression_with_contributions(
                data, features_list, transformations_dict, window_days=60
            )
        )
        results_df_loaded = None
        # NOTE: Let us save the day ahead prediction to a csv file
        day_ahead_prediction_path = f"{predictions_folder}/day_ahead_prediction.csv"
        day_ahead_df = pd.DataFrame(
            {
                "Date": [datetime.today().date()],
                "Price": [day_ahead_prediction["predicted_original"]],
            }
        )
        gcs_manager.upload_dataframe(
            day_ahead_df, day_ahead_prediction_path, index=False
        )
        log_message(
            f"Saved day ahead prediction to gs://dashboard_data_ge/{predictions_folder}",
            "info",
        )
    if not gcs_manager.file_exists(eua_results_path):
        log_message(f"Day ahead prediction LASSO: {day_ahead_prediction}", "info")
        results_df = (
            results_df.set_index("date")
            .join(
                [
                    lasso_results[
                        ["predicted_original", "ci_lower_original", "ci_upper_original"]
                    ].add_prefix("lasso_"),
                    elastic_results[
                        ["predicted_original", "ci_lower_original", "ci_upper_original"]
                    ].add_prefix("elastic_"),
                ]
            )
            .reset_index()
        )
        gcs_manager.upload_dataframe(results_df, eua_results_path, index=False)
        log_message(
            f"Saved eua_results.csv to gs://dashboard_data_ge/{predictions_folder}",
            "info",
        )
        results_df_loaded = results_df.copy()
    else:
        log_message(f"Day ahead prediction LASSO: {day_ahead_prediction}", "info")

    latest_date = results_df["date"].max()
    next_working_day = pd.bdate_range(start=latest_date, periods=2)[1]

    new_row = pd.DataFrame({"date": [next_working_day]})
    results_dashboard = pd.concat([results_df, new_row], ignore_index=True)

    results_dashboard.loc[results_dashboard.index[-1], "lasso_predicted_original"] = (
        day_ahead_prediction["predicted_original"]
    )
    results_dashboard.loc[results_dashboard.index[-1], "lasso_ci_lower_original"] = (
        day_ahead_prediction["ci_lower_original"]
    )
    results_dashboard.loc[results_dashboard.index[-1], "lasso_ci_upper_original"] = (
        day_ahead_prediction["ci_upper_original"]
    )

    drop_cols = [
        "sarimax_predicted",
        "sarimax_ci_lower",
        "sarimax_ci_upper",
        "garch_predicted",
        "garch_ci_lower",
        "garch_ci_upper",
        "garch_volatility",
        "actual_original",
    ]
    results_dashboard = results_dashboard.drop(columns=drop_cols, errors="ignore")

    df = results_dashboard.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    df["abs_error"] = np.abs(df["actual"] - df["lasso_predicted_original"])
    df["pct_error"] = (
        (df["lasso_predicted_original"] - df["actual"]) / df["actual"]
    ) * 100
    df["squared_error"] = (df["actual"] - df["lasso_predicted_original"]) ** 2

    df["rolling_mae"] = df["abs_error"].rolling(window=30, min_periods=30).mean()
    df["rolling_mpe"] = df["pct_error"].rolling(window=30, min_periods=30).mean()
    df["rolling_mse"] = df["squared_error"].rolling(window=30, min_periods=30).mean()

    forecast_row = df.iloc[-1]
    yesterday_actual = df.iloc[-2]["actual"]
    position = (
        "Buy/Long"
        if forecast_row["lasso_predicted_original"] > yesterday_actual
        else "Sell/Short"
    )

    last_5_days = df.iloc[-6:].copy()
    forecast_table = pd.DataFrame(
        {
            "Date": [
                row["date"].strftime("%Y-%m-%d") for _, row in last_5_days.iterrows()
            ],
            "Actual Price": [
                f"{row['actual']:.2f}" if pd.notna(row["actual"]) else "N/A"
                for _, row in last_5_days.iterrows()
            ],
            "Forecast": [
                f"{row['lasso_predicted_original']:.2f}"
                for _, row in last_5_days.iterrows()
            ],
            "CI Lower": [
                f"{row['lasso_ci_lower_original']:.2f}"
                for _, row in last_5_days.iterrows()
            ],
            "CI Upper": [
                f"{row['lasso_ci_upper_original']:.2f}"
                for _, row in last_5_days.iterrows()
            ],
            "Position": [""] * 5 + [position],
        }
    )

    def color_position(val):
        if val == "Buy/Long":
            return "background-color: lightgreen"
        elif val == "Sell/Short":
            return "background-color: lightcoral"
        return ""

    styled_table = forecast_table.style.map(color_position, subset=["Position"])
    # display(styled_table)

    valid_data = df.dropna(subset=["rolling_mae", "rolling_mpe", "rolling_mse"])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=valid_data["date"],
            y=valid_data["rolling_mae"],
            name="MAE",
            line=dict(color="blue", width=2),
            hovertemplate="<b>MAE</b><br>Date: %{x}<br>Value: %{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=valid_data["date"],
            y=valid_data["rolling_mse"],
            name="MSE",
            line=dict(color="green", width=2),
            hovertemplate="<b>MSE</b><br>Date: %{x}<br>Value: %{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=valid_data["date"],
            y=valid_data["rolling_mpe"],
            name="MPE",
            line=dict(color="red", width=2),
            hovertemplate="<b>MPE</b><br>Date: %{x}<br>Value: %{y:.2f}%<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="MAE / MSE (Actual Units)", secondary_y=False)
    fig.update_yaxes(title_text="MPE (%)", secondary_y=True)

    fig.update_layout(
        title="Rolling 30-Day Error Metrics",
        width=1000,
        height=600,
        hovermode="x unified",
    )

    # fig.show()
    fig = plot_forecasts_interactive(results_dashboard, show_lr=False, last_days=90)
    # plot_forecasts_interactive(results_dashboard, show_lr=False, last_days=90).show()
    log_message(day_ahead_prediction["calculation_report"], "info")
    results_df = results_dashboard[
        results_dashboard["date"] != results_dashboard["date"].max()
    ]

    # Note: save_forecast_comparison_csv will be called after trading results are calculated

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    for period in [7, 15, 30]:
        results_dashboard[f"actual_high_{period}_mean"] = (
            results_dashboard["actual_high"].rolling(period).mean()
        )
        results_dashboard[f"actual_low_{period}_mean"] = (
            results_dashboard["actual_low"].rolling(period).mean()
        )
        results_dashboard[f"actual_high_{period}_max"] = (
            results_dashboard["actual_high"].rolling(period).max()
        )
        results_dashboard[f"actual_low_{period}_min"] = (
            results_dashboard["actual_low"].rolling(period).min()
        )

    new_cols = [
        "date",
        "actual_high_7_mean",
        "actual_high_15_mean",
        "actual_high_30_mean",
        "actual_low_7_mean",
        "actual_low_15_mean",
        "actual_low_30_mean",
        "actual_high_7_max",
        "actual_high_15_max",
        "actual_high_30_max",
        "actual_low_7_min",
        "actual_low_15_min",
        "actual_low_30_min",
    ]

    results_dashboard = results_dashboard[
        new_cols + [c for c in results_dashboard.columns if c not in new_cols]
    ]

    results_15day_max_min, _ = (
        calculate_directional_trading_returns_filtered_trading_cost(
            results_dashboard,
            transaction_cost=0.1,
            window_days=90,
            high_range_col="actual_high_15_max",
            low_range_col="actual_low_15_min",
            # start_date="2025-09-15",
        )
    )
    results_15day_mean, _ = calculate_directional_trading_returns_filtered_trading_cost(
        results_dashboard,
        transaction_cost=0.1,
        window_days=90,
        high_range_col="actual_high_15_mean",
        low_range_col="actual_low_15_mean",
        # start_date="2025-09-15",
    )

    # Save forecast comparison CSV using the trading results
    save_forecast_comparison_csv(
        results_dashboard,
        mean_trading_results=results_15day_mean,
        maxmin_trading_results=results_15day_max_min,
        predictions_folder=predictions_folder,
        filename="forecast_comparison.csv",
    )

    calculate_directional_accuracy(
        results_dashboard, window_days=90, start_date="2025-09-15"
    )
