import pandas as pd
import logging
import os
from datetime import datetime, timedelta
from typing import List
import requests
from flask import Flask
from pandas.errors import EmptyDataError

# --- Your Constants (assuming they are in a constants.py file) ---
# You would have a constants.py file with these dictionaries
FUTURES = {
    "eua": "cc_fut_ice-endex_eua_c_ice_d_eur/t",
    "uka": "cc_fut_ice-endex_uka_c_ice_d_eur/t",
    "rggi": "cc_fut_ice-endex_rggi_c_ice_d_usd/t",
    "wci": "cc_fut_ice-endex_cca_c_ice_d_usd/t",
}
MARKET = {
    "ttf": "ng_fut_ice-endex_ttf_c_ice_d_eur/mwh",
    "coal": "coal_fut_ice-endex_api2_c_ice_d_usd/t",
    "germanpw_fm": "pwr_fut_eex_german-power_c_eex_d_eur/mwh",
    "germanpw_fy": "pwr_fut_eex_german-power_y_eex_d_eur/mwh",
    "brent": "oil_fut_ice-endex_brent_c_ice_d_usd/bbl",
}
SPREADS = {
    "fuel_switch": "cc_spread-clean-dark-fm_a_veyt_quant-cspread_ice-eua-front-dec_medium-efficiency_d_d_eur/mwh",
    "clean_dark": "cc_spread-clean-dark-spread_a_veyt_quant-cspread_ice-eua-front-dec_high-efficiency_d_d_eur/mwh",
    "clean_spark": "cc_spread-clean-spark-spread_a_veyt_quant-cspread_ice-eua-front-dec_high-efficiency_d_d_eur/mwh",
}
AUCTIONS_EUA = {
    "EU": "cc_ar_eex_eu_eua-auction_eex_3w_eur/t",
    "DE": "cc_ar_eex_de_eua-auction_eex_w_eur/t",
    "PL": "cc_ar_eex_pl_eua-auction_eex_w2_eur/t",
}
AUCTIONS_UKA = {
    "vol": "cc_ar_ice-endex_uka-auction-volume_ice_m_t",
    "ar": "cc_ar_ice-endex_uka-auction_ice_w_gbp/t",
}
AUCTIONS_RGGI = {
    "compl": "cc_ar_rggi_compliance-entities-demand_rggi_q_t",
    "non_compl": "cc_ar_rggi_non-compliance-entities-demand_rggi_q_t",
    "ar": "cc_ar_rggi_rggi-auction_rggi_q_usd/t",
}
AUCTIONS_WCI = {
    "compl": "cc_ar_wci_compliance-entities-demand_wci_q_t",
    "pcr": "cc_ar_wci_pcr-median-price_wci_q_usd/t",
    "ar": "cc_ar_wci_wci-auction_wci_q_usd/t",
}
COT = {
    "IF": ["cc_pos_ice-endex_eua-future_if_ice_w_t", "cc_pos_ice-endex_eua-future_if-traders_ice_w_#"],
    "IFCI": ["cc_pos_ice-endex_eua-future_ifci_ice_w_t", "cc_pos_ice-endex_eua-future_ifci-traders_ice_w_#"],
    "CU": ["cc_pos_ice-endex_eua-future_cu_ice_w_t", "cc_pos_ice-endex_eua-future_cu-traders_ice_w_#"],
    "OWCO": ["cc_pos_ice-endex_eua-future_owco_ice_w_t", "cc_pos_ice-endex_eua-future_owco-traders_ice_w_#"],
}
VOLATILITY = {
    "5d": "cc_vol_ice-endex_eua-future-vol-5d_ice_d_%",
    "20d": "cc_vol_ice-endex_eua-future-vol-20d_ice_d_%",
}
EUA_UKA_SPREAD = "cc_spread_ice-endex_eua-uka-front-dec_c_ice_d_eur/t"
OPTIONS = {
    "put": "cc_opt-put_ice-endex_eua-future-opt_mc_ice_d_eur/t",
    "call": "cc_opt-call_ice-endex_eua-future-opt_mc_ice_d_eur/t",
}
# --- End Constants ---


app = Flask(__name__)

gcloud_filepath = "gs://dashboard_data_ge"
NUMBER_DAYS = 3

logging.basicConfig(level=logging.INFO)

def get_token(
    clientId: str = "UA72RJH8YQzjaK0CI7x8yIZLpFdyQked",
    clientSecret: str = "l46OXzSm95VmWrEGuy-htpZLeOlU5f-jLEQhbMtmDgD-r_SydBXtJdZDVCSIlp86",
    baseUrlApi: str = "https://curves.veyt.com",
) -> str:
    """Get BEAVER token for https://curves.veyt.com."""
    url = f"{baseUrlApi}/api/v1/user/token/?client_id={clientId}&client_secret={clientSecret}"
    logging.info("Retrieving bearer token for https://curves.veyt.com API...")
    try:
        response = requests.post(url, timeout=30)
        response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
        return response.json()["access_token"]
    except requests.exceptions.RequestException as e:
        logging.error(f"Authentication failed: {e}")
        return None

def get_market_price(
    name: str,
    bearerToken: str,
    fromTradeDate: str = "2021-01-01",
    untilTradeDate: str = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d"),
) -> pd.DataFrame:
    """Get market price from for specific good from https://curves.veyt.com."""
    endpoint = "https://curves.veyt.com/api/v1/marketprice/"
    headers = {"Authorization": f"Bearer {bearerToken}"}
    data = {"name": name, "trade_from": fromTradeDate, "trade_to": untilTradeDate}
    logging.info(f"Retrieving market price data for {name}...")
    try:
        response = requests.get(endpoint, headers=headers, params=data, timeout=60)
        response.raise_for_status()
        logging.info(f"Successfully retrieved data for {name}.")
        return pd.DataFrame(response.json()["series"])
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed for {name}: {e}")
        return None
    except (KeyError, ValueError) as e:
        logging.error(f"Failed to parse JSON response for {name}: {e}")
        return None


def get_options(
    name: str,
    bearerToken: str,
    fromTradeDate: str = "2021-01-01",
    untilTradeDate: str = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d"),
) -> pd.DataFrame:
    """Get options data from https://curves.veyt.com."""
    endpoint = "https://curves.veyt.com/api/v1/option/"
    headers = {"Authorization": f"Bearer {bearerToken}"}
    data = {"name": name, "trade_from": fromTradeDate, "trade_to": untilTradeDate}
    logging.info(f"Retrieving options data for {name} from {fromTradeDate}...")
    try:
        response = requests.get(endpoint, headers=headers, params=data, timeout=60)
        response.raise_for_status()
        logging.info(f"Successfully retrieved data for {name}.")
        return pd.DataFrame(response.json()["series"])
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed for {name}: {e}")
        return None
    except (KeyError, ValueError) as e:
        logging.error(f"Failed to parse JSON response for {name}: {e}")
        return None

# --- (Add similar try/except blocks to all your other API fetching functions) ---
# getTimeseriesPrice, getTsgroupPrice

# --- Your data processing functions (get_first_available, etc.) go here ---
# These functions do not need modification if the inputs are checked
def get_label_market_price(
    label: str, marketprice_name: str, maturity_type: str, bearerToken: str, vdate=None
) -> pd.DataFrame:
    df1 = get_market_price(marketprice_name, bearerToken)
    # CRITICAL: Check if the dataframe is None before processing
    if df1 is None or df1.empty:
        logging.warning(f"No data received for {marketprice_name}, skipping.")
        return None
    
    # Your existing processing logic...
    try:
        df1["trade_date"] = pd.to_datetime(df1["trade_date"])
        df1["maturity_date"] = pd.to_datetime(df1["maturity_date"])
        # ... rest of your logic
    except Exception as e:
        logging.error(f"Error processing dataframe for {label}: {e}")
        return None
    
    # Return df_selected at the end
    # This is a placeholder, you need to return your actual processed dataframe
    return df1 

# --- (Add similar checks for None in all your other get_* functions) ---

def retrieve_all_data(bearerToken: str) -> List[pd.DataFrame]:
    """Retrieve all datasets from the API."""
    eua = get_label_market_price(label="EUA", marketprice_name=FUTURES["eua"], maturity_type="FY", bearerToken=bearerToken)
    uka = get_label_market_price(label="UKA", marketprice_name=FUTURES["uka"], maturity_type="FY", bearerToken=bearerToken)
    # ... all your other data fetching calls
    
    dataframes = [
        eua,
        uka,
        # ... add all other dataframe variables here
    ]
    
    # IMPORTANT: Filter out any None values from failed API calls
    valid_dataframes = [df for df in dataframes if df is not None]
    
    if not valid_dataframes:
        logging.warning("No valid dataframes were retrieved. No data to merge.")
        return []

    return valid_dataframes

def merge_timeseries(base_df, timeseries_dfs):
    """Merge each time series DataFrame into the base DataFrame"""
    if not timeseries_dfs:
        return base_df
    for df in timeseries_dfs:
        base_df = pd.merge(base_df, df, on="Date", how="left")
    return base_df

def create_base_df() -> pd.DataFrame:
    """Create a base DataFrame with a date range."""
    start_date = (datetime.today() - timedelta(days=NUMBER_DAYS)).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")
    date_range = pd.date_range(start=start_date, end=end_date)
    base_df = pd.DataFrame(date_range, columns=["Date"])
    return base_df

def update_dataset(new_data: pd.DataFrame, file_path: str):
    """Update the existing dataset with new data in GCS."""
    try:
        existing_dataset = pd.read_csv(file_path)
        existing_dataset["Date"] = pd.to_datetime(existing_dataset["Date"])
        new_data["Date"] = pd.to_datetime(new_data["Date"])

        updated_dataset = pd.concat(
            [existing_dataset.set_index("Date"), new_data.set_index("Date")]
        ).reset_index()
        updated_dataset = (
            updated_dataset.drop_duplicates(subset="Date", keep="last")
            .sort_values(by="Date")
            .reset_index(drop=True)
        )

        updated_dataset.to_csv(file_path, index=False)
        logging.info(f"The dataset is updated and has been saved in {file_path}")
    except Exception as e:
        logging.error(f"Failed to update dataset at {file_path}: {e}")
        raise

def filter_options(mode: str, bearerToken: str) -> pd.DataFrame:
    """Fetches the last 3 days of options data to handle gaps like weekends."""
    # Set fromTradeDate to 3 days ago to account for weekends/holidays
    from_date = (datetime.today() - timedelta(days=3)).strftime("%Y-%m-%d")

    if mode == "put":
        df = get_options(OPTIONS["put"], bearerToken, fromTradeDate=from_date)
    elif mode == "call":
        df = get_options(OPTIONS["call"], bearerToken, fromTradeDate=from_date)
    else:
        return None
        
    if df is None: 
        return None

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["maturity_date"] = pd.to_datetime(df["maturity_date"])

    df_selected = df[
        [
            "trade_date",
            "maturity_date",
            "strike_price",
            "settlement",
            "volume",
            "open_interest",
            "option_volatility",
        ]
    ].copy()
    return df_selected

def update_option_dataset(new_data: pd.DataFrame, file_path: str):
    """Update the existing option dataset with new data."""
    if new_data is None or new_data.empty:
        logging.warning(f"No new data to update for {file_path}")
        return
    try:
        try:
            existing_dataset = pd.read_csv(file_path)
        except (FileNotFoundError, EmptyDataError):
            logging.warning(f"Could not read existing dataset at {file_path}. Creating a new one.")
            existing_dataset = pd.DataFrame()

        if not existing_dataset.empty:
            existing_dataset["trade_date"] = pd.to_datetime(existing_dataset["trade_date"])
        
        new_data["trade_date"] = pd.to_datetime(new_data["trade_date"])

        updated_dataset = pd.concat([existing_dataset, new_data])
        
        # Define unique keys for options data
        unique_keys = ["trade_date", "maturity_date", "strike_price"]
        updated_dataset = updated_dataset.drop_duplicates(
            subset=unique_keys, keep="last"
        ).reset_index(drop=True)
        
        updated_dataset.to_csv(file_path, index=False)
        logging.info(f"The option dataset has been updated in {file_path}")

    except Exception as e:
        logging.error(f"Failed to update option dataset at {file_path}: {e}")
        raise

@app.route("/", methods=["POST"])
def main_handler():
    """Main function triggered by HTTP request."""
    try:
        TOKEN = get_token()
        if not TOKEN:
            logging.error("Failed to get token, aborting update.")
            return "Failed to get token", 500

        # Update the main dataset
        dataframes = retrieve_all_data(TOKEN)
        if not dataframes:
            logging.warning("No dataframes were retrieved. Halting main dataset update.")
        else:
            base_df = create_base_df()
            new_dataset = merge_timeseries(base_df, dataframes)
            logging.info("The new dataset has been created.")
            update_dataset(new_dataset, f"{gcloud_filepath}/dataset.csv")

        # Update the options datasets
        logging.info("Updating options datasets...")
        new_call = filter_options("call", TOKEN)
        new_put = filter_options("put", TOKEN)
        update_option_dataset(new_call, f"{gcloud_filepath}/options/call_options.csv")
        update_option_dataset(new_put, f"{gcloud_filepath}/options/put_options.csv")
        
        logging.info("The datasets have been updated successfully.")
        return "Update successful", 200
    except Exception as e:
        # Log any unexpected exception during the process
        logging.critical(f"An unhandled error occurred: {e}", exc_info=True)
        # Return a 500 error to signal failure to Cloud Scheduler
        return "An internal error occurred", 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
