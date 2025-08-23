import pandas as pd
import logging
import os
from datetime import datetime, timedelta
from typing import List
import requests
from flask import Flask

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

def get_token(
    clientId: str = "UA72RJH8YQzjaK0CI7x8yIZLpFdyQked",
    clientSecret: str = "l46OXzSm95VmWrEGuy-htpZLeOlU5f-jLEQhbMtmDgD-r_SydBXtJdZDVCSIlp86",
    baseUrlApi: str = "https://curves.veyt.com",
) -> str:
    """Get BEAVER token for https://curves.veyt.com."""
    url = f"{baseUrlApi}/api/v1/user/token/?client_id={clientId}&client_secret={clientSecret}"
    logging.info("Retrieving bearer token for https://curves.veyt.com API...")
    response = requests.post(url)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        logging.error("Authentication failed...")
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
    logging.info(f"Retrieving data for {name}...")
    response = requests.get(endpoint, headers=headers, params=data)
    if response.status_code == 200:
        logging.info("Formatting output to DataFrame...")
        return pd.DataFrame(response.json()["series"])
    else:
        logging.error(f"Request failed for {name} with status code {response.status_code}")
        return None

# ... (Add all your other get_* data functions here, making sure to pass the bearerToken)
# Example modification for get_options:
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
    logging.info(f"Retrieving data for {name}...")
    response = requests.get(endpoint, headers=headers, params=data)
    if response.status_code == 200:
        logging.info("Formatting output to DataFrame...")
        return pd.DataFrame(response.json()["series"])
    else:
        logging.error(f"Request failed for {name} with status code {response.status_code}")
        return None

def getTimeseriesPrice(
    name: str,
    bearerToken: str,
    fromDate: str = "2021-01-01",
    untilDate: str = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d"),
) -> pd.DataFrame:
    """Get timeseries price from https://curves.veyt.com."""
    endpoint = "https://curves.veyt.com/api/v1/timeseries/"
    headers = {"Authorization": f"Bearer {bearerToken}"}
    data = {"name": name, "data_from": fromDate, "data_to": untilDate}
    logging.info(f"Retrieving data for {name}...")
    response = requests.get(endpoint, headers=headers, params=data)
    if response.status_code == 200:
        logging.info("Formatting output to DataFrame...")
        return pd.DataFrame(response.json()["series"])
    else:
        logging.error(f"Request failed for {name} with status code {response.status_code}")
        return None

def getTsgroupPrice(
    name: str,
    bearerToken: str,
    fromDate: str = "2021-01-01",
    untilDate: str = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d"),
) -> pd.DataFrame:
    """Get tsgroup price from https://curves.veyt.com."""
    endpoint = "https://curves.veyt.com/api/v1/tsgroup/"
    headers = {"Authorization": f"Bearer {bearerToken}"}
    data = {"name": name, "data_from": fromDate, "data_to": untilDate}
    logging.info(f"Retrieving data for {name}...")
    response = requests.get(endpoint, headers=headers, params=data)
    if response.status_code == 200:
        logging.info("Formatting output to DataFrame...")
        return response.json()["series"]
    else:
        logging.error(f"Request failed for {name} with status code {response.status_code}")
        return None


# --- Your data processing functions (get_first_available, etc.) go here ---
# These functions do not need modification
def get_first_available(row_group):
    front_month_row = row_group[
        row_group["maturity_date"] == row_group.iloc[0]["front_month"]
    ]
    if not front_month_row.empty:
        return front_month_row.iloc[0]
    else:
        return row_group.iloc[0]

def get_first_available2(row_group):
    front_month_row = (
        row_group.dropna(subset=["open", "high", "low"]).iloc[0]
        if not row_group.dropna(subset=["open", "high", "low"]).empty
        else row_group.iloc[0]
    )
    return front_month_row

def get_label_market_price(
    label: str, marketprice_name: str, maturity_type: str, bearerToken: str, vdate=None
) -> pd.DataFrame:
    df1 = get_market_price(marketprice_name, bearerToken)
    if df1 is None: 
        return None
    df1["trade_date"] = pd.to_datetime(df1["trade_date"])
    df1["maturity_date"] = pd.to_datetime(df1["maturity_date"])
    if vdate:
        df1["vintage_date"] = pd.to_datetime(df1["vintage_date"])

    if maturity_type == "FM":
        df_filtered = (
            df1.groupby("trade_date").apply(get_first_available2).reset_index(drop=True)
        )
    elif maturity_type == "FY":
        if vdate:
            vdate_year = pd.to_datetime(vdate).year
            df_filtered = df1[
                (df1["maturity_date"].dt.month == 12)
                & (df1["maturity_date"].dt.year == vdate_year)
            ]
            if df_filtered.empty:
                df_filtered = df1[
                    (df1["maturity_date"].dt.year == vdate_year + 1)
                    & (df1["maturity_date"].dt.month == 1)
                    & (df1["maturity_date"].dt.day == 1)
                ]
        else:
            df_filtered = df1[
                (df1["maturity_date"].dt.month == 12)
                & (df1["maturity_date"].dt.year == df1["trade_date"].dt.year)
            ]
            if df_filtered.empty:
                df_filtered = df1[
                    (df1["maturity_date"].dt.year == df1["trade_date"].dt.year + 1)
                    & (df1["maturity_date"].dt.month == 1)
                    & (df1["maturity_date"].dt.day == 1)
                ]
    else:
        df_filtered = pd.DataFrame() # Handle cases where maturity_type is not FM or FY

    if vdate:
        df_filtered = df_filtered[df_filtered["vintage_date"] == pd.to_datetime(vdate)]
    if vdate:
        df_selected = df_filtered[
            ["trade_date", "settlement", "open_interest", "volume"]
        ].copy()
        df_selected.rename(
            columns={
                "trade_date": "Date",
                "settlement": f"{label}_settl",
                "open_interest": f"{label}_open_int",
                "volume": f"{label}_volume",
            },
            inplace=True,
        )
    else:
        df_selected = df_filtered[
            [
                "trade_date",
                "open",
                "low",
                "high",
                "settlement",
                "open_interest",
                "volume",
            ]
        ].copy()
        df_selected.rename(
            columns={
                "trade_date": "Date",
                "settlement": f"{label}_settl",
                "open": f"{label}_open",
                "low": f"{label}_low",
                "high": f"{label}_high",
                "open_interest": f"{label}_open_int",
                "volume": f"{label}_volume",
            },
            inplace=True,
        )
    return df_selected

# --- Add all your other data processing functions here, passing the bearerToken ---
# Example for get_timeserie_recent
def get_timeserie_recent(label: str, timeserie_name: str, bearerToken: str) -> pd.DataFrame:
    df = getTimeseriesPrice(timeserie_name, bearerToken)
    if df is None: 
        return None
    df["times"] = pd.to_datetime(df["times"], unit="ms")
    df = df[["times", "values"]].copy()
    df.rename(columns={"times": "Date", "values": label}, inplace=True)
    return df

# ... and so on for all your functions like get_cot, get_auction_uka, etc.
# Remember to pass the `bearerToken` down to any function that calls an API.

def retrieve_all_data(bearerToken: str) -> List[pd.DataFrame]:
    """Retrieve all datasets from the API."""
    # Pass bearerToken to all data fetching functions
    eua = get_label_market_price(label="EUA", marketprice_name=FUTURES["eua"], maturity_type="FY", bearerToken=bearerToken)
    uka = get_label_market_price(label="UKA", marketprice_name=FUTURES["uka"], maturity_type="FY", bearerToken=bearerToken)
    # ... and so on for every function call inside here
    
    # Example for fuelsw
    fuelsw = get_timeserie_recent("FuelSwitch", timeserie_name=SPREADS["fuel_switch"], bearerToken=bearerToken)

    # This is just a placeholder, you need to update ALL calls
    dataframes = [eua, uka, fuelsw] # Add all your dataframes here
    
    # Filter out None values in case of API failures
    dataframes = [df for df in dataframes if df is not None]

    return dataframes

def merge_timeseries(base_df, timeseries_dfs):
    """Merge each time series DataFrame into the base DataFrame"""
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
    # For writing to GCS, pandas can handle it directly if gcsfs is installed
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

# ... (include your filter_options and update_option_dataset functions)
def filter_options(mode: str, bearerToken: str) -> pd.DataFrame:
    if mode == "put":
        df = get_options(OPTIONS["put"], bearerToken)
    elif mode == "call":
        df = get_options(OPTIONS["call"], bearerToken)
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
    if new_data is None:
        logging.warning(f"No new data to update for {file_path}")
        return
        
    existing_dataset = pd.read_csv(file_path)
    existing_dataset["trade_date"] = pd.to_datetime(existing_dataset["trade_date"])
    new_data["trade_date"] = pd.to_datetime(new_data["trade_date"])

    updated_dataset = pd.concat(
        [existing_dataset.set_index("trade_date"), new_data.set_index("trade_date")]
    ).reset_index()
    updated_dataset = updated_dataset.drop_duplicates(
        subset=["trade_date", "maturity_date", "strike_price"], keep="last" # Use more keys to identify unique rows
    ).reset_index(drop=True)
    
    updated_dataset.to_csv(file_path, index=False)
    logging.info(f"The option dataset has been updated in {file_path}")


@app.route("/", methods=["POST"])
def main():
    """Main function triggered by HTTP request."""
    # It's better to get a fresh token on each run
    TOKEN = get_token()
    if not TOKEN:
        return "Failed to get token", 500

    # Update the main dataset
    dataframes = retrieve_all_data(TOKEN)
    if not dataframes:
        logging.warning("No dataframes were retrieved. Halting update.")
        return "No data retrieved", 200

    base_df = create_base_df()
    new_dataset = merge_timeseries(base_df, dataframes)
    logging.info("The new dataset has been created.")
    update_dataset(new_dataset, f"{gcloud_filepath}/dataset.csv")

    # Update the options datasets
    new_call = filter_options("call", TOKEN)
    new_put = filter_options("put", TOKEN)
    update_option_dataset(new_call, f"{gcloud_filepath}/options/call_options.csv")
    update_option_dataset(new_put, f"{gcloud_filepath}/options/put_options.csv")
    
    logging.info("The datasets have been updated successfully.")
    return "Update successful", 200

if __name__ == "__main__":
    # This is used for local testing
    # Cloud Run will use a production WSGI server like Gunicorn
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
