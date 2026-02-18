# Total lines: 108

import logging
from datetime import datetime, timedelta

import pandas as pd
import requests

from gcs_utils import gcs_manager

gcloud_filepath = "gs://dashboard_data_ge"  # Set if uploading to google cloud server


def get_token(
    clientId: str = "UA72RJH8YQzjaK0CI7x8yIZLpFdyQked",
    clientSecret: str = "l46OXzSm95VmWrEGuy-htpZLeOlU5f-jLEQhbMtmDgD-r_SydBXtJdZDVCSIlp86",
    baseUrlApi: str = "https://curves.veyt.com",
) -> str:
    """
    Get BEAVER token for https://curves.veyt.com.
    """

    url = f"{baseUrlApi}/api/v1/user/token/?client_id={clientId}&client_secret={clientSecret}"

    logging.info("Retrieving bearer token for https://curves.veyt.com API...")
    response = requests.post(url)

    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        logging.info("Authentication failed...")
        return None


TOKEN = get_token()


def get_options(
    name: str,
    fromTradeDate: str = "2021-01-01",
    untilTradeDate: str = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d"),
    bearerToken: str = TOKEN,
) -> pd.DataFrame:
    """
    Get market price from for specific good from https://curves.veyt.com.
    """

    endpoint = "https://curves.veyt.com/api/v1/option/"
    headers = {"Authorization": f"Bearer {bearerToken}"}
    data = {"name": name, "trade_from": fromTradeDate, "trade_to": untilTradeDate}
    logging.info(f"Retrieving data for {name}...")
    response = requests.get(endpoint, headers=headers, params=data)

    if response.status_code == 200:
        logging.info("Formatting output to DataFrame...")
        df = pd.DataFrame(response.json()["series"])
        if df.empty:
            logging.warning(f"Empty DataFrame returned for {name}")
        return df
    else:
        logging.warning(f"Request failed for {name}: {response.status_code}")
        return None


def get_specific_option(
    mode: str, expiration_month: int, expiration_year: int
) -> pd.DataFrame:
    if mode == "put":
        df = get_options(OPTIONS["put"])
    elif mode == "call":
        df = get_options(OPTIONS["call"])
    else:
        logging.warning(f"Invalid mode: {mode}. Must be 'put' or 'call'")
        return pd.DataFrame(
            columns=[
                "Date",
                f"{mode}_settl",
                f"{mode}_open_int",
                f"{mode}_volume",
                f"{mode}",
            ]
        )

    if df is None or df.empty:
        logging.warning(f"No data retrieved for {mode} options")
        return pd.DataFrame(
            columns=[
                "Date",
                f"{mode}_settl",
                f"{mode}_open_int",
                f"{mode}_volume",
                f"{mode}",
            ]
        )

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["maturity_date"] = pd.to_datetime(df["maturity_date"])

    # Filter by both month and year for accuracy
    df_filtered = df[
        (df["maturity_date"].dt.month == expiration_month)
        & (df["maturity_date"].dt.year == expiration_year)
    ]

    if df_filtered.empty:
        logging.warning(
            f"No data found for {mode} options expiring {expiration_month}/{expiration_year}"
        )
        # Fallback: try just by month if no year match
        df_filtered = df[df["maturity_date"].dt.month == expiration_month]
        if df_filtered.empty:
            logging.warning(
                f"No data found for {mode} options in month {expiration_month}"
            )
            return pd.DataFrame(
                columns=[
                    "Date",
                    f"{mode}_settl",
                    f"{mode}_open_int",
                    f"{mode}_volume",
                    f"{mode}",
                ]
            )

    df_selected = df_filtered[
        [
            "trade_date",
            "settlement",
            "volume",
            "open_interest",
            "option_volatility",
        ]
    ].copy()
    df_selected.rename(
        columns={
            "trade_date": "Date",
            "settlement": f"{mode}_settl",
            "open_interest": f"{mode}_open_int",
            "volume": f"{mode}_volume",
            "option_volatility": f"{mode}",
        },
        inplace=True,
    )
    return df_selected


def filter_options(mode: str) -> pd.DataFrame:
    if mode == "put":
        df = get_options(OPTIONS["put"])
    elif mode == "call":
        df = get_options(OPTIONS["call"])
    else:
        logging.warning(f"Invalid mode: {mode}. Must be 'put' or 'call'")
        return pd.DataFrame(
            columns=[
                "trade_date",
                "maturity_date",
                "strike_price",
                "settlement",
                "volume",
                "open_interest",
                "option_volatility",
                "delta_factor",
            ]
        )

    if df is None or df.empty:
        logging.warning(f"No data retrieved for {mode} options")
        return pd.DataFrame(
            columns=[
                "trade_date",
                "maturity_date",
                "strike_price",
                "settlement",
                "volume",
                "open_interest",
                "option_volatility",
                "delta_factor",
            ]
        )

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["maturity_date"] = pd.to_datetime(df["maturity_date"])

    # Check if required columns exist
    required_columns = [
        "trade_date",
        "maturity_date",
        "strike_price",
        "settlement",
        "volume",
        "open_interest",
        "option_volatility",
        "delta_factor",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.warning(
            f"Missing columns in {mode} options data: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )
        # Only select columns that exist
        available_columns = [col for col in required_columns if col in df.columns]
        df_selected = df[available_columns].copy()
    else:
        df_selected = df[required_columns].copy()

    return df_selected


OPTIONS = {
    "put": "cc_opt-put_ice-endex_eua-future-opt_mc_ice_d_eur/t",
    "call": "cc_opt-call_ice-endex_eua-future-opt_mc_ice_d_eur/t",
}


if __name__ == "__main__":
    # //put = get_options(OPTIONS['put'])
    # * Creating the datasets
    call = filter_options("call")
    put = filter_options("put")

    # * Save the updated dataset to Google Cloud Storage using GCS Manager
    gcloud_path_options = "gs://dashboard_data_ge/options"
    gcs_put_path = "options/put_options.csv"
    gcs_call_path = "options/call_options.csv"

    # Upload put options to GCS
    if put is not None and not put.empty:
        success_put = gcs_manager.upload_dataframe(put, gcs_put_path, index=False)
        if success_put:
            print(
                f"Successfully uploaded put_options to gs://dashboard_data_ge/{gcs_put_path}"
            )
        else:
            print("Failed to upload put_options to GCS")
    else:
        print("No put options data to upload - DataFrame is empty or None")

    # Upload call options to GCS
    if call is not None and not call.empty:
        success_call = gcs_manager.upload_dataframe(call, gcs_call_path, index=False)
        if success_call:
            print(
                f"Successfully uploaded call_options to gs://dashboard_data_ge/{gcs_call_path}"
            )
        else:
            print("Failed to upload call_options to GCS")
    else:
        print("No call options data to upload - DataFrame is empty or None")

    logging.info("The dataset has been saved")
