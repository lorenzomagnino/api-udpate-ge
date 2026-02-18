import logging

import pandas as pd
from datetime import datetime, timedelta
from typing import List

import requests
from google.cloud import storage

from constants import FUTURES, MARKET, SPREADS, COT, VOLATILITY, AUCTIONS_EUA, AUCTIONS_UKA, AUCTIONS_RGGI, AUCTIONS_WCI
from gcs_utils import gcs_manager
logging.basicConfig(level=logging.INFO)


def upload_file(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


gcloud_filepath = "gs://dashboard_data_ge"  # Set if uploading to google cloud server
NUMBER_DAYS = 3


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


def get_market_price(
    name: str,
    fromTradeDate: str = "2021-01-01",
    untilTradeDate: str = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d"),
    bearerToken: str = TOKEN,
) -> pd.DataFrame:
    """
    Get market price from for specific good from https://curves.veyt.com.
    """

    endpoint = "https://curves.veyt.com/api/v1/marketprice/"
    headers = {"Authorization": f"Bearer {bearerToken}"}
    data = {"name": name, "trade_from": fromTradeDate, "trade_to": untilTradeDate}

    logging.info(f"Retrieving data for {name}...")
    response = requests.get(endpoint, headers=headers, params=data)

    if response.status_code == 200:
        logging.info("Formatting output to DataFrame...")
        return pd.DataFrame(response.json()["series"])

    else:
        logging.info("Request failed...")
        logging.info(f"{response.status_code}")
        return None


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
        return pd.DataFrame(response.json()["series"])
    else:
        logging.info("Request failed...")
        logging.info(f"{response.status_code}")
        return None


def getTimeseriesPrice(
    name: str,
    fromDate: str = "2021-01-01",
    untilDate: str = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d"),
    bearerToken: str = TOKEN,
) -> pd.DataFrame:
    """
    Get market price from for specific good from https://curves.veyt.com.
    """

    endpoint = "https://curves.veyt.com/api/v1/timeseries/"
    headers = {"Authorization": f"Bearer {bearerToken}"}
    data = {"name": name, "data_from": fromDate, "data_to": untilDate}

    logging.info(f"Retrieving data for {name}...")
    response = requests.get(endpoint, headers=headers, params=data)

    if response.status_code == 200:
        logging.info("Formatting output to DataFrame...")
        return pd.DataFrame(response.json()["series"])

    else:
        logging.info("Request failed...")
        logging.info(f"{response.status_code}")
        return None


def getTsgroupPrice(
    name: str,
    fromDate: str = "2021-01-01",
    untilDate: str = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d"),
    bearerToken: str = TOKEN,
) -> pd.DataFrame:
    """
    Get market price from for specific good from https://curves.veyt.com.
    """

    endpoint = "https://curves.veyt.com/api/v1/tsgroup/"
    headers = {"Authorization": f"Bearer {bearerToken}"}
    data = {"name": name, "data_from": fromDate, "data_to": untilDate}

    logging.info(f"Retrieving data for {name}...")
    response = requests.get(endpoint, headers=headers, params=data)

    if response.status_code == 200:
        logging.info("Formatting output to DataFrame...")
        return response.json()["series"]
    else:
        logging.info("Request failed...")
        logging.info(f"{response.status_code}")
        return None


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
    label: str, marketprice_name: str, maturity_type: str, vdate=None
) -> pd.DataFrame:
    """
    Gets market price data of the last three days based on the maturity type.
    """
    df1 = get_market_price(marketprice_name)
    if df1 is None or df1.empty:
        logging.warning(f"No data retrieved for {label}: {marketprice_name}")
        return pd.DataFrame(columns=["Date"])
    df1["trade_date"] = pd.to_datetime(df1["trade_date"])
    df1["maturity_date"] = pd.to_datetime(df1["maturity_date"])
    if vdate:
        df1["vintage_date"] = pd.to_datetime(df1["vintage_date"])

    # * If we want to extract Front Month or Front Year data
    if maturity_type == "FM":
        # df1['front_month'] = (df1['trade_date'] + pd.DateOffset(months=1)).apply(lambda x: x.replace(day=1))
        # df_filtered = df1.groupby('trade_date').apply(get_first_available).reset_index(drop=True)
        # df_filtered = df_filtered.drop(columns=['front_month'])
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


def get_open_and_volume(
    label: str, marketprice_name: str, maturity_type: str
) -> pd.DataFrame:
    """
    Gets open interest and volume of the last three days based on the maturity type
    """
    df1 = get_market_price(marketprice_name)
    if df1 is None or df1.empty:
        logging.warning(
            f"No data retrieved for {label} open/volume: {marketprice_name}"
        )
        return pd.DataFrame(columns=["Date", f"{label}_volume", f"{label}_open_int"])
    df1["trade_date"] = pd.to_datetime(df1["trade_date"])
    df1["maturity_date"] = pd.to_datetime(df1["maturity_date"])
    if maturity_type == "FM":
        df1["front_month"] = (df1["trade_date"] + pd.DateOffset(months=1)).apply(
            lambda x: x.replace(day=1)
        )
        df_filtered = (
            df1.groupby("trade_date").apply(get_first_available).reset_index(drop=True)
        )
        df_filtered = df_filtered.drop(columns=["front_month"])
    elif maturity_type == "FY":
        df_filtered = df1[
            (df1["maturity_date"].dt.year == df1["trade_date"].dt.year + 1)
            & (df1["maturity_date"].dt.month == 1)
            & (df1["maturity_date"].dt.day == 1)
        ]
    df_selected = df_filtered[["trade_date", "volume", "open_interest"]].copy()
    df_selected.rename(
        columns={
            "trade_date": "Date",
            "volume": f"{label}_volume",
            "open_interest": f"{label}_open_int",
        },
        inplace=True,
    )
    return df_selected


def get_timeserie_recent(label: str, timeserie_name: str) -> pd.DataFrame:
    df = getTimeseriesPrice(timeserie_name)
    if df is None or df.empty:
        logging.warning(f"No data retrieved for timeseries {label}: {timeserie_name}")
        return pd.DataFrame(columns=["Date", label])
    df["times"] = pd.to_datetime(df["times"], unit="ms")
    df = df[["times", "values"]].copy()
    df.rename(columns={"times": "Date", "values": label}, inplace=True)

    return df


def get_cot(label: str, name: List[str]) -> pd.DataFrame:
    df = getTsgroupPrice(name[0])
    df1 = getTimeseriesPrice(name[1])
    if df is None or df1 is None:
        logging.warning(f"No data retrieved for CoT {label}")
        return pd.DataFrame(
            columns=[
                "Date",
                f"{label}_long",
                f"{label}_short",
                f"{label}_net",
                f"{label}_#persons",
            ]
        )
    # Convert string columns to numeric
    df["values"]["long_total"] = pd.to_numeric(
        df["values"]["long_total"], errors="coerce"
    )
    df["values"]["short_total"] = pd.to_numeric(
        df["values"]["short_total"], errors="coerce"
    )
    print(df["values"]["short_total"].shape)
    print(df["values"]["long_total"].shape)
    print(df1["values"].shape)
    min_length = min(
        len(df["times"]),
        len(df["values"]["long_total"]),
        len(df["values"]["short_total"]),
        len(df1["values"]),
    )

    df_cot = pd.DataFrame(
        {
            "Date": df["times"],
            f"{label}_long": df["values"]["long_total"][:min_length],
            f"{label}_short": df["values"]["short_total"][:min_length],
            f"{label}_net": df["values"]["long_total"]
            - df["values"]["short_total"][:min_length],
            f"{label}_#persons": df1["values"][:min_length],
        }
    )
    df_cot["Date"] = pd.to_datetime(df_cot["Date"], unit="ms")

    return df_cot


def get_auction_uka(label: str, dict: str) -> pd.DataFrame:
    df = getTimeseriesPrice(dict["vol"])
    if df is None or df.empty:
        logging.warning(f"No volume data retrieved for {label} auction")
        df = pd.DataFrame(columns=["Date", f"{label}_auction_volume"])
    else:
        df = df[["times", "values"]].copy()
        df = df.rename(columns={"times": "Date", "values": f"{label}_auction_volume"})
        df["Date"] = pd.to_datetime(df["Date"], unit="ms").dt.strftime("%Y-%m-%d")

    df2 = getTsgroupPrice(dict["ar"])
    if df2 is None or "times" not in df2 or "values" not in df2:
        logging.warning(f"No auction data retrieved for {label}")
        df_auction2 = pd.DataFrame(
            columns=[
                "Date",
                "UKA_Bid_vol_sub",
                "UKA_bidders",
                "UKA_clearing_price",
                "UKA_cover_ratio",
                "UKA_successful_bidders",
                "UKA_vol_offered",
            ]
        )
    else:
        df_auction2 = pd.DataFrame(
            {
                "Date": df2["times"],
                "UKA_Bid_vol_sub": df2["values"]["bid_vol_submitted"],
                "UKA_bidders": df2["values"]["bidders"],
                "UKA_clearing_price": df2["values"]["clearing_price"],
                "UKA_cover_ratio": df2["values"]["cover_ratio"],
                "UKA_successful_bidders": df2["values"]["successful_bidders"],
                "UKA_vol_offered": df2["values"]["vol_offered"],
            }
        )
        df_auction2["Date"] = pd.to_datetime(
            df_auction2["Date"], unit="ms"
        ).dt.strftime("%Y-%m-%d")

    df_merged = df.merge(df_auction2, on="Date", how="outer")
    return df_merged


def get_auction(label: str, name: str) -> pd.DataFrame:
    df = getTsgroupPrice(name)
    if df is None or "times" not in df or "values" not in df:
        logging.warning(f"No auction data retrieved for {label}: {name}")
        return pd.DataFrame(
            columns=[
                "Date",
                f"{label}_auction_Price",
                f"{label}_auction_Cover Ratio",
                f"{label}_auction_Bid_vol_sub",
                f"{label}_auction_Bid_vol_offered",
            ]
        )
    df_cot = pd.DataFrame(
        {
            "Date": df["times"],
            f"{label}_auction_Price": df["values"]["clearing_price"],
            f"{label}_auction_Cover Ratio": df["values"]["cover_ratio"],
            f"{label}_auction_Bid_vol_sub": df["values"]["bid_vol_submitted"],
            f"{label}_auction_Bid_vol_offered": df["values"]["vol_offered"],
        }
    )
    df_cot["Date"] = pd.to_datetime(df_cot["Date"], unit="ms")

    return df_cot


def get_auction_rggi(label: str, dict: dict) -> pd.DataFrame:
    df = getTimeseriesPrice(dict["compl"])
    if df is None or df.empty:
        logging.warning(f"No compliant bidders data retrieved for {label}")
        df_auction = pd.DataFrame(columns=["Date", "RGGI_compl_bidders"])
    else:
        df_auction = df[["times", "values"]].copy()
        df_auction = df_auction.rename(
            columns={"times": "Date", "values": "RGGI_compl_bidders"}
        )
        df_auction["Date"] = pd.to_datetime(df_auction["Date"], unit="ms").dt.strftime(
            "%Y-%m-%d"
        )

    df1 = getTimeseriesPrice(dict["non_compl"])
    if df1 is None or df1.empty:
        logging.warning(f"No non-compliant bidders data retrieved for {label}")
        df_auction1 = pd.DataFrame(columns=["Date", "RGGI_non_compl_bidders"])
    else:
        df_auction1 = df1[["times", "values"]].copy()
        df_auction1 = df_auction1.rename(
            columns={"times": "Date", "values": "RGGI_non_compl_bidders"}
        )
        df_auction1["Date"] = pd.to_datetime(
            df_auction1["Date"], unit="ms"
        ).dt.strftime("%Y-%m-%d")
    df2 = getTsgroupPrice(dict["ar"])
    if df2 is None or "times" not in df2 or "values" not in df2:
        logging.warning(f"No auction data retrieved for {label}")
        df_auction2 = pd.DataFrame(
            columns=[
                "Date",
                "RGGI_Bid_vol_sub",
                "RGGI_bidders",
                "RGGI_clearing_price",
                "RGGI_cover_ratio",
                "RGGI_successful_bidders",
                "RGGI_vol_offered",
            ]
        )
    else:
        df_auction2 = pd.DataFrame(
            {
                "Date": df2["times"],
                "RGGI_Bid_vol_sub": df2["values"]["bid_vol_submitted"],
                "RGGI_bidders": df2["values"]["bidders"],
                "RGGI_clearing_price": df2["values"]["clearing_price"],
                "RGGI_cover_ratio": df2["values"]["cover_ratio"],
                "RGGI_successful_bidders": df2["values"]["successful_bidders"],
                "RGGI_vol_offered": df2["values"]["vol_offered"],
            }
        )
        df_auction2["Date"] = pd.to_datetime(
            df_auction2["Date"], unit="ms"
        ).dt.strftime("%Y-%m-%d")
    df_merged = df_auction.merge(df_auction1, on="Date", how="outer").merge(
        df_auction2, on="Date", how="outer"
    )
    return df_merged


def get_auction_wci(label: str, dict: dict) -> pd.DataFrame:
    df = getTimeseriesPrice(dict["compl"])
    if df is None or df.empty:
        logging.warning(f"No compliant bidders data retrieved for {label}")
        df_auction = pd.DataFrame(columns=["Date", "WCI_%_compl_bidders"])
    else:
        df_auction = df[["times", "values"]].copy()
        df_auction = df_auction.rename(
            columns={"times": "Date", "values": "WCI_%_compl_bidders"}
        )
        df_auction["Date"] = pd.to_datetime(df_auction["Date"], unit="ms").dt.strftime(
            "%Y-%m-%d"
        )

    df1 = getTimeseriesPrice(dict["pcr"])
    if df1 is None or df1.empty:
        logging.warning(f"No PCR data retrieved for {label}")
        df_auction1 = pd.DataFrame(columns=["Date", "WCI_median_allowance_price"])
    else:
        df_auction1 = df1[["times", "values"]].copy()
        df_auction1 = df_auction1.rename(
            columns={"times": "Date", "values": "WCI_median_allowance_price"}
        )
        df_auction1["Date"] = pd.to_datetime(
            df_auction1["Date"], unit="ms"
        ).dt.strftime("%Y-%m-%d")
    df2 = getTsgroupPrice(dict["ar"])
    if df2 is None or "times" not in df2 or "values" not in df2:
        logging.warning(f"No auction data retrieved for {label}")
        df_auction2 = pd.DataFrame(
            columns=[
                "Date",
                "WCI_Bid_vol_sub",
                "WCI_clearing_price",
                "WCI_cover_ratio",
                "WCI_vol_offered",
            ]
        )
    else:
        df_auction2 = pd.DataFrame(
            {
                "Date": df2["times"],
                "WCI_Bid_vol_sub": df2["values"]["bid_vol_submitted"],
                "WCI_clearing_price": df2["values"]["clearing_price"],
                "WCI_cover_ratio": df2["values"]["cover_ratio"],
                "WCI_vol_offered": df2["values"]["vol_offered"],
            }
        )
        df_auction2["Date"] = pd.to_datetime(
            df_auction2["Date"], unit="ms"
        ).dt.strftime("%Y-%m-%d")
    df_merged = df_auction.merge(df_auction1, on="Date", how="outer").merge(
        df_auction2, on="Date", how="outer"
    )
    return df_merged


def get_recent_cot(
    file_path: str = f"{gcloud_filepath}/price_fcst_model_data/Net positions (ICE).csv",
):
    df_cot = pd.read_csv(file_path)
    df_cot["Date"] = pd.to_datetime(df_cot["Date"])
    last_df_cot = df_cot[
        df_cot["Date"] >= (datetime.today() - timedelta(days=2)).strftime("%Y-%m-%d")
    ]
    last_df_cot = last_df_cot.drop(df_cot.columns[1], axis=1)
    return last_df_cot


def get_volatility(marketprice_name: str) -> pd.DataFrame:
    df1 = get_market_price(marketprice_name)
    if df1 is None or df1.empty:
        logging.warning(f"No data retrieved for volatility: {marketprice_name}")
        return pd.DataFrame(columns=["trade_date", "settlement"])
    df1["trade_date"] = pd.to_datetime(df1["trade_date"])
    df1["maturity_date"] = pd.to_datetime(df1["maturity_date"])
    current_year = datetime.today().year
    df_filtered = df1[df1["maturity_date"].dt.year == current_year]
    if df_filtered.empty:
        # Fallback: use the most recent year available
        logging.warning(
            f"No data for current year {current_year}, using most recent available year"
        )
        available_years = df1["maturity_date"].dt.year.unique()
        if len(available_years) > 0:
            most_recent_year = max(available_years)
            df_filtered = df1[df1["maturity_date"].dt.year == most_recent_year]
            logging.info(f"Using data from year {most_recent_year}")
    df_selected = df_filtered[["trade_date", "settlement"]].copy()
    return df_selected


def get_spread(marketprice_name: str) -> pd.DataFrame:
    df1 = get_market_price(marketprice_name)
    if df1 is None or df1.empty:
        logging.warning(f"No data retrieved for spread: {marketprice_name}")
        return pd.DataFrame(columns=["Date", "EUA/UKA_spread"])
    df1["trade_date"] = pd.to_datetime(df1["trade_date"])
    df1["maturity_date"] = pd.to_datetime(df1["maturity_date"])
    current_year = datetime.today().year
    df_filtered = df1[
        (df1["maturity_date"].dt.year == current_year)
        & (df1["maturity_date"].dt.month == 12)
    ]
    # If no December data for current year (e.g., early in the year), try previous year's December
    if df_filtered.empty:
        logging.warning(
            f"No December {current_year} data available, trying previous year"
        )
        prev_year = current_year - 1
        df_filtered = df1[
            (df1["maturity_date"].dt.year == prev_year)
            & (df1["maturity_date"].dt.month == 12)
        ]
        if df_filtered.empty:
            # Fallback: use the most recent December contract available
            logging.warning(
                f"No December {prev_year} data, using most recent December contract"
            )
            december_contracts = df1[df1["maturity_date"].dt.month == 12]
            if not december_contracts.empty:
                most_recent_december_year = december_contracts[
                    "maturity_date"
                ].dt.year.max()
                df_filtered = df1[
                    (df1["maturity_date"].dt.year == most_recent_december_year)
                    & (df1["maturity_date"].dt.month == 12)
                ]
                logging.info(f"Using December {most_recent_december_year} data")
    df_selected = df_filtered[["trade_date", "settlement"]].copy()
    df_selected = df_selected.rename(
        columns={"trade_date": "Date", "settlement": "EUA/UKA_spread"}
    )
    return df_selected


EUA_UKA_SPREAD = "cc_price_a_uka-eua-spread_mc_veyt_quant-future-spread_d_d_eur/t"


if __name__ == "__main__":
    # * Futures
    # //eua = get_label_market_price(label='EUA', marketprice_name=FUTURES['eua'], maturity_type='FM')
    eua = get_label_market_price(
        label="EUA", marketprice_name=FUTURES["eua"], maturity_type="FY"
    )
    uka = get_label_market_price(
        label="UKA", marketprice_name=FUTURES["uka"], maturity_type="FY"
    )
    rggi = get_label_market_price(
        label="RGGI",
        marketprice_name=FUTURES["rggi"],
        maturity_type="FY",
        vdate=f"{datetime.today().year}-01-01",
    )
    wci = get_label_market_price(
        label="WCI",
        marketprice_name=FUTURES["wci"],
        maturity_type="FY",
        vdate=f"{datetime.today().year}-01-01",
    )
    # * Market Energy Prices
    ttf = get_label_market_price(
        label="TTF", marketprice_name=MARKET["ttf"], maturity_type="FM"
    )
    coal = get_label_market_price(
        label="Coal", marketprice_name=MARKET["coal"], maturity_type="FM"
    )  # Dollars
    germanpw_fm = get_label_market_price(
        label="GermanPower(FM)",
        marketprice_name=MARKET["germanpw_fm"],
        maturity_type="FM",
    )
    germanpw_fy = get_label_market_price(
        label="GermanPower(FY)",
        marketprice_name=MARKET["germanpw_fy"],
        maturity_type="FY",
    )
    brent = get_label_market_price(
        label="Brent", marketprice_name=MARKET["brent"], maturity_type="FM"
    )  # Dollars
    # * Market Spread
    eua_uka_spread = get_spread(EUA_UKA_SPREAD)
    # * Spread
    # #//df = getTimeseriesPrice('cc_spread-clean-dark-fm_a_veyt_quant-cspread_ice-eua-front-dec_medium-efficiency_d_d_eur/mwh')
    fuelsw = get_timeserie_recent("FuelSwitch", timeserie_name=SPREADS["fuel_switch"])
    clean_dark = get_timeserie_recent("CD", timeserie_name=SPREADS["clean_dark"])
    clean_spark = get_timeserie_recent("CS", timeserie_name=SPREADS["clean_spark"])
    # * Auctions
    # //df = getTimeseriesPrice(AUCTIONS_WCI['compl'])
    # //df = getTimeseriesPrice(AUCTIONS_RGGI['compl'])
    # df = getTimeseriesPrice(AUCTIONS_WCI['pcr'])
    # //df = getTimeseriesPrice(AUCTIONS_UKA)
    auction_uka = get_auction_uka(label="UKA", dict=AUCTIONS_UKA)
    auction_wci = get_auction_wci(label="WCI", dict=AUCTIONS_WCI)
    auction_rggi = get_auction_rggi(label="RGGI", dict=AUCTIONS_RGGI)
    auction_eu = get_auction(label="eu", name=AUCTIONS_EUA["EU"])
    auction_de = get_auction(label="de", name=AUCTIONS_EUA["DE"])
    auction_pl = get_auction(label="pl", name=AUCTIONS_EUA["PL"])
    # //df_Auction = get_last_auction(["cc_ar_eex_de_eua-auction_eex_w_eur/t", "cc_ar_eex_eu_eua-auction_eex_3w_eur/t", "cc_ar_eex_pl_eua-auction_eex_w2_eur/t"])
    # * Cot
    # //df = getTsgroupPrice(AUCTIONS_UKA['ar']),
    # // df1  = getTimeseriesPrice(COT['IF'][1])
    if_cot = get_cot(label="IF", name=COT["IF"])
    ifci_cot = get_cot(label="IFCI", name=COT["IFCI"])
    # //ofi_cot  = get_cot(label = 'OFI', name = COT['OFI']) #? Still some problems
    cu_cot = get_cot(label="CU", name=COT["CU"])
    owco_cot = get_cot(label="OWCO", name=COT["OWCO"])

    # * Volatility
    # //vol = get_market_price(VOLATILITY['20d'])
    vol_5d = get_volatility(VOLATILITY["5d"])
    vol_20d = get_volatility(VOLATILITY["20d"])

    dataframes = [
        eua,
        uka,
        rggi,
        wci,
        eua_uka_spread,
        ttf,
        coal,
        germanpw_fm,
        germanpw_fy,
        brent,
        fuelsw,
        clean_dark,
        clean_spark,
        auction_eu,
        auction_de,
        auction_pl,
        auction_uka,
        auction_wci,
        auction_rggi,
        if_cot,
        ifci_cot,
        cu_cot,
        owco_cot,
        vol_5d,
        vol_20d,
    ]
    dataframes[-1] = dataframes[-1].rename(
        columns={"trade_date": "Date", "settlement": "vol_20d"}
    )
    dataframes[-2] = dataframes[-2].rename(
        columns={"trade_date": "Date", "settlement": "vol_5d"}
    )

    # * Step 1: Create a date range from 2020-01-01 to today
    start_date = "2021-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")
    date_range = pd.date_range(start=start_date, end=end_date)

    # * Step 2: Initialize a DataFrame with this date range
    base_df = pd.DataFrame(date_range, columns=["Date"])

    # * Function to merge time series dataframes into the base dataframe
    def merge_timeseries(base_df, timeseries_dfs):
        for df in timeseries_dfs:
            df["Date"] = pd.to_datetime(df["Date"])
            base_df = pd.merge(base_df, df, on="Date", how="left")
        return base_df

    # Step 3: Merge each time series DataFrame into the base DataFrame
    merged_df = merge_timeseries(base_df, dataframes)
    date = "2022-01-01"
    dataset = merged_df[merged_df["Date"] >= date]
    # Display the merged DataFrame
    # display(dataset)
    # * Step 4: Save the merged DataFrame to a CSV file using GCS Manager
    local_filepath = "/Users/lorenzomagnino/Desktop/dataset.csv"
    gcloud_filepath = (
        "gs://dashboard_data_ge"  # Set if uploading to google cloud server
    )
    gcs_path = "dataset.csv"

    # Upload to Google Cloud Storage using gcs_manager
    success = gcs_manager.upload_dataframe(dataset, gcs_path, index=False)
    if success:
        print(f"Successfully uploaded dataset to gs://dashboard_data_ge/{gcs_path}")
    else:
        print("Failed to upload dataset to GCS")
