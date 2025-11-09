import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
import feedparser
import dash_bootstrap_components as dbc
from dash import html
from google.cloud import storage
import yfinance as yf
import hashlib
import time

# gcloud_filepath = '/Users/joemiles/Code/Greenedge/greenedge_gcloud' #Set if running code locally
gcloud_filepath = "gs://dashboard_data_ge"  # Set if uploading to google cloud server

# News cache for smart updates
NEWS_CACHE = {}
NEWS_CACHE_TIMESTAMP = {}
NEWS_CACHE_DURATION = 300  # 5 minutes cache duration


def get_sp500():
    """
    Download monthly S&P 500 data from Yahoo Finance.

    Returns:
        pd.DataFrame: DataFrame with Date and Close columns for S&P 500 index
    """
    df = yf.download(
        "^GSPC", start="2023-12-01", end="2025-02-02", interval="1mo", progress=False
    )
    df.columns = ["_".join(str(s) for s in col).strip("_") for col in df.columns]
    df.rename(columns={"Close_^GSPC": "Close"}, inplace=True)
    df.reset_index(inplace=True)
    return df


def get_commodities():
    """
    Load commodities data from CSV file.

    Returns:
        pd.DataFrame: DataFrame with Date and Bloomberg Commodity columns
    """
    df = pd.read_csv(f"{gcloud_filepath}/chart.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%Y")
    return df


def get_euro600():
    """
    Load and process Euro Stoxx 600 data.

    Returns:
        pd.DataFrame: DataFrame with Date (MM/YYYY) and Euro600 columns, filtered to 2024-2025
    """
    df = pd.read_csv(f"{gcloud_filepath}/performances/Euro stoxx 600.csv")
    df = df.drop(df.index[0])

    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    df = df[["Date", "Close"]].rename(columns={"Close": "Euro600"})

    df = df.set_index("Date")
    df = df.resample("M").last()

    df.index = df.index.to_period("M").to_timestamp()
    df = df.reset_index()
    df = df[df["Date"] <= "2025-01-01"]
    df = df[df["Date"] >= "2024-01-01"]

    df["Date"] = df["Date"].dt.strftime("%m/%Y")
    return df


def get_fst300():
    """
    Load and process FTSEurofirst 300 data.

    Returns:
        pd.DataFrame: DataFrame with Date (MM/YYYY) and FST300 columns, filtered to 2024-2025
    """
    df = pd.read_csv(
        f"{gcloud_filepath}/performances/FTSEurofirst 300 Historical Data.csv"
    )
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

    df = df[["Date", "Price"]].rename(columns={"Price": "FST300"})

    df = df.sort_values("Date")

    df = df[df["Date"] <= "2025-01-01"]
    df = df[df["Date"] >= "2024-01-01"]

    df["Date"] = df["Date"].dt.strftime("%m/%Y")
    df["FST300"] = df["FST300"].astype(str).str.replace(",", "")
    df["FST300"] = pd.to_numeric(df["FST300"], errors="coerce")
    return df


def get_jpm():
    """
    Load and process JPM Global Government Bond A data.

    Returns:
        pd.DataFrame: DataFrame with Date (MM/YYYY) and JPM columns, filtered to 2024-2025
    """
    df = pd.read_csv(f"{gcloud_filepath}/performances/JPM Global Government Bond A.csv")
    df = df.drop(df.index[0])

    df.rename(columns={"Unnamed: 0": "Date", "Unnamed: 1": "JPM"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)

    df.set_index("Date", inplace=True)

    df = df.resample("ME").last()

    df.reset_index(inplace=True)

    df["Date"] = df["Date"].dt.to_period("M").dt.start_time

    df = df[df["Date"] <= "2025-01-01"]
    df = df[df["Date"] >= "2024-01-01"]

    df["Date"] = df["Date"].dt.strftime("%m/%Y")
    return df


def get_gcc():
    """
    Load and process S&P Global Carbon Credit Index data.

    Returns:
        pd.DataFrame: DataFrame with Date (MM/YYYY) and S&P GCC columns, filtered to 2024-2025
    """
    df = pd.read_csv(f"{gcloud_filepath}/performances/PerformanceGraphExport.csv")
    df.rename(
        columns={
            "Effective date ": "Date",
            "S&P Global Carbon Credit Index (USD) ER": "S&P GCC",
        },
        inplace=True,
    )
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df.set_index("Date", inplace=True)
    df = df.resample("ME").last()
    df.reset_index(inplace=True)
    df["Date"] = df["Date"].dt.to_period("M").dt.start_time
    df = df[df["Date"] <= "2025-01-01"]
    df = df[df["Date"] >= "2024-01-01"]
    df["Date"] = df["Date"].dt.strftime("%m/%Y")
    return df


def get_ftse():
    """
    Load and process FTSE World Government Bond data.

    Returns:
        pd.DataFrame: DataFrame with Date (MM/YYYY) and FTSE columns, filtered to 2024-2025
    """
    df = pd.read_csv(f"{gcloud_filepath}/performances/ FTSE World Government Bond.csv")
    df.rename(
        columns={"FTSE World Government Bond - Developed Markets": "FTSE"}, inplace=True
    )

    df["Date"] = pd.to_datetime(df["Date"])

    df = df[df["Date"] <= "2025-01-01"]
    df = df[df["Date"] >= "2024-01-01"]
    df["Date"] = df["Date"].dt.strftime("%m/%Y")
    df["FTSE"] = df["FTSE"].astype(str).str.replace(",", "")
    df["FTSE"] = pd.to_numeric(df["FTSE"], errors="coerce")
    return df


def get_second_thursday(year):
    """
    Calculate the second Thursday of December for a given year.

    Args:
        year (int): The year to calculate for

    Returns:
        datetime: The second Thursday of December
    """
    dec_first = datetime(year, 12, 1)
    first_thursday = dec_first + timedelta(days=(3 - dec_first.weekday()) % 7)
    second_thursday = first_thursday + timedelta(days=7)
    return second_thursday


def get_dec_year():
    """
    Get the current December contract year for EUA and UKA products.

    Returns:
        str: Two-digit year string (e.g., '25' for 2025)
    """
    today = datetime.today()
    current_year = today.year

    second_thursday = get_second_thursday(current_year)

    if today <= second_thursday:
        contract_year = f"{str(current_year)[-2:]}"
    else:
        contract_year = f"{str(current_year + 1)[-2:]}"
    return contract_year


def custom_key(date):
    """
    Convert date string from 'Dec25' format to datetime object.

    Args:
        date (str): Date string in 'Dec25' format

    Returns:
        datetime: Parsed datetime object
    """
    return datetime.strptime(date, "%b%y")


def get_options_vol(date, contract):
    """
    Get options volume data for a specific date and contract.

    Args:
        date (datetime): Date to get data for
        contract (str): Contract identifier (e.g., 'EUA25')

    Returns:
        tuple: (yesterday_data, week_ago_data) as DataFrames
    """
    client = storage.Client()
    bucket = client.bucket("dashboard_data_ge")

    previous_day = (date).strftime("%Y_%m_%d")
    previous_week = (date - BDay(5)).strftime("%Y_%m_%d")

    options_vol_yday = (
        f"/ice_csv_data/{contract}_option/{contract}_option_vol_{previous_day}.csv"
    )
    options_vol_yweek = (
        f"/ice_csv_data/{contract}_option/{contract}_option_vol_{previous_week}.csv"
    )
    options_docs_dict = {
        "df_options_yday": options_vol_yday,
        "df_options_yweek": options_vol_yweek,
    }

    dfs = {}

    for name, file in options_docs_dict.items():
        blob = bucket.blob(file[1:])

        if blob.exists():
            file_path = gcloud_filepath + file
            dfs[name] = pd.read_csv(file_path)
        else:
            dfs[name] = pd.DataFrame()

    return dfs["df_options_yday"], dfs["df_options_yweek"]


def convert_abbrev_to_full_date(short_date: str) -> str:
    """
    Convert a date string like 'Dec25' to '2025-12-01'.

    Args:
        short_date (str): Date string in 'Dec25' format

    Returns:
        str: Full date string in 'YYYY-MM-DD' format
    """
    parse_str = "01" + short_date
    date_obj = datetime.strptime(parse_str, "%d%b%y")
    return date_obj.strftime("%Y-%m-%d")


def get_specific_option(date, mode: str, maturity: str = None):
    """
    Get specific options data for a given date and mode.

    Args:
        date (datetime): Trade date to filter by
        mode (str): 'put' or 'call' options
        maturity (str, optional): Maturity date in 'Dec25' format

    Returns:
        pd.DataFrame: Filtered options data
    """
    if mode == "put":
        df = pd.read_csv(f"{gcloud_filepath}/options/put_options.csv")
    elif mode == "call":
        df = pd.read_csv(f"{gcloud_filepath}/options/call_options.csv")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["maturity_date"] = pd.to_datetime(df["maturity_date"])
    df_spec = df[df["trade_date"] == date]
    if maturity:
        maturity = convert_abbrev_to_full_date(maturity)
        df_filtered = df_spec[df_spec["maturity_date"] == maturity]
        df_selected = df_filtered[
            [
                "trade_date",
                "strike_price",
                "settlement",
                "volume",
                "open_interest",
                "option_volatility",
                "delta_factor",
            ]
        ].copy()
    else:
        df_filtered = df_spec
        df_selected = df_filtered[
            [
                "trade_date",
                "maturity_date",
                "strike_price",
                "settlement",
                "volume",
                "open_interest",
                "option_volatility",
                "delta_factor",
            ]
        ].copy()
    return df_selected


def get_options_table(date, month, contract):
    """
    Create options table with both put and call data for each strike price.

    Args:
        date (datetime): Date to get data for
        month (str): Month to filter by
        contract (str): Contract identifier

    Returns:
        pd.DataFrame: Options table with put/call data merged by strike
    """
    df_yday, df_yweek = get_options_vol(date, contract)
    df_yday = df_yday[df_yday["Month"] == month]

    df_yday_puts = df_yday[df_yday["P/C"] == "P"]
    df_yday_puts = df_yday_puts.rename(
        columns={"Price": "Price Puts", "Delta": "Delta Puts", "OI": "OI Puts"}
    )
    df_yday_calls = df_yday[df_yday["P/C"] == "C"]
    df_yday_calls = df_yday_calls.rename(
        columns={"Price": "Price Calls", "Delta": "Delta Calls", "OI": "OI Calls"}
    )
    df_yday_calls = df_yday_calls[["Strike", "Price Calls", "Delta Calls", "OI Calls"]]
    df_yday_puts = df_yday_puts[["Strike", "Price Puts", "Delta Puts", "OI Puts"]]

    df_data = df_yday_calls.merge(df_yday_puts, how="inner", on="Strike")
    df_data = df_data.reindex(
        columns=[
            "OI Calls",
            "Delta Calls",
            "Price Calls",
            "Strike",
            "Price Puts",
            "Delta Puts",
            "OI Puts",
        ]
    )
    return df_data


def get_open_interest(date, month, p_or_c, contract):
    """
    Get open interest data for options graphs.

    Args:
        date (datetime): Date to get data for
        month (str): Month to filter by
        p_or_c (str): 'P' for puts or 'C' for calls
        contract (str): Contract identifier

    Returns:
        tuple: (yesterday_data, week_ago_data) as DataFrames
    """
    df_yday, df_yweek = get_options_vol(date, contract)
    df_yweek = df_yweek[df_yweek["Month"] == month]
    df_yweek = df_yweek[df_yweek["P/C"] == p_or_c]
    df_yday = df_yday[df_yday["Month"] == month]
    df_yday = df_yday[df_yday["P/C"] == p_or_c]
    return df_yday, df_yweek


def get_future_price(contract):
    """
    Get futures price data for a specific contract.

    Args:
        contract (str): Contract identifier (EUA, UKA, CB4, RJ4)

    Returns:
        pd.DataFrame: Futures data with Date, settlement, open interest, and volume
    """
    source_file = f"{gcloud_filepath}/dataset.csv"
    df = pd.read_csv(source_file)
    if contract == "EUA" or contract == "UKA":
        df_selected = df[
            [
                "Date",
                f"{contract}_open",
                f"{contract}_low",
                f"{contract}_high",
                f"{contract}_settl",
                f"{contract}_open_int",
                f"{contract}_volume",
            ]
        ]
    else:
        df_selected = df[
            ["Date", f"{contract}_settl", f"{contract}_open_int", f"{contract}_volume"]
        ]
    df_selected["Date"] = pd.to_datetime(df_selected["Date"], format="%Y-%m-%d")

    for col in df_selected.columns:
        if col != "Date":
            df_selected[col] = pd.to_numeric(df_selected[col], errors="coerce")
    return df_selected


def get_contract_calcs(contract, df_price_data):
    """
    Calculate technical indicators for futures contracts.

    Args:
        contract (str): Contract identifier
        df_price_data (pd.DataFrame): Price data DataFrame

    Returns:
        pd.DataFrame: DataFrame with technical indicators (moving averages, Bollinger bands, RSI)
    """
    df = df_price_data.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")

    df["mean10"] = df[f"{contract}_settl"].ffill().rolling(10).mean()
    df["mean20"] = df[f"{contract}_settl"].ffill().rolling(20).mean()
    df["mean50"] = df[f"{contract}_settl"].ffill().rolling(50).mean()
    df["bol_high"] = (
        df["mean20"]
        + df[f"{contract}_settl"].ffill().rolling(window=20, min_periods=20).std(ddof=0)
        * 2
    )
    df["bol_low"] = (
        df["mean20"]
        - df[f"{contract}_settl"].ffill().rolling(window=20, min_periods=20).std(ddof=0)
        * 2
    )

    close_delta = df[f"{contract}_settl"].ffill().diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    rsi = ma_up / ma_down

    df["rsi14"] = 100 - (100 / (1 + rsi))

    if contract in ["EUA", "UKA"]:
        df = df.rename(
            columns={
                f"{contract}_settl": "Last",
                f"{contract}_open_int": "Open_Int",
                f"{contract}_volume": "Volume",
                "Date": "Time",
                f"{contract}_open": "Open",
                f"{contract}_low": "Low",
                f"{contract}_high": "High",
            }
        )
    else:
        df = df.rename(
            columns={
                f"{contract}_settl": "Last",
                f"{contract}_open_int": "Open_Int",
                f"{contract}_volume": "Volume",
                "Date": "Time",
            }
        )
        df["Open"] = df["Last"]
        df["Low"] = df["Last"]
        df["High"] = df["Last"]

    return df


def get_ttf_price():
    """
    Get TTF gas hub price history data.

    Returns:
        pd.DataFrame: TTF price data with Time and Price columns
    """
    source_file = f"{gcloud_filepath}/price_history_data/front_month_TFM.csv"
    df_ttf_price = pd.read_csv(source_file)
    df_ttf_price["Time"] = pd.to_datetime(df_ttf_price["Time"], format="%Y-%m-%d")
    df_ttf_price = df_ttf_price.sort_values(by="Time")
    return df_ttf_price


def get_ttf_prices(start_date="2024-01-01", months_back=6):
    """
    Get TTF gas prices from the SH dataset for the gas page with configurable start date and lookback period.

    Args:
        start_date (str): Start date in YYYY-MM-DD format (default: "2024-01-01")
        months_back (int): Number of months to look back from the end date (default: 6)

    Returns:
        pd.DataFrame: TTF price data with Time and Price columns for the specified period
    """
    df_sh = get_sh_dataset()
    df_ttf = df_sh[["Date", "TTF"]].copy()
    df_ttf.rename(columns={"Date": "Time", "TTF": "Price"}, inplace=True)
    df_ttf["Time"] = pd.to_datetime(df_ttf["Time"])
    
    # Filter by start date
    df_ttf = df_ttf[df_ttf["Time"] >= start_date]
    
    # Get the last 6 months of data from the filtered dataset
    if len(df_ttf) > 0:
        end_date = df_ttf["Time"].max()
        start_lookback = end_date - pd.DateOffset(months=months_back)
        df_ttf = df_ttf[df_ttf["Time"] >= start_lookback]
    
    df_ttf = df_ttf.sort_values(by="Time")
    return df_ttf

def get_ttf_prices_full_range(start_date="2024-01-01"):
    """
    Get TTF gas prices from the SH dataset for the full range since start_date.
    This is used for the range slider to show the complete data range.

    Args:
        start_date (str): Start date in YYYY-MM-DD format (default: "2024-01-01")

    Returns:
        pd.DataFrame: TTF price data with Time and Price columns for the full range
    """
    df_sh = get_sh_dataset()
    df_ttf = df_sh[["Date", "TTF"]].copy()
    df_ttf.rename(columns={"Date": "Time", "TTF": "Price"}, inplace=True)
    df_ttf["Time"] = pd.to_datetime(df_ttf["Time"])
    
    # Filter by start date
    df_ttf = df_ttf[df_ttf["Time"] >= start_date]
    df_ttf = df_ttf.sort_values(by="Time")
    return df_ttf


def get_gas_prices():
    """
    Get gas prices for JKM and TTF hubs with currency conversion.

    Returns:
        pd.DataFrame: Combined gas price data with JKM converted to EUR/MWh
    """
    source_file_jkm = f"{gcloud_filepath}/price_history_data/front_month_JKM.csv"
    source_file_ttf = f"{gcloud_filepath}/price_history_data/front_month_TFM.csv"
    source_file_forex = f"{gcloud_filepath}/price_history_data/forex_rates.csv"
    df_jkm_price = pd.read_csv(source_file_jkm, index_col="Time")
    df_ttf_price = pd.read_csv(source_file_ttf, index_col="Time")
    df_rates_price = pd.read_csv(source_file_forex, index_col="Time")

    df_jkm_price.index = pd.to_datetime(df_jkm_price.index, format="%Y-%m-%d").strftime(
        "%Y-%m-%d"
    )
    df_ttf_price.index = pd.to_datetime(df_ttf_price.index, format="%Y-%m-%d").strftime(
        "%Y-%m-%d"
    )
    df_rates_price.index = pd.to_datetime(
        df_rates_price.index, format="%d-%m-%Y"
    ).strftime("%Y-%m-%d")

    jkm_names = {"Price": "price_jkm", "Volume": "volume_jkm"}
    df_jkm_price = df_jkm_price.rename(columns=jkm_names)[[*jkm_names.values()]]
    ttf_names = {"Price": "price_ttf", "Volume": "volume_ttf"}
    df_ttf_price = df_ttf_price.rename(columns=ttf_names)[[*ttf_names.values()]]

    df_gas_prices = pd.concat([df_jkm_price, df_ttf_price, df_rates_price], axis=1)
    df_gas_prices["price_jkm"] = (
        df_gas_prices["price_jkm"] * 3.412142 / df_gas_prices["USD"]
    )
    df_gas_prices = df_gas_prices.dropna()
    df_gas_prices = df_gas_prices.sort_values(by="Time")

    return df_gas_prices


def get_imp_vol_hist():
    """
    Get implied volatility history data for options.

    Returns:
        pd.DataFrame: Implied volatility data with Time and volatility columns
    """
    source_file = f"{gcloud_filepath}/vol_history_data/vol_dec_{get_dec_year()}_C.csv"
    df_imp_vol_hist = pd.read_csv(source_file)
    df_imp_vol_hist["Time"] = pd.to_datetime(df_imp_vol_hist["Time"], format="%Y-%m-%d")
    df_imp_vol_hist = df_imp_vol_hist.sort_values(by="Time")
    return df_imp_vol_hist


def get_cot_data(contract):
    """
    Get Commitment of Traders (COT) data for futures contracts.

    Args:
        contract (str): Contract identifier

    Returns:
        pd.DataFrame: COT data with renamed columns for better readability
    """
    source_file = (
        f"{gcloud_filepath}/ice_csv_data/cot_data/{contract}_future_cot_data.csv"
    )
    df_cot_data = pd.read_csv(source_file)
    df_cot_data["Date_of_which_the_weekly_report_refers"] = pd.to_datetime(
        df_cot_data["Date_of_which_the_weekly_report_refers"], format="%Y-%m-%d"
    )
    df_cot_data = df_cot_data.rename(
        {
            "NOPS_CommercialUndertakings_Short_Total": "NOPS_Commercial_Undertakings_Short_Total",
            "CSPR_CommercialUndertakings_Long_Total": "CSPR_Commercial_Undertakings_Long_Total",
            "CSPR_CommercialUndertakings_Short_Total": "CSPR_Commercial_Undertakings_Short_Total",
        },
        axis=1,
    )
    return df_cot_data


def update_article_list_format(URL, feed_length):
    """
    Parse RSS feed and format articles for dashboard display.

    Args:
        URL (str): RSS feed URL
        feed_length (int): Number of articles to display

    Returns:
        list: List of formatted Dash Bootstrap Components for news display
    """
    articles = feedparser.parse(URL)
    entry_list = []

    for entry in articles.entries:
        title = entry.title
        link = entry.link
        published = entry.published
        entry = [title, link, published]
        entry_list.append(entry)

    news_items = entry_list[:feed_length]
    news_format = []

    for entry in news_items:
        entry_format = dbc.ListGroupItem(
            [
                html.P(entry[0], className="mb-1"),
                html.Div(
                    [
                        html.Small(entry[2]),
                        html.A(
                            "Link",
                            href=entry[1],
                            target="_blank",
                            className="text-success",
                        ),
                    ],
                    className="d-flex w-100 justify-content-between",
                ),
            ]
        )
        news_format.append(entry_format)
    return news_format


def get_gas_storage():
    """
    Get European gas storage data and pivot for year-over-year comparison.

    Returns:
        pd.DataFrame: Pivoted gas storage data with years as columns
    """
    gas_file = f"{gcloud_filepath}/gas_data/european_gas_storage.csv"
    df_gas = pd.read_csv(gas_file)
    df_gas["gasDayStart"] = pd.to_datetime(df_gas["gasDayStart"], format="%Y-%m-%d")

    df_gas["year"] = df_gas["gasDayStart"].dt.year
    df_gas["day_of_year"] = df_gas["gasDayStart"].dt.strftime("%m-%d")

    df_gas = df_gas.pivot(index="day_of_year", columns="year", values="full")
    df_gas.index = pd.to_datetime("2020-" + df_gas.index, format="%Y-%m-%d")
    df_gas = df_gas.sort_values(by="day_of_year")
    return df_gas


def get_vol_hist(df_price_hist_daily):
    """
    Calculate historical volatility for EUA futures.

    Args:
        df_price_hist_daily (pd.DataFrame): Daily price history data

    Returns:
        pd.DataFrame: Historical volatility data with multiple time horizons
    """
    horizon_list = [10, 20, 50]

    df_hist_vol = df_price_hist_daily[["Time", "Last"]].copy()

    for i in range(1, len(df_hist_vol.Time)):
        df_hist_vol.loc[i, "price_relative"] = (
            df_hist_vol.loc[i, "Last"] / df_hist_vol.loc[i - 1, "Last"]
        )

    for i in range(1, len(df_hist_vol.Time)):
        df_hist_vol.loc[i, "daily_return"] = np.log(
            df_hist_vol.loc[i, "Last"] / df_hist_vol.loc[i - 1, "Last"]
        )

    for horizon in horizon_list:
        df_hist_vol["vol"] = df_hist_vol["daily_return"].rolling(horizon).std()
        df_hist_vol[f"vol_anulised_{horizon}"] = df_hist_vol["vol"] * np.sqrt(252) * 100

    df_hist_vol["Time"] = pd.to_datetime(df_hist_vol["Time"], dayfirst=True)
    df_hist_vol = df_hist_vol.sort_values(by="Time")
    return df_hist_vol


def get_auction_hist(contract):
    """
    Get auction history data for different carbon contracts.

    Args:
        contract (str): Contract type (EUA, UKA, RGGI, WCI)

    Returns:
        pd.DataFrame or tuple: Auction data for the specified contract

    Raises:
        ValueError: If contract type is not supported
    """
    source_file = f"{gcloud_filepath}/dataset.csv"
    df = pd.read_csv(source_file)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")

    if contract == "EUA":
        auction_eu = df[
            [
                "Date",
                "eu_auction_Price",
                "eu_auction_Cover Ratio",
                "eu_auction_Bid_vol_sub",
                "eu_auction_Bid_vol_offered",
            ]
        ]
        auction_de = df[
            [
                "Date",
                "de_auction_Price",
                "de_auction_Cover Ratio",
                "de_auction_Bid_vol_sub",
                "de_auction_Bid_vol_offered",
            ]
        ]
        auction_pl = df[
            [
                "Date",
                "pl_auction_Price",
                "pl_auction_Cover Ratio",
                "pl_auction_Bid_vol_sub",
                "pl_auction_Bid_vol_offered",
            ]
        ]
        return auction_eu, auction_de, auction_pl
    elif contract == "UKA":
        auction_uka = df[
            [
                "Date",
                "UKA_auction_volume",
                "UKA_clearing_price",
                "UKA_Bid_vol_sub",
                "UKA_bidders",
                "UKA_cover_ratio",
                "UKA_successful_bidders",
                "UKA_vol_offered",
            ]
        ]
        return auction_uka
    elif contract == "RGGI":
        auction_rgg = df[
            [
                "Date",
                "RGGI_compl_bidders",
                "RGGI_non_compl_bidders",
                "RGGI_Bid_vol_sub",
                "RGGI_bidders",
                "RGGI_clearing_price",
                "RGGI_cover_ratio",
                "RGGI_vol_offered",
                "RGGI_successful_bidders",
            ]
        ]
        return auction_rgg
    elif contract == "WCI":
        auction_wci = df[
            [
                "Date",
                "WCI_%_compl_bidders",
                "WCI_median_allowance_price",
                "WCI_Bid_vol_sub",
                "WCI_clearing_price",
                "WCI_cover_ratio",
                "WCI_vol_offered",
            ]
        ]
        return auction_wci
    else:
        raise ValueError("Contract not found")


def get_dataset():
    """
    Get the main dataset used for settlement modeling.

    Returns:
        pd.DataFrame: Main dataset with all contract data
    """
    file_path = f"{gcloud_filepath}/dataset.csv"
    dataset_df = pd.read_csv(file_path)
    return dataset_df


def get_eua_price():
    """
    Get EUA price data from the main dataset.

    Returns:
        pd.DataFrame: EUA price data with Time and Last columns
    """
    df_ = get_dataset()
    df_eua_price = df_[["Date", "EUA"]]
    df_eua_price.rename(columns={"Date": "Time", "EUA": "Last"}, inplace=True)
    df_eua_price["Time"] = pd.to_datetime(df_eua_price["Time"])

    return df_eua_price


def get_sh_dataset():
    """
    Get SHCP dataset for price forecasting model.

    Returns:
        pd.DataFrame: Cleaned dataset with renamed columns and interpolated values
    """
    file_path = f"{gcloud_filepath}/price_fcst_model_data/DatasetSHCP.csv"
    df_datasetSHCP = pd.read_csv(file_path)

    df_dataset = df_datasetSHCP.drop(
        columns=[
            "EUA_open",
            "EUA_low",
            "EUA_high",
            "TTF_open",
            "TTF_low",
            "TTF_high",
            "Coal_open",
            "Coal_low",
            "Coal_high",
            "GermanPower(FM)_open",
            "GermanPower(FM)_low",
            "GermanPower(FM)_high",
            "GermanPower(FY)_open",
            "GermanPower(FY)_low",
            "GermanPower(FY)_high",
            "Brent_open",
            "Brent_low",
            "Brent_high",
        ],
        axis=1,
    )

    df_dataset.rename(
        columns={
            "other fin. inst.": "CoT_Other",
            "ops w/ compliance obligations": "CoT_Obligations",
            "inv. funds": "CoT_InvFunds",
            "inv. firms or credit inst.": "CoT_InvFrims_Credit",
            "com. undertakings": "CoT_Undertakings",
        },
        inplace=True,
    )

    df_dataset.interpolate(method="linear", axis=0, inplace=True)
    return df_dataset


def get_sh_beta():
    """
    Get SH beta variations data for price forecasting model.

    Returns:
        tuple: (contribution_data, block_contribution_data) as DataFrames
    """
    file_path_contribution = (
        f"{gcloud_filepath}/price_fcst_model_data/beta_variations1_contribution.csv"
    )
    file_path_blocks = f"{gcloud_filepath}/price_fcst_model_data/beta_variations1_block_contribution.csv"
    df = pd.read_csv(file_path_contribution)
    df_blocks = pd.read_csv(file_path_blocks)
    return df, df_blocks


def get_sh_beta_energy():
    """
    Get SH beta variations data for energy price forecasting model.

    Returns:
        tuple: (contribution_data, block_contribution_data) as DataFrames
    """
    file_path_contribution = (
        f"{gcloud_filepath}/price_fcst_model_data/beta_variations2_contribution.csv"
    )
    file_path_blocks = f"{gcloud_filepath}/price_fcst_model_data/beta_variations2_block_contribution.csv"
    df = pd.read_csv(file_path_contribution)
    df_blocks = pd.read_csv(file_path_blocks)
    return df, df_blocks


def get_dashboard_summary():
    """
    Get summary data for the dashboard including latest news from multiple sources.
    Uses smart caching to avoid unnecessary RSS fetches and preserve display during updates.
    
    Returns:
        dict: Dictionary containing summary data with news from different categories
    """
    try:
        # Get latest Carbon Pulse news with smart caching
        carbon_news, carbon_changed, _ = get_cached_news_feed(
            "https://carbon-pulse.com/feed/", "carbon_pulse", 3
        )
        
        # Get latest Gas news with smart caching
        gas_news, gas_changed, _ = get_cached_news_feed(
            "https://www.naturalgasintel.com/feed/", "gas_news", 3
        )
        
        # Get latest Power news
        power_news = get_power_news()
        
        # Get latest EUA price
        eua_price_data = get_eua_latest_price()
        
        result = {
            "carbon_news": carbon_news,
            "gas_news": gas_news,
            "power_news": power_news,
            "eua_price": eua_price_data,
            "changes": {
                "carbon_changed": carbon_changed,
                "gas_changed": gas_changed
            }
        }
        print(f"DEBUG: Final result with changes: {result['changes']}")
        return result
        
    except Exception as e:
        print(f"DEBUG: Main exception: {e}")
        return {
            "carbon_news": [],
            "gas_news": [],
            "power_news": [],
            "eua_price": {
                'price': None,
                'timestamp': None,
                'currency': 'EUR',
                'success': False,
                'error': str(e)
            },
            "changes": {
                "carbon_changed": False,
                "gas_changed": False
            }
        }


def get_cached_news_feed(url, feed_name, max_articles=3):
    """
    Smart news fetching with caching and change detection.
    
    Args:
        url (str): RSS feed URL
        feed_name (str): Name of the feed for caching
        max_articles (int): Maximum number of articles to return
        
    Returns:
        tuple: (news_items, has_changed, cache_key)
    """
    global NEWS_CACHE, NEWS_CACHE_TIMESTAMP
    
    current_time = time.time()
    cache_key = f"{feed_name}_{hashlib.md5(url.encode()).hexdigest()}"
    
    # Check if we have valid cached data
    if (cache_key in NEWS_CACHE and 
        cache_key in NEWS_CACHE_TIMESTAMP and 
        current_time - NEWS_CACHE_TIMESTAMP[cache_key] < NEWS_CACHE_DURATION):
        
        # Return cached data with no change flag
        return NEWS_CACHE[cache_key], False, cache_key
    
    try:
        # Fetch new data
        articles = feedparser.parse(url)
        new_news = []
        
        for i, entry in enumerate(articles.entries[:max_articles]):
            news_item = {
                'title': entry.title,
                'link': entry.link,
                'published': entry.published,
                'index': i + 1
            }
            new_news.append(news_item)
        
        # Check if content has actually changed
        has_changed = True
        if cache_key in NEWS_CACHE:
            old_news = NEWS_CACHE[cache_key]
            if len(old_news) == len(new_news):
                # Compare titles to detect changes
                old_titles = [item['title'] for item in old_news]
                new_titles = [item['title'] for item in new_news]
                has_changed = old_titles != new_titles
        
        # Update cache
        NEWS_CACHE[cache_key] = new_news
        NEWS_CACHE_TIMESTAMP[cache_key] = current_time
        
        return new_news, has_changed, cache_key
        
    except Exception as e:
        print(f"Error fetching {feed_name} news: {e}")
        # Return cached data if available, otherwise empty list
        if cache_key in NEWS_CACHE:
            return NEWS_CACHE[cache_key], False, cache_key
        return [], False, cache_key


def get_eua_latest_price():
    """
    Retrieve the latest EUA price from Yahoo Finance using the ECF=F ticker.
    
    Returns:
        dict: Dictionary containing the latest price and timestamp, or error info
    """
    try:
        # Use the ECF=F ticker for EUA futures
        ticker = yf.Ticker("ECF=F")
        
        # Get the latest price from history
        hist = ticker.history(period="1d")
        
        if not hist.empty:
            latest_price = hist['Close'].iloc[-1]
            latest_time = hist.index[-1]
            
            return {
                'price': round(latest_price, 2),
                'timestamp': latest_time.strftime('%d %b'),
                'currency': 'EUR',
                'success': True
            }
        else:
            return {
                'price': None,
                'timestamp': None,
                'currency': 'EUR',
                'success': False,
                'error': 'No price data available'
            }
            
    except Exception as e:
        return {
            'price': None,
            'timestamp': None,
            'currency': 'EUR',
            'success': False,
            'error': str(e)
        }


def get_power_news():
    """
    Get power market news from a relevant RSS feed.
    
    Returns:
        list: List of power news articles
    """
    try:
        # Using the same working feed as the news page
        power_feed_url = "https://www.powermag.com/feed/"
        articles = feedparser.parse(power_feed_url)
        
        power_news = []
        for i, entry in enumerate(articles.entries[:3]):  # Get top 3 articles
            news_item = {
                'title': entry.title,
                'link': entry.link,
                'published': entry.published,
                'index': i + 1
            }
            power_news.append(news_item)
        
        return power_news
    except Exception as e:
        print(f"Error fetching power news: {e}")
        return []
