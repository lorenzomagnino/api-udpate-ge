import pandas as pd
import logging
from datetime import datetime
from veyt_client import VeytAPIClient, save_to_csv
from config import FUTURES, COMMODITIES, VOLATILITY, DEFAULT_FROM_DATE, DEFAULT_UNTIL_DATE, DEFAULT_OUTPUT_DIR


def save_eua_futures_data(client, from_date=DEFAULT_FROM_DATE, until_date=DEFAULT_UNTIL_DATE, output_dir=DEFAULT_OUTPUT_DIR):
    """Process EUA futures data with front and benchmark contract extraction"""
    print(f"Retrieving EUA futures data from {from_date} to {until_date}...")
    
    eua_df = client.get_market_price(FUTURES['eua'], from_trade_date=from_date, until_trade_date=until_date)
    
    if eua_df is None or eua_df.empty:
        print("No EUA data retrieved")
        return None, None
    
    print(f"Retrieved {len(eua_df)} raw EUA records")
    
    print("Saving raw EUA futures data...")
    save_to_csv(eua_df, "eua_futures_raw_data.csv", output_dir)
    
    eua_df['trade_date_dt'] = pd.to_datetime(eua_df['trade_date'])
    eua_df['maturity_date_dt'] = pd.to_datetime(eua_df['maturity_date'])
    
    eua_df['trade_year'] = eua_df['trade_date_dt'].dt.year
    eua_df['trade_month'] = eua_df['trade_date_dt'].dt.month
    eua_df['trade_day'] = eua_df['trade_date_dt'].dt.day
    eua_df['maturity_year'] = eua_df['maturity_date_dt'].dt.year
    eua_df['maturity_month'] = eua_df['maturity_date_dt'].dt.month
    
    result_rows = []
    
    for trade_date in eua_df['trade_date'].unique():
        trade_data = eua_df[eua_df['trade_date'] == trade_date].copy()
        
        trade_dt = pd.to_datetime(trade_date)
        trade_year = trade_dt.year
        trade_month = trade_dt.month
        trade_day = trade_dt.day
        
        front_contracts = trade_data[
            (trade_data['trade_year'] == trade_data['maturity_year']) & 
            (trade_data['trade_month'] == trade_data['maturity_month'])
        ]
        
        if front_contracts.empty:
            next_month = trade_month + 1 if trade_month < 12 else 1
            next_year = trade_year if trade_month < 12 else trade_year + 1
            
            front_contracts = trade_data[
                (trade_data['maturity_year'] == next_year) & 
                (trade_data['maturity_month'] == next_month)
            ]
        
        benchmark_contracts = trade_data[
            (trade_data['maturity_year'] == trade_year) & 
            (trade_data['maturity_month'] == 12)
        ]
        
        if benchmark_contracts.empty and trade_month == 12:
            benchmark_contracts = trade_data[
                (trade_data['maturity_year'] == trade_year + 1) & 
                (trade_data['maturity_month'] == 12)
            ]
            if not benchmark_contracts.empty:
                print(f"Note: Using next year's December contract for trade date {trade_date} (current year December not available)")
        
        row = {'trade_date': trade_date}
        
        if not front_contracts.empty:
            front_contract = front_contracts.iloc[0]
            row.update({
                'EUA_front_maturity_date': front_contract['maturity_date'],
                'EUA_front_open': front_contract['open'],
                'EUA_front_high': front_contract['high'],
                'EUA_front_low': front_contract['low'],
                'EUA_front_settlement': front_contract['settlement'],
                'EUA_front_volume': front_contract['volume'],
                'EUA_front_open_interest': front_contract['open_interest'],
                'EUA_front_modified': front_contract['modified']
            })
        else:
            row.update({
                'EUA_front_maturity_date': None,
                'EUA_front_open': None,
                'EUA_front_high': None,
                'EUA_front_low': None,
                'EUA_front_settlement': None,
                'EUA_front_volume': None,
                'EUA_front_open_interest': None,
                'EUA_front_modified': None
            })
        
        if not benchmark_contracts.empty:
            benchmark_contract = benchmark_contracts.iloc[0]
            row.update({
                'EUA_benchmark_maturity_date': benchmark_contract['maturity_date'],
                'EUA_benchmark_open': benchmark_contract['open'],
                'EUA_benchmark_high': benchmark_contract['high'],
                'EUA_benchmark_low': benchmark_contract['low'],
                'EUA_benchmark_settlement': benchmark_contract['settlement'],
                'EUA_benchmark_volume': benchmark_contract['volume'],
                'EUA_benchmark_open_interest': benchmark_contract['open_interest'],
                'EUA_benchmark_modified': benchmark_contract['modified']
            })
        else:
            row.update({
                'EUA_benchmark_maturity_date': None,
                'EUA_benchmark_open': None,
                'EUA_benchmark_high': None,
                'EUA_benchmark_low': None,
                'EUA_benchmark_settlement': None,
                'EUA_benchmark_volume': None,
                'EUA_benchmark_open_interest': None,
                'EUA_benchmark_modified': None
            })
            print(f"Warning: No December benchmark contract found for trade date {trade_date}")
        
        result_rows.append(row)
    
    result_df = pd.DataFrame(result_rows)
    result_df = result_df.sort_values('trade_date').reset_index(drop=True)
    
    print(f"Processed {len(result_df)} unique trade dates")
    print(f"Found EUA_front contracts for {result_df['EUA_front_open'].notna().sum()} dates")
    print(f"Found EUA_benchmark contracts for {result_df['EUA_benchmark_open'].notna().sum()} dates")
    
    result_df = result_df.set_index('trade_date').sort_index()
    result_df = result_df.fillna(method='ffill')
    result_df = result_df.reset_index()
    
    for col in ['EUA_front_maturity_date', 'EUA_benchmark_maturity_date']:
        if col in result_df.columns:
            result_df[col] = pd.to_datetime(result_df[col]).dt.to_period('M').dt.end_time.dt.date
    
    save_to_csv(result_df, "eua_futures_front_and_benchmark.csv", output_dir)
    
    print(f"Saved raw EUA data to: eua_futures_raw_data.csv")
    print(f"Saved processed EUA data to: eua_futures_front_and_benchmark.csv")
    
    return eua_df, result_df


def get_other_futures_data(client, from_date=DEFAULT_FROM_DATE, until_date=DEFAULT_UNTIL_DATE, output_dir=DEFAULT_OUTPUT_DIR):
    """Get data for other futures (UKA, RGGI, WCI)"""
    print("\n=== Retrieving Other Futures Data ===")
    
    for name, curve_id in FUTURES.items():
        if name == 'eua':
            continue
            
        print(f"\nGetting {name.upper()} futures data...")
        df = client.get_market_price(curve_id, from_trade_date=from_date, until_trade_date=until_date)
        
        if df is not None and not df.empty:
            print(f"Retrieved {len(df)} {name.upper()} records")
            print(f"Columns: {list(df.columns)}")
            print(f"Date range: {df['trade_date'].min()} to {df['trade_date'].max()}")
            print("Sample data:")
            print(df.head(3))
            
            filename = f"{name}_futures_raw_data.csv"
            save_to_csv(df, filename, output_dir)
        else:
            print(f"No {name.upper()} data retrieved")


def get_commodities_data(client, from_date=DEFAULT_FROM_DATE, until_date=DEFAULT_UNTIL_DATE, output_dir=DEFAULT_OUTPUT_DIR):
    """Get data for commodities with front month processing"""
    print("\n=== Retrieving Commodities Data ===")
    
    for name, curve_id in COMMODITIES.items():
        print(f"\nGetting {name} data...")
        df = client.get_market_price(curve_id, from_trade_date=from_date, until_trade_date=until_date)
        
        if df is not None and not df.empty:
            print(f"Retrieved {len(df)} {name} records")
            print(f"Columns: {list(df.columns)}")
            print(f"Date range: {df['trade_date'].min()} to {df['trade_date'].max()}")
            print("Sample data:")
            print(df.head(3))
            
            filename = f"{name.lower().replace(' ', '_')}_raw_data.csv"
            save_to_csv(df, filename, output_dir)
            
            processed_df = process_commodity_front(df, name)
            if processed_df is not None:
                processed_filename = f"{name.lower().replace(' ', '_')}_front_month.csv"
                save_to_csv(processed_df, processed_filename, output_dir)
        else:
            print(f"No {name} data retrieved")


def process_commodity_front(df, name):
    """Process commodity data to extract front month contracts"""
    if df is None or df.empty:
        return None
    
    df['trade_date_dt'] = pd.to_datetime(df['trade_date'])
    df['maturity_date_dt'] = pd.to_datetime(df['maturity_date'])
    
    df['trade_year'] = df['trade_date_dt'].dt.year
    df['trade_month'] = df['trade_date_dt'].dt.month
    df['maturity_year'] = df['maturity_date_dt'].dt.year
    df['maturity_month'] = df['maturity_date_dt'].dt.month
    
    result_rows = []
    
    for trade_date in df['trade_date'].unique():
        trade_data = df[df['trade_date'] == trade_date].copy()
        
        trade_dt = pd.to_datetime(trade_date)
        trade_year = trade_dt.year
        trade_month = trade_dt.month
        
        row = {'trade_date': trade_date}
        
        if 'brent' in name.lower():
            front_contracts, front_month, front_year = get_brent_front_contract(trade_data, trade_year, trade_month)
            benchmark_contracts = get_benchmark_contract(trade_data, trade_year)
            
            add_contract_data(row, front_contracts, f'{name}_front')
            add_contract_data(row, benchmark_contracts, f'{name}_benchmark')
            
        elif 'coal' in name.lower() or 'german' in name.lower():
            front_contracts, front_month, front_year = get_standard_front_contract(trade_data, trade_year, trade_month)
            front2_contracts = get_front2_contract(trade_data, front_month, front_year)
            
            add_contract_data(row, front_contracts, f'{name}_front')
            add_contract_data(row, front2_contracts, f'{name}_front2')
            
        else:
            front_contracts, _, _ = get_standard_front_contract(trade_data, trade_year, trade_month)
            add_contract_data(row, front_contracts, f'{name}_front')
        
        result_rows.append(row)
    
    result_df = pd.DataFrame(result_rows)
    result_df = result_df.sort_values('trade_date').reset_index(drop=True)
    
    result_df = result_df.set_index('trade_date').sort_index()
    result_df = result_df.fillna(method='ffill')
    result_df = result_df.reset_index()
    
    for col in result_df.columns:
        if 'maturity_date' in col:
            result_df[col] = pd.to_datetime(result_df[col]).dt.to_period('M').dt.end_time.dt.date
    
    return result_df


def get_standard_front_contract(trade_data, trade_year, trade_month):
    """Get standard front contract (current month, then next month)"""
    front_contracts = trade_data[
        (trade_data['trade_year'] == trade_data['maturity_year']) & 
        (trade_data['trade_month'] == trade_data['maturity_month'])
    ]
    
    front_month, front_year = trade_month, trade_year
    
    if front_contracts.empty:
        front_month = trade_month + 1 if trade_month < 12 else 1
        front_year = trade_year if trade_month < 12 else trade_year + 1
        
        front_contracts = trade_data[
            (trade_data['maturity_year'] == front_year) & 
            (trade_data['maturity_month'] == front_month)
        ]
    
    return front_contracts, front_month, front_year


def get_brent_front_contract(trade_data, trade_year, trade_month):
    """Get Brent front contract (current, next, or next+2 month)"""
    front_contracts = trade_data[
        (trade_data['trade_year'] == trade_data['maturity_year']) & 
        (trade_data['trade_month'] == trade_data['maturity_month'])
    ]
    
    front_month, front_year = trade_month, trade_year
    
    if front_contracts.empty:
        front_month = trade_month + 1 if trade_month < 12 else 1
        front_year = trade_year if trade_month < 12 else trade_year + 1
        
        front_contracts = trade_data[
            (trade_data['maturity_year'] == front_year) & 
            (trade_data['maturity_month'] == front_month)
        ]
        
        if front_contracts.empty:
            if trade_month <= 10:
                front_month = trade_month + 2
                front_year = trade_year
            elif trade_month == 11:
                front_month = 1
                front_year = trade_year + 1
            else:
                front_month = 2
                front_year = trade_year + 1
            
            front_contracts = trade_data[
                (trade_data['maturity_year'] == front_year) & 
                (trade_data['maturity_month'] == front_month)
            ]
    
    return front_contracts, front_month, front_year


def get_front2_contract(trade_data, front_month, front_year):
    """Get front2 contract (month after front)"""
    front2_month = front_month + 1 if front_month < 12 else 1
    front2_year = front_year if front_month < 12 else front_year + 1
    
    front2_contracts = trade_data[
        (trade_data['maturity_year'] == front2_year) & 
        (trade_data['maturity_month'] == front2_month)
    ]
    
    return front2_contracts


def get_benchmark_contract(trade_data, trade_year):
    """Get benchmark contract (December of current year)"""
    benchmark_contracts = trade_data[
        (trade_data['maturity_year'] == trade_year) & 
        (trade_data['maturity_month'] == 12)
    ]
    
    return benchmark_contracts


def add_contract_data(row, contracts, prefix):
    """Add contract data to row with given prefix"""
    if not contracts.empty:
        contract = contracts.iloc[0]
        row.update({
            f'{prefix}_maturity_date': contract['maturity_date'],
            f'{prefix}_open': contract['open'],
            f'{prefix}_high': contract['high'],
            f'{prefix}_low': contract['low'],
            f'{prefix}_settlement': contract['settlement'],
            f'{prefix}_volume': contract['volume'],
            f'{prefix}_open_interest': contract['open_interest'],
            f'{prefix}_modified': contract['modified']
        })
    else:
        row.update({
            f'{prefix}_maturity_date': None,
            f'{prefix}_open': None,
            f'{prefix}_high': None,
            f'{prefix}_low': None,
            f'{prefix}_settlement': None,
            f'{prefix}_volume': None,
            f'{prefix}_open_interest': None,
            f'{prefix}_modified': None
        })


def get_volatility_data(client, from_date=DEFAULT_FROM_DATE, until_date=DEFAULT_UNTIL_DATE, output_dir=DEFAULT_OUTPUT_DIR):
    """Get volatility data with benchmark processing"""
    print("\n=== Retrieving Volatility Data ===")
    
    for name, curve_id in VOLATILITY.items():
        print(f"\nGetting {name} volatility data...")
        df = client.get_market_price(curve_id, from_trade_date=from_date, until_trade_date=until_date)
        
        if df is not None and not df.empty:
            print(f"Retrieved {len(df)} {name} volatility records")
            print(f"Columns: {list(df.columns)}")
            print(f"Date range: {df['trade_date'].min()} to {df['trade_date'].max()}")
            print("Sample data:")
            print(df.head(3))
            
            filename = f"volatility_{name}_raw_data.csv"
            save_to_csv(df, filename, output_dir)
            
            processed_df = process_volatility_benchmark(df, name)
            if processed_df is not None:
                processed_filename = f"volatility_{name}_benchmark.csv"
                save_to_csv(processed_df, processed_filename, output_dir)
        else:
            print(f"No {name} volatility data retrieved")


def process_volatility_benchmark(df, name):
    """Process volatility data to extract benchmark contracts"""
    if df is None or df.empty:
        return None
    
    df['trade_date_dt'] = pd.to_datetime(df['trade_date'])
    df['maturity_date_dt'] = pd.to_datetime(df['maturity_date'])
    
    df['trade_year'] = df['trade_date_dt'].dt.year
    df['maturity_year'] = df['maturity_date_dt'].dt.year
    df['maturity_month'] = df['maturity_date_dt'].dt.month
    
    result_rows = []
    
    for trade_date in df['trade_date'].unique():
        trade_data = df[df['trade_date'] == trade_date].copy()
        
        trade_dt = pd.to_datetime(trade_date)
        trade_year = trade_dt.year
        
        benchmark_contracts = trade_data[
            (trade_data['maturity_year'] == trade_year) & 
            (trade_data['maturity_month'] == 12)
        ]
        
        row = {'trade_date': trade_date}
        
        if not benchmark_contracts.empty:
            benchmark_contract = benchmark_contracts.iloc[0]
            row.update({
                f'{name}_benchmark_maturity_date': benchmark_contract['maturity_date'],
                f'{name}_benchmark_open': benchmark_contract['open'],
                f'{name}_benchmark_high': benchmark_contract['high'],
                f'{name}_benchmark_low': benchmark_contract['low'],
                f'{name}_benchmark_settlement': benchmark_contract['settlement'],
                f'{name}_benchmark_volume': benchmark_contract['volume'],
                f'{name}_benchmark_open_interest': benchmark_contract['open_interest'],
                f'{name}_benchmark_modified': benchmark_contract['modified']
            })
        else:
            row.update({
                f'{name}_benchmark_maturity_date': None,
                f'{name}_benchmark_open': None,
                f'{name}_benchmark_high': None,
                f'{name}_benchmark_low': None,
                f'{name}_benchmark_settlement': None,
                f'{name}_benchmark_volume': None,
                f'{name}_benchmark_open_interest': None,
                f'{name}_benchmark_modified': None
            })
        
        result_rows.append(row)
    
    result_df = pd.DataFrame(result_rows)
    result_df = result_df.sort_values('trade_date').reset_index(drop=True)
    
    result_df = result_df.set_index('trade_date').sort_index()
    result_df = result_df.fillna(method='ffill')
    result_df = result_df.reset_index()
    
    maturity_col = f'{name}_benchmark_maturity_date'
    if maturity_col in result_df.columns:
        result_df[maturity_col] = pd.to_datetime(result_df[maturity_col]).dt.to_period('M').dt.end_time.dt.date
    
    return result_df


def process_all_futures_and_commodities(from_date=DEFAULT_FROM_DATE, until_date=DEFAULT_UNTIL_DATE, output_dir=DEFAULT_OUTPUT_DIR):
    """Main function to process all futures and commodities data"""
    client = VeytAPIClient()
    
    print("Starting futures and commodities data retrieval...")
    
    eua_raw, eua_processed = save_eua_futures_data(client, from_date, until_date, output_dir)
    
    get_other_futures_data(client, from_date, until_date, output_dir)
    
    get_commodities_data(client, from_date, until_date, output_dir)
    
    get_volatility_data(client, from_date, until_date, output_dir)
    
    print("\n=== Futures and Commodities Data Retrieval Complete ===")
    
    return eua_raw, eua_processed


if __name__ == "__main__":
    process_all_futures_and_commodities()