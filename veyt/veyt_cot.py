import pandas as pd
from veyt.veyt_client import VeytAPIClient, save_to_csv
from veyt.config import COT_ICE, COT_EEX, COT_TTF, DEFAULT_FROM_DATE, DEFAULT_UNTIL_DATE, DEFAULT_OUTPUT_DIR
import logging

def standardize_date_column(df):
    """Standardize date column to 'Date'"""
    if df is None or df.empty:
        return df
    
    df = df.copy()
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'Date'})
    elif 'times' in df.columns and 'Date' not in df.columns:
        df['Date'] = pd.to_datetime(df['times'], unit='ms').dt.strftime('%Y-%m-%d')
    
    return df


def merge_auction_data(auction_data_dict, output_dir=DEFAULT_OUTPUT_DIR):
    """Merge auction data from different countries (only one auction per day)"""
    logging.debug("\n=== Merging Auction Data ===")
    
    auction_dfs = []
    for country, df in auction_data_dict.items():
        if df is not None and not df.empty:
            df = standardize_date_column(df)
            df['country'] = country
            auction_dfs.append(df)
    
    if not auction_dfs:
        logging.debug("No auction data to merge")
        return None
    
    merged_auctions = pd.concat(auction_dfs, ignore_index=True)
    merged_auctions = merged_auctions.sort_values('Date')
    
    logging.debug(f"Merged auction data: {len(merged_auctions)} total records")
    logging.debug(f"Date range: {merged_auctions['Date'].min()} to {merged_auctions['Date'].max()}")
    
    filename = "merged_auctions.csv"
    save_to_csv(merged_auctions, filename, output_dir)
    
    return merged_auctions


def create_cot_wide_format(ice_data, eex_data, output_dir=DEFAULT_OUTPUT_DIR):
    """Create wide format COT data with renamed columns"""
    logging.debug("\n=== Creating COT Wide Format ===")
    
    wide_dfs = []
    
    for exchange, data_dict in [('ice', ice_data), ('eex', eex_data)]:
        if not data_dict:
            continue
            
        for category, df in data_dict.items():
            if df is None or df.empty:
                continue
                
            df = standardize_date_column(df)
            
            # Get numeric columns (exclude Date, times, timestamp, exchange)
            exclude_cols = ['Date', 'times', 'timestamp', 'exchange']
            numeric_cols = [col for col in df.columns if col not in exclude_cols]
            
            # Rename columns with prefix
            rename_dict = {col: f"{category}_{exchange}_{col}" for col in numeric_cols}
            df_renamed = df[['Date'] + numeric_cols].rename(columns=rename_dict)
            
            wide_dfs.append(df_renamed)
    
    if not wide_dfs:
        logging.debug("No COT data to create wide format")
        return None
    
    # Merge all on Date
    cot_wide = wide_dfs[0]
    for df in wide_dfs[1:]:
        cot_wide = cot_wide.merge(df, on='Date', how='outer')
    
    cot_wide = cot_wide.sort_values('Date')
    
    logging.debug(f"COT wide format: {len(cot_wide)} rows, {len(cot_wide.columns)} columns")
    logging.debug(f"Date range: {cot_wide['Date'].min()} to {cot_wide['Date'].max()}")
    
    filename = "cot_wide_format.csv"
    save_to_csv(cot_wide, filename, output_dir)
    
    return cot_wide


def create_ttf_wide_format(ttf_data, output_dir=DEFAULT_OUTPUT_DIR):
    """Create wide format TTF COT data with renamed columns"""
    logging.debug("\n=== Creating TTF COT Wide Format ===")
    
    if not ttf_data:
        logging.debug("No TTF data to create wide format")
        return None
    
    wide_dfs = []
    
    for category, df in ttf_data.items():
        if df is None or df.empty:
            continue
            
        df = standardize_date_column(df)
        
        # Get numeric columns (exclude Date, times, timestamp, exchange)
        exclude_cols = ['Date', 'times', 'timestamp', 'exchange']
        numeric_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Rename columns with prefix
        rename_dict = {col: f"{category}_ttf_{col}" for col in numeric_cols}
        df_renamed = df[['Date'] + numeric_cols].rename(columns=rename_dict)
        
        wide_dfs.append(df_renamed)
    
    if not wide_dfs:
        logging.debug("No TTF COT data to create wide format")
        return None
    
    # Merge all on Date
    ttf_wide = wide_dfs[0]
    for df in wide_dfs[1:]:
        ttf_wide = ttf_wide.merge(df, on='Date', how='outer')
    
    ttf_wide = ttf_wide.sort_values('Date')
    
    logging.debug(f"TTF COT wide format: {len(ttf_wide)} rows, {len(ttf_wide.columns)} columns")
    logging.debug(f"Date range: {ttf_wide['Date'].min()} to {ttf_wide['Date'].max()}")
    
    filename = "ttf_cot_wide_format.csv"
    save_to_csv(ttf_wide, filename, output_dir)
    
    return ttf_wide


def create_cot_summed_by_type(ice_data, eex_data, output_dir=DEFAULT_OUTPUT_DIR):
    """Create COT data summed across exchanges but preserving types"""
    logging.debug("\n=== Creating COT Summed by Type ===")
    
    type_dfs = []
    
    # Get all unique categories across both exchanges
    ice_categories = set(ice_data.keys()) if ice_data else set()
    eex_categories = set(eex_data.keys()) if eex_data else set()
    all_categories = ice_categories.union(eex_categories)
    
    for category in all_categories:
        ice_df = ice_data.get(category) if ice_data else None
        eex_df = eex_data.get(category) if eex_data else None
        
        category_dfs = []
        
        # Process ICE data for this category
        if ice_df is not None and not ice_df.empty:
            ice_df = standardize_date_column(ice_df.copy())
            exclude_cols = ['Date', 'times', 'timestamp', 'exchange']
            numeric_cols = [col for col in ice_df.columns if col not in exclude_cols and ice_df[col].dtype in ['int64', 'float64']]
            ice_numeric = ice_df[['Date'] + numeric_cols].copy()
            category_dfs.append(ice_numeric)
        
        # Process EEX data for this category  
        if eex_df is not None and not eex_df.empty:
            eex_df = standardize_date_column(eex_df.copy())
            exclude_cols = ['Date', 'times', 'timestamp', 'exchange']
            numeric_cols = [col for col in eex_df.columns if col not in exclude_cols and eex_df[col].dtype in ['int64', 'float64']]
            eex_numeric = eex_df[['Date'] + numeric_cols].copy()
            category_dfs.append(eex_numeric)
        
        # Sum across exchanges for this category
        if category_dfs:
            combined_category = pd.concat(category_dfs, ignore_index=True)
            summed_category = combined_category.groupby('Date').sum().reset_index()
            
            # Rename columns to include category prefix
            rename_dict = {col: f"{category}_{col}" for col in summed_category.columns if col != 'Date'}
            summed_category = summed_category.rename(columns=rename_dict)
            
            type_dfs.append(summed_category)
    
    if not type_dfs:
        logging.debug("No COT data to sum by type")
        return None
    
    # Merge all categories on Date
    cot_by_type = type_dfs[0]
    for df in type_dfs[1:]:
        cot_by_type = cot_by_type.merge(df, on='Date', how='outer')
    
    cot_by_type = cot_by_type.sort_values('Date')
    
    logging.debug(f"COT summed by type: {len(cot_by_type)} rows, {len(cot_by_type.columns)} columns")
    logging.debug(f"Date range: {cot_by_type['Date'].min()} to {cot_by_type['Date'].max()}")
    logging.debug(f"Categories preserved: {list(all_categories)}")
    
    filename = "cot_summed_by_type.csv"
    save_to_csv(cot_by_type, filename, output_dir)
    
    return cot_by_type


def create_ttf_summed_by_type(ttf_data, output_dir=DEFAULT_OUTPUT_DIR):
    """Create TTF COT data summed by type (separate from other COT data)"""
    logging.debug("\n=== Creating TTF COT Summed by Type ===")
    
    if not ttf_data:
        logging.debug("No TTF data to sum by type")
        return None
    
    type_dfs = []
    ttf_categories = set(ttf_data.keys())
    
    for category in ttf_categories:
        ttf_df = ttf_data.get(category)
        
        if ttf_df is not None and not ttf_df.empty:
            ttf_df = standardize_date_column(ttf_df.copy())
            exclude_cols = ['Date', 'times', 'timestamp', 'exchange']
            numeric_cols = [col for col in ttf_df.columns if col not in exclude_cols and ttf_df[col].dtype in ['int64', 'float64']]
            ttf_numeric = ttf_df[['Date'] + numeric_cols].copy()
            
            # Rename columns to include TTF_ prefix and category
            rename_dict = {col: f"TTF_{category}_{col}" for col in ttf_numeric.columns if col != 'Date'}
            ttf_renamed = ttf_numeric.rename(columns=rename_dict)
            
            type_dfs.append(ttf_renamed)
    
    if not type_dfs:
        logging.debug("No TTF COT data to sum by type")
        return None
    
    # Merge all TTF categories on Date
    ttf_by_type = type_dfs[0]
    for df in type_dfs[1:]:
        ttf_by_type = ttf_by_type.merge(df, on='Date', how='outer')
    
    ttf_by_type = ttf_by_type.sort_values('Date')
    
    logging.debug(f"TTF COT summed by type: {len(ttf_by_type)} rows, {len(ttf_by_type.columns)} columns")
    logging.debug(f"Date range: {ttf_by_type['Date'].min()} to {ttf_by_type['Date'].max()}")
    logging.debug(f"TTF categories preserved: {list(ttf_categories)}")
    
    filename = "ttf_cot_summed_by_type.csv"
    save_to_csv(ttf_by_type, filename, output_dir)
    
    return ttf_by_type


def create_cot_summed(ice_data, eex_data, output_dir=DEFAULT_OUTPUT_DIR):
    """Create summed COT data across all categories and exchanges"""
    logging.debug("\n=== Creating COT Summed Data ===")
    
    all_dfs = []
    
    for exchange, data_dict in [('ice', ice_data), ('eex', eex_data)]:
        if not data_dict:
            continue
            
        for category, df in data_dict.items():
            if df is None or df.empty:
                continue
                
            df = standardize_date_column(df)
            
            # Get numeric columns only
            exclude_cols = ['Date', 'times', 'timestamp', 'exchange']
            numeric_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
            
            df_numeric = df[['Date'] + numeric_cols].copy()
            all_dfs.append(df_numeric)
    
    if not all_dfs:
        logging.debug("No COT data to sum")
        return None
    
    # Combine all data
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Group by date and sum all numeric columns
    cot_summed = combined_df.groupby('Date').sum().reset_index()
    cot_summed = cot_summed.sort_values('Date')
    
    logging.debug(f"COT summed data: {len(cot_summed)} rows")
    logging.debug(f"Date range: {cot_summed['Date'].min()} to {cot_summed['Date'].max()}")
    logging.debug(f"Summed columns: {[col for col in cot_summed.columns if col != 'Date']}")
    
    filename = "cot_summed.csv"
    save_to_csv(cot_summed, filename, output_dir)
    
    return cot_summed


def create_ttf_summed(ttf_data, output_dir=DEFAULT_OUTPUT_DIR):
    """Create summed TTF COT data across all categories"""
    logging.debug("\n=== Creating TTF COT Summed Data ===")
    
    if not ttf_data:
        logging.debug("No TTF data to sum")
        return None
    
    all_dfs = []
    
    for category, df in ttf_data.items():
        if df is None or df.empty:
            continue
            
        df = standardize_date_column(df)
        
        # Get numeric columns only
        exclude_cols = ['Date', 'times', 'timestamp', 'exchange']
        numeric_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        
        df_numeric = df[['Date'] + numeric_cols].copy()
        all_dfs.append(df_numeric)
    
    if not all_dfs:
        logging.debug("No TTF data to sum")
        return None
    
    # Combine all TTF data
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Group by date and sum all numeric columns
    ttf_summed = combined_df.groupby('Date').sum().reset_index()
    ttf_summed = ttf_summed.sort_values('Date')
    
    logging.debug(f"TTF summed data: {len(ttf_summed)} rows")
    logging.debug(f"Date range: {ttf_summed['Date'].min()} to {ttf_summed['Date'].max()}")
    logging.debug(f"Summed columns: {[col for col in ttf_summed.columns if col != 'Date']}")
    
    filename = "ttf_cot_summed.csv"
    save_to_csv(ttf_summed, filename, output_dir)
    
    return ttf_summed


def get_cot_data(client, cot_dict, exchange_name, from_date=DEFAULT_FROM_DATE, until_date=DEFAULT_UNTIL_DATE, output_dir=DEFAULT_OUTPUT_DIR):
    logging.debug(f"\n=== Retrieving {exchange_name} COT Data ===")
    
    cot_data = {}
    
    for category, curve_id in cot_dict.items():
        logging.debug(f"\nGetting {exchange_name} {category} data...")
        
        df = client.get_tsgroup_data(curve_id, from_date=from_date, until_date=until_date)
        
        if df is not None and not df.empty:
            df = standardize_date_column(df)
            
            logging.debug(f"Retrieved {len(df)} {category} records")
            logging.debug(f"Columns: {list(df.columns)}")
            logging.debug(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
            logging.debug("Sample data:")
            logging.debug(df.head(3))
            
            filename = f"{exchange_name.lower()}_cot_{category.lower()}.csv"
            save_to_csv(df, filename, output_dir)
            cot_data[category] = df
        else:
            logging.debug(f"No {exchange_name} {category} data retrieved")
            cot_data[category] = None
    
    return cot_data


def create_combined_cot_data(ice_data, eex_data, output_dir=DEFAULT_OUTPUT_DIR):
    logging.debug("\n=== Creating Combined COT Data ===")
    
    combined_data = {}
    ice_categories = set(ice_data.keys()) if ice_data else set()
    eex_categories = set(eex_data.keys()) if eex_data else set()
    common_categories = ice_categories.intersection(eex_categories)
    
    for category in common_categories:
        ice_df = ice_data.get(category)
        eex_df = eex_data.get(category)
        
        if ice_df is not None and eex_df is not None and not ice_df.empty and not eex_df.empty:
            logging.debug(f"\nCombining {category} data from ICE and EEX...")
            
            ice_df_copy = standardize_date_column(ice_df.copy())
            eex_df_copy = standardize_date_column(eex_df.copy())
            ice_df_copy['exchange'] = 'ICE'
            eex_df_copy['exchange'] = 'EEX'
            
            combined_df = pd.concat([ice_df_copy, eex_df_copy], ignore_index=True)
            combined_df = combined_df.sort_values('Date')
            
            logging.debug(f"Combined {category} data has {len(combined_df)} total records")
            
            filename = f"combined_cot_{category.lower()}.csv"
            save_to_csv(combined_df, filename, output_dir)
            combined_data[category] = combined_df
    
    return combined_data


def process_all_cot_data(from_date=DEFAULT_FROM_DATE, until_date=DEFAULT_UNTIL_DATE, output_dir=DEFAULT_OUTPUT_DIR, include_auctions=False, auction_data=None, include_ttf=True):
    client = VeytAPIClient()
    
    logging.debug("Starting COT data retrieval and transformation...")
    
    ice_cot_data = get_cot_data(client, COT_ICE, "ICE", from_date, until_date, output_dir)
    eex_cot_data = get_cot_data(client, COT_EEX, "EEX", from_date, until_date, output_dir)
    
    # Get TTF data separately if requested
    ttf_cot_data = None
    ttf_wide = None
    ttf_summed = None
    ttf_summed_by_type = None
    
    if include_ttf:
        ttf_cot_data = get_cot_data(client, COT_TTF, "ICE_TTF", from_date, until_date, output_dir)
        
        # Create separate TTF transformations
        ttf_wide = create_ttf_wide_format(ttf_cot_data, output_dir)
        ttf_summed = create_ttf_summed(ttf_cot_data, output_dir)
        ttf_summed_by_type = create_ttf_summed_by_type(ttf_cot_data, output_dir)
    
    # Create regular COT transformations (ICE + EEX only)
    combined_cot_data = create_combined_cot_data(ice_cot_data, eex_cot_data, output_dir)
    cot_wide = create_cot_wide_format(ice_cot_data, eex_cot_data, output_dir)
    cot_summed = create_cot_summed(ice_cot_data, eex_cot_data, output_dir)
    cot_summed_by_type = create_cot_summed_by_type(ice_cot_data, eex_cot_data, output_dir)
    
    # Handle auctions if provided
    merged_auctions = None
    if include_auctions and auction_data:
        merged_auctions = merge_auction_data(auction_data, output_dir)
    
    logging.debug("\n=== COT Data Processing Complete ===")
    logging.debug(f"ICE COT categories retrieved: {len([k for k, v in ice_cot_data.items() if v is not None])}")
    logging.debug(f"EEX COT categories retrieved: {len([k for k, v in eex_cot_data.items() if v is not None])}")
    if include_ttf and ttf_cot_data:
        logging.debug(f"TTF COT categories retrieved: {len([k for k, v in ttf_cot_data.items() if v is not None])}")
    logging.debug(f"Combined categories created: {len(combined_cot_data)}")
    logging.debug(f"COT wide format created: {'Yes' if cot_wide is not None else 'No'}")
    logging.debug(f"COT summed data created: {'Yes' if cot_summed is not None else 'No'}")
    logging.debug(f"COT summed by type created: {'Yes' if cot_summed_by_type is not None else 'No'}")
    if include_ttf:
        logging.debug(f"TTF wide format created: {'Yes' if ttf_wide is not None else 'No'}")
        logging.debug(f"TTF summed data created: {'Yes' if ttf_summed is not None else 'No'}")
        logging.debug(f"TTF summed by type created: {'Yes' if ttf_summed_by_type is not None else 'No'}")
    if include_auctions:
        logging.debug(f"Merged auctions created: {'Yes' if merged_auctions is not None else 'No'}")
    
    return {
        'ice_cot_data': ice_cot_data,
        'eex_cot_data': eex_cot_data,
        'ttf_cot_data': ttf_cot_data,
        'combined_cot_data': combined_cot_data,
        'cot_wide': cot_wide,
        'cot_summed': cot_summed,
        'cot_summed_by_type': cot_summed_by_type,
        'ttf_wide': ttf_wide,
        'ttf_summed': ttf_summed,
        'ttf_summed_by_type': ttf_summed_by_type,
        'merged_auctions': merged_auctions
    }


if __name__ == "__main__":
    # Basic COT processing with TTF kept separate
    results = process_all_cot_data(include_ttf=True)
    
    # Example with auction data (uncomment and modify as needed)
    # auction_data = {
    #     'EU': eu_auction_df,
    #     'Poland': poland_auction_df, 
    #     'Germany': germany_auction_df
    # }
    # results = process_all_cot_data(include_auctions=True, auction_data=auction_data, include_ttf=True)