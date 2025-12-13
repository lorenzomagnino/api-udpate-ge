import pandas as pd
import logging
from veyt.veyt_client import VeytAPIClient, save_to_csv
from veyt.config import EU_AUCTIONS, DEFAULT_FROM_DATE, DEFAULT_UNTIL_DATE, DEFAULT_OUTPUT_DIR


def get_auction_data(client, auction_dict, auction_name, from_date=DEFAULT_FROM_DATE, until_date=DEFAULT_UNTIL_DATE, output_dir=DEFAULT_OUTPUT_DIR):
    logging.debug(f"\n=== Retrieving {auction_name} Auction Data ===")
    auction_data = {}
    
    for category, curve_id in auction_dict.items():
        logging.debug(f"\nGetting {auction_name} {category} data...")
        df = client.get_tsgroup_data(curve_id, from_date=from_date, until_date=until_date)
        
        if df is not None and not df.empty:
            logging.debug(f"Retrieved {len(df)} {category} records")
            
            if 'times' in df.columns:
                df['timestamp'] = pd.to_datetime(df['times'], unit='ms')
                df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
                logging.debug(f"Date range: {df['date'].min()} to {df['date'].max()}")
            elif 'date' in df.columns:
                logging.debug(f"Date range: {df['date'].min()} to {df['date'].max()}")
            
            df['auction_type'] = category
            
            safe_category = category.lower().replace(' ', '_').replace('/', '_')
            safe_auction_name = auction_name.lower().replace(' ', '_')
            filename = f"{safe_auction_name}_auction_{safe_category}.csv"
            save_to_csv(df, filename, output_dir)
            auction_data[category] = df
        else:
            logging.debug(f"No {auction_name} {category} data retrieved")
            auction_data[category] = None
    
    return auction_data


def create_combined_eu_auctions(auction_data, output_dir=DEFAULT_OUTPUT_DIR):
    logging.debug("\n=== Creating Combined EU Auctions Dataset ===")
    
    auction_dfs = []
    for category, df in auction_data.items():
        if df is not None and not df.empty:
            logging.debug(f"Adding {category} auction data: {len(df)} records")
            auction_dfs.append(df)
    
    if not auction_dfs:
        logging.debug("No auction data available to combine")
        return None
    
    combined_df = pd.concat(auction_dfs, ignore_index=True)
    
    if 'timestamp' in combined_df.columns:
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    elif 'date' in combined_df.columns:
        combined_df = combined_df.sort_values('date').reset_index(drop=True)
    
    logging.debug(f"Combined dataset: {len(combined_df)} records, {combined_df['date'].min()} to {combined_df['date'].max()}")
    logging.debug(f"Auction types: {combined_df['auction_type'].value_counts().to_dict()}")
    
    save_to_csv(combined_df, "eu_auctions_combined.csv", output_dir)
    
    daily_summary = combined_df.groupby('date').agg({
        'auction_type': 'first',
        combined_df.columns[1]: 'count'
    }).reset_index()
    daily_summary.columns = ['date', 'auction_type', 'record_count']
    
    save_to_csv(daily_summary, "eu_auctions_daily_summary.csv", output_dir)
    
    return combined_df


def create_auction_summary(all_auction_data, output_dir=DEFAULT_OUTPUT_DIR):
    logging.debug("\n=== Creating Auction Summary ===")
    
    summary_rows = []
    for auction_type, auction_data in all_auction_data.items():
        for category, df in auction_data.items():
            if df is not None and not df.empty:
                row = {
                    'auction_type': auction_type,
                    'category': category,
                    'records': len(df),
                    'columns': ', '.join(df.columns),
                    'date_min': df['date'].min() if 'date' in df.columns else 'N/A',
                    'date_max': df['date'].max() if 'date' in df.columns else 'N/A'
                }
                summary_rows.append(row)
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        logging.debug("Auction data summary:")
        logging.debug(summary_df.to_string(index=False))
        save_to_csv(summary_df, "auction_data_summary.csv", output_dir)
    
    return summary_df if summary_rows else None


def process_all_auction_data(from_date=DEFAULT_FROM_DATE, until_date=DEFAULT_UNTIL_DATE, output_dir=DEFAULT_OUTPUT_DIR):
    client = VeytAPIClient()
    logging.debug("Starting auction data retrieval...")
    
    all_auction_data = {}
    all_auction_data['EU'] = get_auction_data(client, EU_AUCTIONS, "EU", from_date, until_date, output_dir)
    
    combined_eu_df = create_combined_eu_auctions(all_auction_data['EU'], output_dir)
    summary_df = create_auction_summary(all_auction_data, output_dir)
    
    logging.debug("\n=== Auction Data Retrieval Complete ===")
    
    total_datasets = 0
    total_records = 0
    for auction_type, auction_data in all_auction_data.items():
        datasets_count = len([k for k, v in auction_data.items() if v is not None])
        records_count = sum([len(v) for v in auction_data.values() if v is not None])
        total_datasets += datasets_count
        total_records += records_count
        logging.debug(f"{auction_type} auctions: {datasets_count} datasets, {records_count} total records")
    
    logging.debug(f"Total: {total_datasets} datasets, {total_records} total records")
    
    if combined_eu_df is not None:
        logging.debug(f"Combined EU auctions dataset: {len(combined_eu_df)} total records")
    
    return all_auction_data, summary_df


if __name__ == "__main__":
    process_all_auction_data()