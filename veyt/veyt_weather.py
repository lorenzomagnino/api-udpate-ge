import pandas as pd
import logging
from veyt.veyt_client import VeytAPIClient, save_to_csv
from veyt.config import WEATHER, DEFAULT_FROM_DATE, DEFAULT_UNTIL_DATE, DEFAULT_OUTPUT_DIR


def format_weather_data(df: pd.DataFrame, weather_type: str) -> pd.DataFrame:
    """Format weather data with timestamp conversions"""
    if df is None or df.empty:
        return None
    
    formatted_df = df.copy()
    
    if 'times' in formatted_df.columns:
        formatted_df['timestamp'] = pd.to_datetime(formatted_df['times'], unit='ms')
        formatted_df['date'] = formatted_df['timestamp'].dt.strftime('%Y-%m-%d')
        formatted_df['datetime'] = formatted_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        formatted_df = formatted_df[['date', 'datetime', 'times', 'values', 'modified', 'value_modified']]
    
    formatted_df = formatted_df.rename(columns={
        'values': f'{weather_type.lower()}_value',
        'modified': 'last_modified',
        'value_modified': 'value_last_modified'
    })
    
    return formatted_df


def get_weather_data(client, from_date=DEFAULT_FROM_DATE, until_date=DEFAULT_UNTIL_DATE, output_dir=DEFAULT_OUTPUT_DIR):
    """Get and process weather data (HDD and CDD)"""
    logging.debug("\n=== Retrieving Weather Data ===")
    
    weather_data = {}
    
    for weather_type, curve_id in WEATHER.items():
        logging.debug(f"\nGetting {weather_type.upper()} timeseries data...")
        df = client.get_timeseries_price(curve_id, from_date=from_date, until_date=until_date)
        
        if df is not None and not df.empty:
            formatted_df = format_weather_data(df, weather_type.upper())
            logging.debug(f"Retrieved {len(formatted_df)} {weather_type.upper()} records")
            logging.debug(f"Columns: {list(formatted_df.columns)}")
            logging.debug("Sample data:")
            logging.debug(formatted_df.head(3))
            
            filename = f"{weather_type}_timeseries_data.csv"
            save_to_csv(formatted_df, filename, output_dir)
            weather_data[weather_type] = formatted_df
        else:
            logging.debug(f"No {weather_type.upper()} data retrieved")
            weather_data[weather_type] = None
    
    return weather_data


def create_combined_weather_data(weather_data, output_dir=DEFAULT_OUTPUT_DIR):
    """Combine HDD and CDD data and create aggregations"""
    logging.debug("\n=== Creating Combined Weather Data ===")
    
    hdd_data = weather_data.get('hdd')
    cdd_data = weather_data.get('cdd')
    
    if hdd_data is not None and cdd_data is not None and not hdd_data.empty and not cdd_data.empty:
        # Combine HDD and CDD data
        combined_df = pd.merge(
            hdd_data[['date', 'datetime', 'hdd_value']],
            cdd_data[['date', 'datetime', 'cdd_value']],
            on=['date', 'datetime'],
            how='outer'
        )
        
        combined_df = combined_df.sort_values('datetime')
        
        logging.debug(f"Combined data has {len(combined_df)} records")
        logging.debug("Sample combined data:")
        logging.debug(combined_df.head(3))
        save_to_csv(combined_df, "combined_hdd_cdd_data.csv", output_dir)
        
        # Create daily aggregated data
        logging.debug("Creating daily aggregated data...")
        daily_aggregate = combined_df.groupby('date').agg({
            'hdd_value': 'sum',
            'cdd_value': 'sum'
        }).reset_index()
        
        # Create daily statistics
        daily_stats = combined_df.groupby('date').agg({
            'hdd_value': ['sum', 'mean', 'min', 'max'],
            'cdd_value': ['sum', 'mean', 'min', 'max']
        }).reset_index()
        
        daily_stats.columns = ['date', 'hdd_sum', 'hdd_mean', 'hdd_min', 'hdd_max',
                              'cdd_sum', 'cdd_mean', 'cdd_min', 'cdd_max']
        
        logging.debug(f"Daily aggregate data has {len(daily_aggregate)} records")
        logging.debug("Sample daily aggregate:")
        logging.debug(daily_aggregate.head(3))
        
        save_to_csv(daily_aggregate, "daily_hdd_cdd_aggregated.csv", output_dir)
        save_to_csv(daily_stats, "daily_hdd_cdd_statistics.csv", output_dir)
        
        return combined_df, daily_aggregate, daily_stats
    else:
        logging.debug("Cannot create combined data - missing HDD or CDD data")
        return None, None, None


def process_all_weather_data(from_date=DEFAULT_FROM_DATE, until_date=DEFAULT_UNTIL_DATE, output_dir=DEFAULT_OUTPUT_DIR):
    """Main function to process all weather data"""
    client = VeytAPIClient()
    
    logging.debug("Starting weather data retrieval...")
    
    # Get individual weather data
    weather_data = get_weather_data(client, from_date, until_date, output_dir)
    
    # Create combined and aggregated data
    combined_df, daily_aggregate, daily_stats = create_combined_weather_data(weather_data, output_dir)
    
    logging.debug("\n=== Weather Data Retrieval Complete ===")
    
    return weather_data, combined_df, daily_aggregate, daily_stats


if __name__ == "__main__":
    process_all_weather_data()