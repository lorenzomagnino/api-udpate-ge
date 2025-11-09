import pandas as pd
import logging
from veyt_client import VeytAPIClient, save_to_csv
from config import SPREADS, DEFAULT_FROM_DATE, DEFAULT_UNTIL_DATE, DEFAULT_OUTPUT_DIR


def format_spreads_data(df: pd.DataFrame, spread_type: str) -> pd.DataFrame:
    """Format spreads data with timestamp conversions"""
    if df is None or df.empty:
        return None
    
    formatted_df = df.copy()
    
    if 'times' in formatted_df.columns:
        formatted_df['timestamp'] = pd.to_datetime(formatted_df['times'], unit='ms')
        formatted_df['date'] = formatted_df['timestamp'].dt.strftime('%Y-%m-%d')
        formatted_df['datetime'] = formatted_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Reorder columns to put date/datetime first
        cols = ['date', 'datetime', 'times']
        cols.extend([col for col in formatted_df.columns if col not in cols])
        formatted_df = formatted_df[cols]
    
    # Rename values column to be more descriptive
    if 'values' in formatted_df.columns:
        formatted_df = formatted_df.rename(columns={
            'values': f'{spread_type.lower()}_value'
        })
    
    return formatted_df


def get_spreads_data(client, from_date=DEFAULT_FROM_DATE, until_date=DEFAULT_UNTIL_DATE, output_dir=DEFAULT_OUTPUT_DIR):
    """Get and process spreads data"""
    print("\n=== Retrieving Spreads Data ===")
    
    spreads_data = {}
    
    for spread_type, curve_id in SPREADS.items():
        print(f"\nGetting {spread_type} spread data...")
        df = client.get_timeseries_price(curve_id, from_date=from_date, until_date=until_date)
        
        if df is not None and not df.empty:
            formatted_df = format_spreads_data(df, spread_type)
            print(f"Retrieved {len(formatted_df)} {spread_type} records")
            print(f"Columns: {list(formatted_df.columns)}")
            
            if 'date' in formatted_df.columns:
                print(f"Date range: {formatted_df['date'].min()} to {formatted_df['date'].max()}")
            
            print("Sample data:")
            print(formatted_df.head(3))
            
            filename = f"{spread_type}_spread_data.csv"
            save_to_csv(formatted_df, filename, output_dir)
            spreads_data[spread_type] = formatted_df
        else:
            print(f"No {spread_type} spread data retrieved")
            spreads_data[spread_type] = None
    
    return spreads_data


def create_combined_spreads_data(spreads_data, output_dir=DEFAULT_OUTPUT_DIR):
    """Combine all spreads data into a single dataset"""
    print("\n=== Creating Combined Spreads Data ===")
    
    # Filter out None values
    available_spreads = {k: v for k, v in spreads_data.items() if v is not None and not v.empty}
    
    if len(available_spreads) < 2:
        print("Not enough spread data to create meaningful combinations")
        return None
    
    # Start with the first spread and merge others
    spread_names = list(available_spreads.keys())
    combined_df = available_spreads[spread_names[0]][['date', 'datetime', f'{spread_names[0]}_value']].copy()
    
    for spread_name in spread_names[1:]:
        spread_df = available_spreads[spread_name]
        merge_cols = ['date', 'datetime', f'{spread_name}_value']
        
        combined_df = pd.merge(
            combined_df,
            spread_df[merge_cols],
            on=['date', 'datetime'],
            how='outer'
        )
    
    combined_df = combined_df.sort_values(['date', 'datetime'])
    
    print(f"Combined spreads data has {len(combined_df)} records")
    print(f"Combined {len(available_spreads)} spread types: {', '.join(available_spreads.keys())}")
    print("Sample combined data:")
    print(combined_df.head(3))
    
    save_to_csv(combined_df, "combined_spreads_data.csv", output_dir)
    
    # Create daily aggregated data
    print("Creating daily aggregated spreads data...")
    value_cols = [col for col in combined_df.columns if col.endswith('_value')]
    
    daily_aggregate = combined_df.groupby('date')[value_cols].agg(['mean', 'min', 'max']).reset_index()
    
    # Flatten column names
    daily_aggregate.columns = ['date'] + [f"{col[0]}_{col[1]}" for col in daily_aggregate.columns[1:]]
    
    print(f"Daily aggregate spreads data has {len(daily_aggregate)} records")
    save_to_csv(daily_aggregate, "daily_spreads_aggregated.csv", output_dir)
    
    return combined_df, daily_aggregate


def analyze_spreads_relationships(spreads_data, output_dir=DEFAULT_OUTPUT_DIR):
    """Analyze relationships between different spreads"""
    print("\n=== Analyzing Spreads Relationships ===")
    
    available_spreads = {k: v for k, v in spreads_data.items() if v is not None and not v.empty}
    
    if len(available_spreads) < 2:
        print("Not enough spread data for relationship analysis")
        return None
    
    # Create correlation analysis
    spread_names = list(available_spreads.keys())
    correlation_data = []
    
    for i, spread1 in enumerate(spread_names):
        for j, spread2 in enumerate(spread_names):
            if i < j:  # Avoid duplicate pairs
                df1 = available_spreads[spread1]
                df2 = available_spreads[spread2]
                
                # Merge on date for correlation
                merged = pd.merge(
                    df1[['date', f'{spread1}_value']],
                    df2[['date', f'{spread2}_value']],
                    on='date',
                    how='inner'
                )
                
                if len(merged) > 10:  # Need enough data points
                    correlation = merged[f'{spread1}_value'].corr(merged[f'{spread2}_value'])
                    correlation_data.append({
                        'spread1': spread1,
                        'spread2': spread2,
                        'correlation': correlation,
                        'data_points': len(merged)
                    })
    
    if correlation_data:
        correlation_df = pd.DataFrame(correlation_data)
        print("Spread correlations:")
        print(correlation_df.to_string(index=False))
        save_to_csv(correlation_df, "spreads_correlations.csv", output_dir)
        return correlation_df
    
    return None


def process_all_spreads_data(from_date=DEFAULT_FROM_DATE, until_date=DEFAULT_UNTIL_DATE, output_dir=DEFAULT_OUTPUT_DIR):
    """Main function to process all spreads data"""
    client = VeytAPIClient()
    
    print("Starting spreads data retrieval...")
    
    # Get individual spreads data
    spreads_data = get_spreads_data(client, from_date, until_date, output_dir)
    
    # Create combined data
    combined_results = create_combined_spreads_data(spreads_data, output_dir)
    combined_df, daily_aggregate = combined_results if combined_results else (None, None)
    
    # Analyze relationships
    correlation_df = analyze_spreads_relationships(spreads_data, output_dir)
    
    print("\n=== Spreads Data Retrieval Complete ===")
    
    # Print summary
    available_count = len([k for k, v in spreads_data.items() if v is not None])
    total_records = sum([len(v) for v in spreads_data.values() if v is not None])
    
    print(f"Spreads types retrieved: {available_count}")
    print(f"Total spread records: {total_records}")
    
    if combined_df is not None:
        print(f"Combined dataset records: {len(combined_df)}")
        
    if correlation_df is not None:
        print(f"Correlation analysis completed for {len(correlation_df)} spread pairs")
    
    return spreads_data, combined_df, daily_aggregate, correlation_df


if __name__ == "__main__":
    process_all_spreads_data()