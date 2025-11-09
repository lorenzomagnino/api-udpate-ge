#!/usr/bin/env python3
"""
Main script to orchestrate all Veyt data retrieval and processing.

This script coordinates the retrieval of:
- Futures and commodities data (via /marketprice endpoint)
- Weather data (via /timeseries endpoint) 
- COT data (via /tsgroup endpoint)
- Auction data (via /tsgroup endpoint)
- Spreads data (via /timeseries endpoint)
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from datetime import date

# Import all processing modules
from veyt_futures import process_all_futures_and_commodities
from veyt_weather import process_all_weather_data
from veyt_cot import process_all_cot_data
from veyt_auctions import process_all_auction_data
from veyt_spreads import process_all_spreads_data
from veyt_options import process_all_options_data
from config import DEFAULT_FROM_DATE, DEFAULT_OUTPUT_DIR

# Always set app base directory, so relative paths always work
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(APP_ROOT)


def setup_logging(log_level='INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('veyt_data_retrieval.log')
        ]
    )


def create_output_directory(output_dir):
    """Create output directory if it doesn't exist"""
    # Check if output_dir is a GCS path (gs://bucket/path)
    if output_dir.startswith("gs://"):
        print(f"Output directory: {output_dir} (GCS)")
        # GCS directories don't need to be created explicitly
        # Files will be uploaded directly to the specified path
    else:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {os.path.abspath(output_dir)}")


def print_summary_report(results, output_dir):
    """Print a summary report of all data retrieved"""
    print("\n" + "="*80)
    print("VEYT DATA RETRIEVAL SUMMARY REPORT")
    print("="*80)
    
    total_files = 0
    
    # Count CSV files in output directory
    if os.path.exists(output_dir):
        csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
        total_files = len(csv_files)
        print(f"Total CSV files created: {total_files}")
        print(f"Output directory: {os.path.abspath(output_dir)}")
    
    print("\nData Categories Retrieved:")
    print("-" * 40)
    
    # Futures and Commodities
    if 'futures' in results:
        eua_raw, eua_processed = results['futures']
        print("✓ Futures & Commodities Data")
        if eua_processed is not None:
            print(f"  - EUA processed data: {len(eua_processed)} trading days")
        print("  - Other futures: UKA, RGGI, WCI")
        print("  - Commodities: TTF, Coal, German Power, Brent Crude")
        print("  - Volatility: 5d, 20d rolling")
    
    # Weather
    if 'weather' in results:
        weather_data, combined_df, daily_aggregate, daily_stats = results['weather']
        print("✓ Weather Data")
        if combined_df is not None:
            print(f"  - Combined HDD/CDD records: {len(combined_df)}")
        if daily_aggregate is not None:
            print(f"  - Daily aggregated records: {len(daily_aggregate)}")
    
    # COT - Fixed to handle dictionary return
    if 'cot' in results:
        cot_results = results['cot']
        ice_cot = cot_results.get('ice_cot_data', {})
        eex_cot = cot_results.get('eex_cot_data', {})
        combined_cot = cot_results.get('combined_cot_data', {})
        print("✓ Commitment of Traders (COT) Data")
        ice_count = len([k for k, v in ice_cot.items() if v is not None]) if ice_cot else 0
        eex_count = len([k for k, v in eex_cot.items() if v is not None]) if eex_cot else 0
        print(f"  - ICE COT categories: {ice_count}")
        print(f"  - EEX COT categories: {eex_count}")
        print(f"  - Combined categories: {len(combined_cot) if combined_cot else 0}")
    
    # Auctions
    if 'auctions' in results:
        all_auction_data, summary_df = results['auctions']
        print("✓ Auction Data")
        total_auction_datasets = sum([len([k for k, v in auction_data.items() if v is not None]) 
                                    for auction_data in all_auction_data.values()])
        print(f"  - Total auction datasets: {total_auction_datasets}")
        print("  - EU, UKA, RGGI, WCI auctions")
    
    # Spreads
    if 'spreads' in results:
        spreads_data, combined_df, daily_aggregate, correlation_df = results['spreads']
        print("✓ Spreads Data")
        available_spreads = len([k for k, v in spreads_data.items() if v is not None]) if spreads_data else 0
        print(f"  - Available spread types: {available_spreads}")
        if combined_df is not None:
            print(f"  - Combined spread records: {len(combined_df)}")
        if correlation_df is not None:
            print(f"  - Correlation pairs analyzed: {len(correlation_df)}")
            
    # Options
    if 'options' in results:
        print("✓ Options Data")
        print("  - Put Options")
        print("  - Call Options")
        
    print("\n" + "="*80)
    print(f"Data retrieval completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


def main():
    """Main function to orchestrate all data retrieval"""
    parser = argparse.ArgumentParser(description='Retrieve all Veyt data')
    parser.add_argument('--from-date', default=DEFAULT_FROM_DATE, 
                        help=f'Start date (default: {DEFAULT_FROM_DATE})')
    # parser.add_argument('--until-date', default=DEFAULT_UNTIL_DATE,
    #                     help=f'End date (default: {DEFAULT_UNTIL_DATE})')
    
    parser.add_argument('--until-date', default=date.today().isoformat(),
                    help=f'End date (default: {date.today().isoformat()})')
    
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level (default: INFO)')
    parser.add_argument('--skip-futures', action='store_true',
                        help='Skip futures and commodities data retrieval')
    parser.add_argument('--skip-weather', action='store_true',
                        help='Skip weather data retrieval')
    parser.add_argument('--skip-cot', action='store_true',
                        help='Skip COT data retrieval')
    parser.add_argument('--skip-auctions', action='store_true',
                        help='Skip auction data retrieval')
    parser.add_argument('--skip-spreads', action='store_true',
                        help='Skip spreads data retrieval')
    parser.add_argument('--skip-options', action='store_true',
                        help='Skip options data retrieval')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Create output directory
    create_output_directory(args.output_dir)
    
    print("Starting comprehensive Veyt data retrieval...")
    print(f"Date range: {args.from_date} to {args.until_date}")
    print(f"Output directory: {args.output_dir}")
    
    results = {}
    
    try:
        # 1. Futures and Commodities Data (/marketprice endpoint)
        if not args.skip_futures:
            print("\n" + "="*60)
            print("STEP 1: FUTURES AND COMMODITIES DATA")
            print("="*60)
            results['futures'] = process_all_futures_and_commodities(
                args.from_date, args.until_date, args.output_dir
            )
        
        # 2. Weather Data (/timeseries endpoint)
        if not args.skip_weather:
            print("\n" + "="*60)
            print("STEP 2: WEATHER DATA")
            print("="*60)
            results['weather'] = process_all_weather_data(
                args.from_date, args.until_date, args.output_dir
            )
        
        # 3. COT Data (/tsgroup endpoint)
        if not args.skip_cot:
            print("\n" + "="*60)
            print("STEP 3: COMMITMENT OF TRADERS (COT) DATA")
            print("="*60)
            results['cot'] = process_all_cot_data(
                args.from_date, args.until_date, args.output_dir
            )
        
        # 4. Auction Data (/tsgroup endpoint)
        if not args.skip_auctions:
            print("\n" + "="*60)
            print("STEP 4: AUCTION DATA")
            print("="*60)
            results['auctions'] = process_all_auction_data(
                args.from_date, args.until_date, args.output_dir
            )
        
        # 5. Spreads Data (/timeseries endpoint)
        if not args.skip_spreads:
            print("\n" + "="*60)
            print("STEP 5: SPREADS DATA")
            print("="*60)
            results['spreads'] = process_all_spreads_data(
                args.from_date, args.until_date, args.output_dir
            )
        # 6. Options Data (/options endpoint)
        if not args.skip_options:
            print("\n" + "="*60)
            print("STEP 6: OPTIONS DATA")
            print("="*60)
            results['options'] = process_all_options_data(
                args.from_date, args.until_date, args.output_dir
            )
        
        # Generate summary report
        print_summary_report(results, args.output_dir)
        
    except Exception as e:
        logging.error(f"Error during data retrieval: {e}")
        print(f"\nError occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()