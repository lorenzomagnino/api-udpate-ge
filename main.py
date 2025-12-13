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
from veyt.veyt_futures import process_all_futures_and_commodities
from veyt.veyt_weather import process_all_weather_data
from veyt.veyt_cot import process_all_cot_data
from veyt.veyt_auctions import process_all_auction_data
from veyt.veyt_spreads import process_all_spreads_data
from veyt.veyt_options import process_all_options_data
from veyt.config import DEFAULT_FROM_DATE, DEFAULT_OUTPUT_DIR


def setup_logging(log_level='INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('veyt/veyt_data_retrieval.log')
        ]
    )


def create_output_directory(output_dir):
    """Create output directory if it doesn't exist"""
    os.makedirs(output_dir, exist_ok=True)
    logging.debug(f"Output directory: {os.path.abspath(output_dir)}")


def print_summary_report(results, output_dir):
    """logging.debug a summary report of all data retrieved"""
    logging.debug("\n" + "="*80)
    logging.debug("VEYT DATA RETRIEVAL SUMMARY REPORT")
    logging.debug("="*80)
    
    total_files = 0
    
    # Count CSV files in output directory
    if os.path.exists(output_dir):
        csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
        total_files = len(csv_files)
        logging.debug(f"Total CSV files created: {total_files}")
        logging.debug(f"Output directory: {os.path.abspath(output_dir)}")
    
    logging.debug("\nData Categories Retrieved:")
    logging.debug("-" * 40)
    
    # Futures and Commodities
    if 'futures' in results:
        eua_raw, eua_processed = results['futures']
        logging.debug("✓ Futures & Commodities Data")
        if eua_processed is not None:
            logging.debug(f"  - EUA processed data: {len(eua_processed)} trading days")
        logging.debug("  - Other futures: UKA, RGGI, WCI")
        logging.debug("  - Commodities: TTF, Coal, German Power, Brent Crude")
        logging.debug("  - Volatility: 5d, 20d rolling")
    
    # Weather
    if 'weather' in results:
        weather_data, combined_df, daily_aggregate, daily_stats = results['weather']
        logging.debug("✓ Weather Data")
        if combined_df is not None:
            logging.debug(f"  - Combined HDD/CDD records: {len(combined_df)}")
        if daily_aggregate is not None:
            logging.debug(f"  - Daily aggregated records: {len(daily_aggregate)}")
    
    # COT - Fixed to handle dictionary return
    if 'cot' in results:
        cot_results = results['cot']
        ice_cot = cot_results.get('ice_cot_data', {})
        eex_cot = cot_results.get('eex_cot_data', {})
        combined_cot = cot_results.get('combined_cot_data', {})
        logging.debug("✓ Commitment of Traders (COT) Data")
        ice_count = len([k for k, v in ice_cot.items() if v is not None]) if ice_cot else 0
        eex_count = len([k for k, v in eex_cot.items() if v is not None]) if eex_cot else 0
        logging.debug(f"  - ICE COT categories: {ice_count}")
        logging.debug(f"  - EEX COT categories: {eex_count}")
        logging.debug(f"  - Combined categories: {len(combined_cot) if combined_cot else 0}")
    
    # Auctions
    if 'auctions' in results:
        all_auction_data, summary_df = results['auctions']
        logging.debug("✓ Auction Data")
        total_auction_datasets = sum([len([k for k, v in auction_data.items() if v is not None]) 
                                    for auction_data in all_auction_data.values()])
        logging.debug(f"  - Total auction datasets: {total_auction_datasets}")
        logging.debug("  - EU, UKA, RGGI, WCI auctions")
    
    # Spreads
    if 'spreads' in results:
        spreads_data, combined_df, daily_aggregate, correlation_df = results['spreads']
        logging.debug("✓ Spreads Data")
        available_spreads = len([k for k, v in spreads_data.items() if v is not None]) if spreads_data else 0
        logging.debug(f"  - Available spread types: {available_spreads}")
        if combined_df is not None:
            logging.debug(f"  - Combined spread records: {len(combined_df)}")
        if correlation_df is not None:
            logging.debug(f"  - Correlation pairs analyzed: {len(correlation_df)}")
            
    # Options
    if 'options' in results:
        logging.debug("✓ Options Data")
        logging.debug("  - Put Options")
        logging.debug("  - Call Options")
        
    logging.debug("\n" + "="*80)
    logging.debug(f"Data retrieval completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.debug("="*80)


def main():
    """Main function to orchestrate all data retrieval"""
    parser = argparse.ArgumentParser(description='Retrieve all Veyt data')
    parser.add_argument('--from-date', default=DEFAULT_FROM_DATE, 
                        help=f'Start date (default: {DEFAULT_FROM_DATE})')
    
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
    
    logging.info("Starting comprehensive Veyt data retrieval...")
    logging.info(f"Date range: {args.from_date} to {args.until_date}")
    logging.info(f"Output directory: {args.output_dir}")
    
    results = {}
    
    try:
        # 1. Futures and Commodities Data (/marketprice endpoint)
        if not args.skip_futures:
            logging.info("STEP 1: FUTURES AND COMMODITIES DATA")
            logging.info("="*60)
            results['futures'] = process_all_futures_and_commodities(
                args.from_date, args.until_date, args.output_dir
            )
        
        # 2. Weather Data (/timeseries endpoint)
        if not args.skip_weather:
            logging.info("STEP 2: WEATHER DATA")
            logging.info("="*60)
            results['weather'] = process_all_weather_data(
                args.from_date, args.until_date, args.output_dir
            )
        
        # 3. COT Data (/tsgroup endpoint)
        if not args.skip_cot:
            logging.info("STEP 3: COMMITMENT OF TRADERS (COT) DATA")
            logging.info("="*60)
            results['cot'] = process_all_cot_data(
                args.from_date, args.until_date, args.output_dir
            )
        
        # 4. Auction Data (/tsgroup endpoint)
        if not args.skip_auctions:
            logging.info("STEP 4: AUCTION DATA")
            logging.info("="*60)
            results['auctions'] = process_all_auction_data(
                args.from_date, args.until_date, args.output_dir
            )
        
        # 5. Spreads Data (/timeseries endpoint)
        if not args.skip_spreads:
            logging.info("STEP 5: SPREADS DATA")
            logging.info("="*60)
            results['spreads'] = process_all_spreads_data(
                args.from_date, args.until_date, args.output_dir
            )
        # 6. Options Data (/options endpoint)
        if not args.skip_options:
            logging.info("STEP 6: OPTIONS DATA")
            logging.info("="*60)
            results['options'] = process_all_options_data(
                args.from_date, args.until_date, args.output_dir
            )
        
        # Generate summary report
        print_summary_report(results, args.output_dir)
        
    except Exception as e:
        logging.error(f"Error during data retrieval: {e}")
        logging.debug(f"\nError occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()