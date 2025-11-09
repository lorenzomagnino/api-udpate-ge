from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

from helpers import DATA_CONFIG


class DataProcessor:
    def __init__(self):
        self.base_path = DATA_CONFIG.get('futures_path', 'gs://dashboard_data_ge/veyt_data_new/eua_futures_front_and_benchmark.csv')
        self.target = DATA_CONFIG['target_column']
        self.data = None
        self.log_transformed = False
        
        self.additional_paths = {
            'ttf': 'gs://dashboard_data_ge/veyt_data_new/ttf_front_month.csv',
            'german_power': 'gs://dashboard_data_ge/veyt_data_new/german_power_front_month.csv',
            'fuel_switch': 'gs://dashboard_data_ge/veyt_data_new/fuel_switch_spread_data.csv',
            'auctions': 'gs://dashboard_data_ge/veyt_data_new/eu_auctions_combined.csv',
            'cot': 'gs://dashboard_data_ge/veyt_data_new/cot_summed_by_type.csv',
            'ttf_cot': 'gs://dashboard_data_ge/veyt_data_new/ttf_cot_summed_by_type.csv',
            'coal': 'gs://dashboard_data_ge/veyt_data_new/coal_front_month.csv',
            'clean_spark': 'gs://dashboard_data_ge/veyt_data_new/clean_spark_spread_data.csv',
            'clean_dark': 'gs://dashboard_data_ge/veyt_data_new/clean_dark_spread_data.csv',
            'hdd_cdd': DATA_CONFIG.get('hdd_cdd_path', 'gs://dashboard_data_ge/veyt_data_new/daily_hdd_cdd_statistics.csv'),
            'call_options': DATA_CONFIG.get('call_option_path', 'gs://dashboard_data_ge/veyt_data_new/opt_call_options_data.csv'),
            'put_options': DATA_CONFIG.get('put_option_path', 'gs://dashboard_data_ge/veyt_data_new/opt_put_options_data.csv'),
            'vol_5d': 'gs://dashboard_data_ge/veyt_data_new/volatility_5d_benchmark.csv',
            'vol_20d': 'gs://dashboard_data_ge/veyt_data_new/volatility_20d_benchmark.csv',
            'brent_crude': 'gs://dashboard_data_ge/veyt_data_new/brent_crude_front_month.csv',
        }
        
        self.columns_to_drop = DATA_CONFIG.get('columns_to_drop', [])

    def load_and_process(self) -> pd.DataFrame:
        self._load_all_datasets()
        self._process_dates_and_index()
        self._handle_missing_values()
        self._process_options_data()
        self._process_cot_data()
        self._process_ttf_cot_data()
        self._clean_target()
        self._filter_recent_data()
        self._drop_columns()
        self._rename_columns()
        
        print(f"Final dataset shape: {self.data.shape}")
        print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
        return self.data

    def _load_all_datasets(self):
        self.data = pd.read_csv(self.base_path)
        self.data = self.data.rename(columns={'trade_date': 'Date'})
        self.data = self._process_futures_dates(self.data)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        datasets_config = [
            ('ttf', 'trade_date', None, ['TTF_front_maturity_date','TTF_front_modified']),
            ('brent_crude', 'trade_date', None, ['Brent Crude_front_maturity_date','Brent Crude_front_modified','Brent Crude_benchmark_maturity_date','Brent Crude_benchmark_modified']),
            ('german_power', 'trade_date', None,['German Power_front_maturity_date','German Power_front_modified','German Power_front2_maturity_date','German Power_front2_modified']),
            ('coal', 'trade_date', None,['Coal_front_maturity_date', 'Coal_front_modified', 'Coal_front2_maturity_date', 'Coal_front2_modified']),
            ('clean_spark', 'date', ['clean_spark_value']),
            ('clean_dark', 'date', ['clean_dark_value']),
            ('fuel_switch', 'date', ['fuel_switch_value']),
            ('vol_5d', 'trade_date', ['5d_benchmark_settlement']),
            ('vol_20d', 'trade_date', ['20d_benchmark_settlement']),
            ('auctions', 'date', None, ['times', 'vol_sold', 'timestamp','auction_type']),
            ('cot', 'Date', None),
            ('ttf_cot', 'Date', None),
            ('hdd_cdd', 'date', None),
        ]
        
        for config in datasets_config:
            name = config[0]
            date_col = config[1]
            keep_cols = config[2] if len(config) > 2 else None
            drop_cols = config[3] if len(config) > 3 else None
            
            try:
                df = pd.read_csv(self.additional_paths[name])
                df = df.rename(columns={date_col: 'Date'})
                df['Date'] = pd.to_datetime(df['Date'])
                
                if drop_cols:
                    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
                
                if keep_cols:
                    df = df[['Date'] + keep_cols]
                
                if name == 'hdd_cdd':
                    df = df.rename(columns={col: f'weather_{col}' for col in df.columns if col != 'Date'})
                
                self.data = pd.merge(self.data, df, on='Date', how='left')
                
            except FileNotFoundError:
                print(f"File not found: {self.additional_paths[name]}")
                continue

    def _process_futures_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Date'] = pd.to_datetime(df['Date'])
        
        for col in df.columns:
            if 'maturity_date' in col:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                base_name = col.replace('_maturity_date', '')
                df[f'{base_name}_days_to_maturity'] = (df[col] - df['Date']).dt.days
                df[f'{base_name}_maturity_month'] = df[col].dt.month
                # df[f'{base_name}_years_to_maturity'] = df[f'{base_name}_days_to_maturity'] / 365.25
                df = df.drop(columns=[col])
                df = df.drop(columns=['EUA_front_modified','EUA_benchmark_mofified','EUA_benchmark_maturity_month'], errors='ignore')
        
        return df

    def _process_dates_and_index(self):
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)
        self.data.sort_index(inplace=True)
        
        for col in self.data.select_dtypes(include=['object']).columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

    def _handle_missing_values(self):
        weather_cols = [col for col in self.data.columns if col.startswith('weather_')]
        if weather_cols:
            first_valid_idx = None
            for col in weather_cols:
                idx = self.data[col].first_valid_index()
                if idx and (not first_valid_idx or idx < first_valid_idx):
                    first_valid_idx = idx
            
            if first_valid_idx:
                for col in weather_cols:
                    self.data.loc[first_valid_idx:, col] = (
                        self.data.loc[first_valid_idx:, col].interpolate(method='linear')
                    )
        
        other_cols = [col for col in self.data.columns if not col.startswith('weather_')]
        for col in other_cols:
            self.data[col] = self.data[col].ffill()
        
        print("Applied missing value handling")

    def _process_options_data(self):
        options_cols = {}
        
        for option_type in ['put', 'call']:
            path = self.additional_paths[f'{option_type}_options']
            try:
                df = pd.read_csv(path)
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df['maturity_date'] = pd.to_datetime(df['maturity_date'])
                df['maturity_date'] = df['maturity_date'] + pd.offsets.MonthEnd(0)
                
                vol_data = self._extract_option_volatilities(df, option_type)
                options_cols.update(vol_data)
                
            except FileNotFoundError:
                print(f"File not found: {path}")
                continue
        
        if options_cols:
            options_df = pd.DataFrame(options_cols)
            options_df.index = pd.to_datetime(options_df.index)
            self.data = pd.merge(self.data, options_df, left_index=True, right_index=True, how='left')
            
            options_columns = [col for col in self.data.columns if col.startswith('op_')]
            for col in options_columns:
                self.data[col] = self.data[col].ffill()
            
            print(f"Added {len(options_cols)} options volatility columns")

    def _extract_option_volatilities(self, df: pd.DataFrame, option_type: str) -> Dict:
        vol_data = {
            f'op_{option_type}_atm_near_vol': {},
            f'op_{option_type}_atm_far_vol': {},
            f'op_{option_type}_bm_atm_near_vol': {},
            f'op_{option_type}_bm_atm_far_vol': {}
        }
        
        for trade_date in df['trade_date'].unique():
            if trade_date not in self.data.index:
                continue
                
            eua_front_settl = None
            eua_benchmark_settl = None
            for col in ['EUA_front_settlement']:
                if col in self.data.columns:
                    val = self.data.loc[trade_date, col]
                    eua_front_settl = val.iloc[0] if isinstance(val, pd.Series) else val
                    break
                    
            for col in ['EUA_benchmark_settlement']:
                if col in self.data.columns:
                    val = self.data.loc[trade_date, col]
                    eua_benchmark_settl = val.iloc[0] if isinstance(val, pd.Series) else val
                    break
            
            if pd.isna(eua_front_settl) or pd.isna(eua_benchmark_settl):
                continue
            
            day_options = df[df['trade_date'] == trade_date].copy()
            
            # Find nearest maturity
            nearest_maturity = day_options['maturity_date'].min()
            
            # Find year-end maturity
            year_end = pd.Timestamp(f"{trade_date.year}-12-31")
            year_end_options = day_options[day_options['maturity_date'] <= year_end]
            if year_end_options.empty and trade_date.month == 12:
                year_end = pd.Timestamp(f"{trade_date.year + 1}-12-31")
                year_end_options = day_options[day_options['maturity_date'] <= year_end]
            
            if not year_end_options.empty:
                far_maturity = year_end_options['maturity_date'].max()
            else:
                far_maturity = None
            
            # Extract volatilities for EUA_front_settlement
            # 1. ATM near vol (nearest maturity, closest strike to front settlement)
            near_options = day_options[day_options['maturity_date'] == nearest_maturity]
            if not near_options.empty:
                closest_strike_idx = (near_options['strike_price'] - eua_front_settl).abs().idxmin()
                vol_data[f'op_{option_type}_atm_near_vol'][trade_date] = near_options.loc[closest_strike_idx, 'option_volatility']
            
            # 2. ATM far vol (year-end maturity, closest strike to front settlement)
            if far_maturity:
                far_options = day_options[day_options['maturity_date'] == far_maturity]
                if not far_options.empty:
                    closest_strike_idx = (far_options['strike_price'] - eua_front_settl).abs().idxmin()
                    vol_data[f'op_{option_type}_atm_far_vol'][trade_date] = far_options.loc[closest_strike_idx, 'option_volatility']
            
            # Extract volatilities for EUA_benchmark_settlement
            # 3. BM ATM near vol (nearest maturity, closest strike to benchmark settlement)
            if not near_options.empty:
                closest_strike_idx = (near_options['strike_price'] - eua_benchmark_settl).abs().idxmin()
                vol_data[f'op_{option_type}_bm_atm_near_vol'][trade_date] = near_options.loc[closest_strike_idx, 'option_volatility']
            
            # 4. BM ATM far vol (year-end maturity, closest strike to benchmark settlement)
            if far_maturity:
                far_options = day_options[day_options['maturity_date'] == far_maturity]
                if not far_options.empty:
                    closest_strike_idx = (far_options['strike_price'] - eua_benchmark_settl).abs().idxmin()
                    vol_data[f'op_{option_type}_bm_atm_far_vol'][trade_date] = far_options.loc[closest_strike_idx, 'option_volatility']
        
        return vol_data

    def _clean_target(self):
        if self.data[self.target].isnull().sum() > 0:
            self.data[self.target] = self.data[self.target].interpolate(method='linear')
            print(f"Interpolated missing values in target: {self.target}")
        
        self.data = self.data.dropna(subset=[self.target])
        
        outliers = (np.abs(stats.zscore(self.data[self.target])) > 3).sum()
        print(f"Outliers in target: {outliers}")
        
        if stats.skew(self.data[self.target]) > 0.5 and self.data[self.target].min() > 0:
            self.data[f"{self.target}_log"] = np.log1p(self.data[self.target])
            self.target = f"{self.target}_log"
            self.log_transformed = True
            print("Applied log transformation to target")

    def _filter_recent_data(self):
        self.data = self.data[(self.data.index >= '2023-01-01') ] # & (self.data.index <= '2025-06-27')

    def _drop_columns(self):
        if self.columns_to_drop:
            existing_cols = [col for col in self.columns_to_drop if col in self.data.columns]
            if existing_cols:
                self.data = self.data.drop(columns=existing_cols)
                print(f"Dropped columns: {existing_cols}")

    def _rename_columns(self):
        rename_map = {
            '5d_benchmark_settlement': 'EUA_benchmark_5d_vol',
            '20d_benchmark_settlement': 'EUA_benchmark_20d_vol',
            'bid_vol_submitted': 'Auction_Total_Allowances_Bid',
            'bidders': 'Auction_bidders',
            'bids': 'Auction_bids',
            'cover_ratio': 'Auction_cover_ratio',
            'max_bid': 'Auction_max_bid',
            'min_bid': 'Auction_min_bid',
            'successful_bidders': 'Auction_successful_bidders',
            'successful_bids': 'Auction_successful_bids',
            'unsuccessful_bidders': 'Auction_unsuccessful_bidders',
            'vol_offered': 'Auction_vol_offered'
        }
        
        column_mapping = {}
        for col in self.data.columns:
            if col.startswith('Brent Crude'):
                column_mapping[col] = col.replace('Brent Crude', 'Brent_Crude')
            elif col.startswith('German Power'):
                column_mapping[col] = col.replace('German Power', 'German_Power')
            elif col in rename_map:
                column_mapping[col] = rename_map[col]
        
        if column_mapping:
            self.data.rename(columns=column_mapping, inplace=True)
            print(f"Renamed {len(column_mapping)} columns")
        
    def _process_cot_data(self):
        groups = ['OFI', 'CU', 'IFCI', 'OWCO', 'IF']
        
        cot_rename_map = {}
        long_cols = []
        short_cols = []
        
        for group in groups:
            long_col = f'{group}_Positions_long_total'
            short_col = f'{group}_Positions_short_total'
            
            if long_col in self.data.columns:
                cot_rename_map[long_col] = f'COT_{long_col}'
                long_cols.append(f'COT_{long_col}')
            if short_col in self.data.columns:
                cot_rename_map[short_col] = f'COT_{short_col}'
                short_cols.append(f'COT_{short_col}')
        
        self.data.rename(columns=cot_rename_map, inplace=True)
        
        if long_cols:
            self.data['COT_Positions_long_total'] = self.data[long_cols].sum(axis=1)
        if short_cols:
            self.data['COT_Positions_short_total'] = self.data[short_cols].sum(axis=1)
        
        cols_to_drop = [col for col in self.data.columns if any(x in col for x in ['OFI_Positions', 'CU_Positions', 'IFCI_Positions', 'OWCO_Positions', 'IF_Positions']) and not col.startswith('COT_') and not col.endswith('_long_total') and not col.endswith('_short_total')]
        self.data.drop(columns=cols_to_drop, inplace=True)
        
        print(f"Renamed {len(cot_rename_map)} individual COT columns and created consolidated totals")
        
        self.data['wednesday'] = (self.data.index.dayofweek == 2).astype(int)
    
    def _process_ttf_cot_data(self):
        groups = ['TTF_OFI', 'TTF_CU', 'TTF_IFCI', 'TTF_OWCO', 'TTF_IF']
        
        cot_rename_map = {}
        long_cols = []
        short_cols = []
        
        for group in groups:
            long_col = f'{group}_Positions_long_total'
            short_col = f'{group}_Positions_short_total'
            
            if long_col in self.data.columns:
                cot_rename_map[long_col] = f'COT_{long_col}'
                long_cols.append(f'COT_{long_col}')
            if short_col in self.data.columns:
                cot_rename_map[short_col] = f'COT_{short_col}'
                short_cols.append(f'COT_{short_col}')
        
        self.data.rename(columns=cot_rename_map, inplace=True)
        
        if long_cols:
            self.data['COT_TTF_Positions_long_total'] = self.data[long_cols].sum(axis=1)
        if short_cols:
            self.data['COT_TTF_Positions_short_total'] = self.data[short_cols].sum(axis=1)
        
        cols_to_drop = [col for col in self.data.columns if any(x in col for x in ['TTF_OFI_Positions', 'TTF_CU_Positions', 'TTF_IFCI_Positions', 'TTF_OWCO_Positions', 'TTF_IF_Positions']) and not col.startswith('COT_TTF_') and not col.endswith('_long_total') and not col.endswith('_short_total')]
        self.data.drop(columns=cols_to_drop, inplace=True)
        
        print(f"Renamed {len(cot_rename_map)} individual TTF COT columns and created consolidated totals")
    
    def get_summary(self) -> Dict:
        if self.data is None:
            return {}
        
        return {
            'shape': self.data.shape,
            'date_range': (self.data.index.min(), self.data.index.max()),
            'target_stats': self.data[self.target].describe().to_dict(),
            'missing_values': {k: v for k, v in self.data.isnull().sum().to_dict().items() if v > 0},
            'log_transformed': self.log_transformed,
            'columns': list(self.data.columns)
        }