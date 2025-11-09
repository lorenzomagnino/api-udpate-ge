import os
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

# from config import OUTPUT_CONFIG
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

DATA_CONFIG = {
    'futures_path': 'gs://dashboard_data_ge/veyt_data_new/eua_futures_front_and_benchmark.csv',
    'hdd_cdd_path': 'gs://dashboard_data_ge/veyt_data_new/daily_hdd_cdd_statistics.csv',
    'put_option_path': 'gs://dashboard_data_ge/veyt_data_new/opt_put_options_data.csv',
    'call_option_path': 'gs://dashboard_data_ge/veyt_data_new/opt_call_options_data.csv',
    'target_column': 'EUA_benchmark_settlement',
    'test_size': 0.2,
    'random_seed': 42,
    'columns_to_drop': [
        'EUA_front_modified',
        'EUA_benchmark_modified',
        #Price Variables        
        # 'EUA_benchmark_open',
        # 'EUA_benchmark_high',
        # 'EUA_benchmark_low',
        'EUA_front_open',
        'EUA_front_high',
        'EUA_front_low',
        'median_bid',
        'mean_bid',
        'clearing_price',
        'COT_TTF_OWCO_Positions_long_total',
        'COT_TTF_OWCO_Positions_short_total',
    ]
}

OUTPUT_CONFIG = {
    # 'model_save_path': 'trained_models/best_model.pkl',
    'visualization_path': 'visualizations/',
    'results_path': 'results/',
    'testing_path': 'gs://dashboard_data_ge/a_testing_results/',
}

# Random seed
np.random.seed(DATA_CONFIG['random_seed'])



def ensure_directory_exists(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
        
def log_message(message: str, level: str = 'info') -> None:
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted_message = f"[{timestamp}] {level.upper()}: {message}"
    print(formatted_message)
    log_path = os.path.join(OUTPUT_CONFIG['results_path'], 'logs.txt')
    directory = os.path.dirname(log_path)
    if directory:
        ensure_directory_exists(directory)
    with open(log_path, 'a') as f:
        f.write(formatted_message + '\n')
        

class Visualizer:
    def __init__(self, output_path: str = None):
        self.output_path = output_path or OUTPUT_CONFIG['visualization_path']

    def plot_time_series(self, data: pd.DataFrame, columns: List[str], 
                        title: str = "Time Series Plot", 
                        save_path: str = None,
                        interactive: bool = False) -> None:
        if interactive:
            fig = go.Figure()
            
            for col in columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[col],
                    name=col,
                    mode='lines'
                ))
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Value",
                hovermode='x unified'
            )
            if save_path:
                fig.write_html(save_path)
            else:
                fig.show()
        else:
            plt.figure(figsize=(14, 5))
            for col in columns:
                plt.plot(data.index, data[col], label=col, linewidth=2)
            plt.title(title, fontsize=16)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Value", fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
    def plot_distribution(self, data: pd.DataFrame, column: str, 
                         title: str = None, bins: int = 30,
                         save_path: str = None) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        sns.histplot(data[column], bins=bins, kde=True, ax=ax1)
        ax1.set_title(title or f'Distribution of {column}', fontsize=14)
        ax1.set_xlabel(column, fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        from scipy import stats
        stats.probplot(data[column].dropna(), dist="norm", plot=ax2)
        ax2.set_title(f'Q-Q Plot - {column}', fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()