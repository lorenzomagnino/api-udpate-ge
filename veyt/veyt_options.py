import pandas as pd
import requests
from veyt_client import VeytAPIClient, save_to_csv
from config import OPTIONS, DEFAULT_FROM_DATE, DEFAULT_UNTIL_DATE, DEFAULT_OUTPUT_DIR


def get_options_data(client, option_type, from_date=DEFAULT_FROM_DATE, until_date=DEFAULT_UNTIL_DATE, output_dir=DEFAULT_OUTPUT_DIR):
    curve_name = OPTIONS[option_type]
    
    token = client.get_token()
    if not token:
        print(f"Failed to get valid token for {option_type}")
        return None
    
    endpoint = f'{client.base_url}/api/v1/option/'
    headers = {'Authorization': f'Bearer {token}'}
    params = {
        'name': curve_name,
        'trade_from': from_date,
        'trade_to': until_date
    }
    
    print(f'Retrieving {option_type} options data...')
    
    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data['series'])
        
        if df.empty:
            print(f"No {option_type} data retrieved")
            return None
        
        desired_columns = ['trade_date', 'maturity_date', 'strike_price', 'settlement', 'volume', 'open_interest', 'option_volatility', 'delta_factor']
        df = df[desired_columns]
        
        print(f"Retrieved {len(df)} {option_type} records")
        
        filename = f"{option_type}_options_data.csv"
        save_to_csv(df, filename, output_dir)
        
        return df
        
    except requests.RequestException as e:
        print(f'Request failed for {option_type}: {e}')
        return None


def process_all_options_data(from_date=DEFAULT_FROM_DATE, until_date=DEFAULT_UNTIL_DATE, output_dir=DEFAULT_OUTPUT_DIR):
    client = VeytAPIClient()
    
    print("Starting options data retrieval...")
    
    put_data = get_options_data(client, 'opt_put', from_date, until_date, output_dir)
    call_data = get_options_data(client, 'opt_call', from_date, until_date, output_dir)
    
    print("Options data retrieval complete")
    
    return put_data, call_data


if __name__ == "__main__":
    process_all_options_data()