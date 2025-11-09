import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

try:
    from gcs_utils import gcs_manager
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False


class VeytAPIClient:
    def __init__(
        self,
        client_id: str = "UA72RJH8YQzjaK0CI7x8yIZLpFdyQked",
        client_secret: str = "l46OXzSm95VmWrEGuy-htpZLeOlU5f-jLEQhbMtmDgD-r_SydBXtJdZDVCSIlp86",
        base_url: str = "https://curves.veyt.com",
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = base_url
        self.token_file = Path(".veyt_token_cache.json")
        self._token = None
        self._token_expires_at = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _save_token_to_cache(self, token_data: dict):
        expires_at = datetime.now() + timedelta(
            seconds=token_data.get("expires_in", 259200)
        )

        cache_data = {
            "access_token": token_data["access_token"],
            "expires_at": expires_at.isoformat(),
            "token_type": token_data.get("token_type", "Bearer"),
            "scope": token_data.get("scope", ""),
        }

        with open(self.token_file, "w") as f:
            json.dump(cache_data, f)

        self._token = token_data["access_token"]
        self._token_expires_at = expires_at

    def _load_token_from_cache(self) -> bool:
        if not self.token_file.exists():
            return False

        try:
            with open(self.token_file, "r") as f:
                cache_data = json.load(f)

            expires_at = datetime.fromisoformat(cache_data["expires_at"])

            if expires_at > datetime.now() + timedelta(hours=1):
                self._token = cache_data["access_token"]
                self._token_expires_at = expires_at
                self.logger.info("Using cached token")
                return True
            else:
                self.logger.info("Cached token expired")
                return False
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.warning(f"Error reading token cache: {e}")
            return False

    def _fetch_new_token(self) -> bool:
        url = f"{self.base_url}/api/v1/user/token/"
        params = {"client_id": self.client_id, "client_secret": self.client_secret}

        self.logger.info("Retrieving new bearer token...")

        try:
            response = requests.post(url, params=params)
            response.raise_for_status()

            token_data = response.json()
            self._save_token_to_cache(token_data)
            self.logger.info("New token retrieved and cached")
            return True

        except requests.RequestException as e:
            self.logger.error(f"Authentication failed: {e}")
            return False

    def get_token(self) -> str:
        if self._token and self._token_expires_at:
            if self._token_expires_at > datetime.now() + timedelta(hours=1):
                return self._token

        if self._load_token_from_cache():
            return self._token

        if self._fetch_new_token():
            return self._token

        return None

    def get_market_price(
        self,
        name: str,
        from_trade_date: str = "2021-01-01",
        until_trade_date: str = None,
    ) -> pd.DataFrame:
        if until_trade_date is None:
            until_trade_date = (datetime.today() + timedelta(days=1)).strftime(
                "%Y-%m-%d"
            )

        token = self.get_token()
        if not token:
            self.logger.error("Failed to get valid token")
            return None

        endpoint = f"{self.base_url}/api/v1/marketprice/"
        headers = {"Authorization": f"Bearer {token}"}
        params = {
            "name": name,
            "trade_from": from_trade_date,
            "trade_to": until_trade_date,
        }

        self.logger.info(f"Retrieving market price data for {name}...")

        try:
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()

            self.logger.info("Formatting output to DataFrame...")
            return pd.DataFrame(response.json()["series"])

        except requests.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            return None

    def get_timeseries_price(
        self, name: str, from_date: str = "2021-01-01", until_date: str = None
    ) -> pd.DataFrame:
        if until_date is None:
            until_date = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

        token = self.get_token()
        if not token:
            self.logger.error("Failed to get valid token")
            return None

        endpoint = f"{self.base_url}/api/v1/timeseries/"
        headers = {"Authorization": f"Bearer {token}"}
        params = {"name": name, "data_from": from_date, "data_to": until_date}

        self.logger.info(f"Retrieving timeseries data for {name}...")

        try:
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()

            self.logger.info("Formatting output to DataFrame...")
            return pd.DataFrame(response.json()["series"])

        except requests.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            return None

    def get_tsgroup_data(
        self, name: str, from_date: str = "2021-01-01", until_date: str = None
    ):
        if until_date is None:
            until_date = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

        token = self.get_token()
        if not token:
            self.logger.error("Failed to get valid token")
            return None

        endpoint = f"{self.base_url}/api/v1/tsgroup/"
        headers = {"Authorization": f"Bearer {token}"}
        params = {"name": name, "data_from": from_date, "data_to": until_date}

        self.logger.info(f"Retrieving tsgroup data for {name}...")

        try:
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()

            if "series" not in data:
                self.logger.warning("No series data found in response")
                return None

            series = data["series"]
            times = series.get("times", [])
            values = series.get("values", {})

            if not times:
                self.logger.warning("No time data found")
                return None

            self.logger.info("Processing tsgroup data to DataFrame...")
            rows = []
            for i, timestamp in enumerate(times):
                row = {"times": timestamp}
                for metric, metric_values in values.items():
                    if i < len(metric_values):
                        row[metric] = metric_values[i]
                    else:
                        row[metric] = None
                rows.append(row)

            df = pd.DataFrame(rows)

            if "times" in df.columns:
                df["timestamp"] = pd.to_datetime(df["times"], unit="ms")
                df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")

            return df

        except requests.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error during data processing: {e}")
            return None


def save_to_csv(df: pd.DataFrame, filename: str, output_dir: str = "veyt_data"):
    if df is None or df.empty:
        logging.warning(f"No data to save for {filename}")
        return

    # Check if output_dir is a GCS path (gs://bucket/path)
    if output_dir.startswith("gs://"):
        if not GCS_AVAILABLE:
            logging.error("GCS utils not available. Cannot save to GCS path.")
            return None
        
        # Extract bucket and path from gs://bucket/path format
        path_parts = output_dir[5:].split("/", 1)  # Remove 'gs://' prefix
        if len(path_parts) < 2:
            logging.error(f"Invalid GCS path format: {output_dir}")
            return None
        
        bucket_name = path_parts[0]
        base_path = path_parts[1]
        gcs_path = f"{base_path}/{filename}" if base_path else filename
        
        # Use GCS manager to upload
        gcs_manager_local = gcs_manager
        if gcs_manager_local.bucket_name != bucket_name:
            from gcs_utils import GCSManager
            gcs_manager_local = GCSManager(bucket_name=bucket_name)
        
        success = gcs_manager_local.upload_dataframe(df, gcs_path, index=False)
        if success:
            full_path = f"gs://{bucket_name}/{gcs_path}"
            logging.info(f"Saved data to {full_path}")
            return full_path
        else:
            logging.error(f"Failed to save data to GCS: gs://{bucket_name}/{gcs_path}")
            return None
    else:
        # Local file system
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        logging.info(f"Saved data to {filepath}")
        return filepath
