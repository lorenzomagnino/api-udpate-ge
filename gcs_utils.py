import os
import pickle
import tempfile
from typing import Optional, Union

import pandas as pd
from google.cloud import storage
from google.cloud.exceptions import NotFound


class GCSManager:
    """Utility class for Google Cloud Storage operations"""
    
    def __init__(self, bucket_name: str = "dashboard_data_ge"):
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
    
    def upload_file(self, local_file_path: str, gcs_path: str) -> bool:
        """Upload a local file to GCS"""
        try:
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded {local_file_path} to gs://{self.bucket_name}/{gcs_path}")
            return True
        except Exception as e:
            print(f"Error uploading {local_file_path} to GCS: {e}")
            return False
    
    def download_file(self, gcs_path: str, local_file_path: str) -> bool:
        """Download a file from GCS to local"""
        try:
            blob = self.bucket.blob(gcs_path)
            blob.download_to_filename(local_file_path)
            print(f"Downloaded gs://{self.bucket_name}/{gcs_path} to {local_file_path}")
            return True
        except NotFound:
            print(f"File gs://{self.bucket_name}/{gcs_path} not found")
            return False
        except Exception as e:
            print(f"Error downloading {gcs_path} from GCS: {e}")
            return False
    
    def upload_dataframe(self, df: pd.DataFrame, gcs_path: str, **kwargs) -> bool:
        """Upload a DataFrame directly to GCS as CSV"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                df.to_csv(tmp_file.name, **kwargs)
                return self.upload_file(tmp_file.name, gcs_path)
        except Exception as e:
            print(f"Error uploading DataFrame to GCS: {e}")
            return False
        finally:
            if 'tmp_file' in locals():
                os.unlink(tmp_file.name)
    
    def download_dataframe(self, gcs_path: str, **kwargs) -> Optional[pd.DataFrame]:
        """Download a CSV file from GCS as DataFrame"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                if self.download_file(gcs_path, tmp_file.name):
                    return pd.read_csv(tmp_file.name, **kwargs)
                return None
        except Exception as e:
            print(f"Error downloading DataFrame from GCS: {e}")
            return None
        finally:
            if 'tmp_file' in locals():
                os.unlink(tmp_file.name)
    
    def upload_pickle(self, data: any, gcs_path: str) -> bool:
        """Upload data as pickle to GCS"""
        try:
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as tmp_file:
                pickle.dump(data, tmp_file)
                return self.upload_file(tmp_file.name, gcs_path)
        except Exception as e:
            print(f"Error uploading pickle to GCS: {e}")
            return False
        finally:
            if 'tmp_file' in locals():
                os.unlink(tmp_file.name)
    
    def download_pickle(self, gcs_path: str) -> Optional[any]:
        """Download pickle data from GCS"""
        try:
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as tmp_file:
                if self.download_file(gcs_path, tmp_file.name):
                    with open(tmp_file.name, 'rb') as f:
                        return pickle.load(f)
                return None
        except Exception as e:
            print(f"Error downloading pickle from GCS: {e}")
            return None
        finally:
            if 'tmp_file' in locals():
                os.unlink(tmp_file.name)
    
    def file_exists(self, gcs_path: str) -> bool:
        """Check if a file exists in GCS"""
        try:
            blob = self.bucket.blob(gcs_path)
            return blob.exists()
        except Exception as e:
            print(f"Error checking file existence: {e}")
            return False
    
    def get_file_modified_time(self, gcs_path: str) -> Optional[str]:
        """Get the last modified time of a file in GCS"""
        try:
            blob = self.bucket.blob(gcs_path)
            blob.reload()
            return blob.time_created.isoformat()
        except Exception as e:
            print(f"Error getting file modified time: {e}")
            return None


# Global GCS manager instance
gcs_manager = GCSManager()
