# -*- coding: utf-8 -*-
"""
S3 Manager Module

Module for S3 communications and local folder management..

Author: Noussair Lazrak
"""

import os
import json
from datetime import datetime, timedelta
from typing import Optional, Union, List, Dict, Any
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, BotoCoreError
import pandas as pd


class S3Manager:
    """
    S3 operations and local folder management.
    
    This class provides methods for:
    - S3 file uploads and downloads
    - S3 metadata retrieval (last modified, file size, etc.)
    - S3 connectivity checks
    - Local folder creation and management
    
    Parameters
    ----------
    bucket_name : str, optional
        Default S3 bucket name (without s3:// prefix)
    region_name : str, optional
        AWS region name (default: 'us-east-1')
    profile_name : str, optional
        AWS profile name for credentials
        
    Examples
    --------
    >>> manager = S3Manager(bucket_name='my-bucket')
    >>> manager.upload_file('local/file.csv', 'remote/path/')
    >>> df = manager.download_csv('remote/path/file.csv')
    """
    
    def __init__(
        self,
        bucket_name: Optional[str] = None,
        region_name: str = 'us-east-1',
        profile_name: Optional[str] = None
    ):
        self.bucket_name = bucket_name.replace('s3://', '') if bucket_name else None
        self.region_name = region_name
        self.profile_name = profile_name
        self._client = None
        self._resource = None
        
    @property
    def client(self) -> boto3.client:
        """Lazy-loaded S3 client."""
        if self._client is None:
            session_kwargs = {'region_name': self.region_name}
            if self.profile_name:
                session_kwargs['profile_name'] = self.profile_name
            session = boto3.Session(**session_kwargs)
            self._client = session.client('s3')
        return self._client
    
    @property
    def resource(self) -> boto3.resource:
        """Lazy-loaded S3 resource."""
        if self._resource is None:
            session_kwargs = {'region_name': self.region_name}
            if self.profile_name:
                session_kwargs['profile_name'] = self.profile_name
            session = boto3.Session(**session_kwargs)
            self._resource = session.resource('s3')
        return self._resource
 
    # Local Folder Management
    
    @staticmethod
    def create_directory(path: str, parents: bool = True, exist_ok: bool = True) -> bool:
        """
        Create a local directory.
        
        Parameters
        ----------
        path : str
            Directory path to create
        parents : bool
            Create parent directories if they don't exist (default: True)
        exist_ok : bool
            Don't raise error if directory exists (default: True)
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            Path(path).mkdir(parents=parents, exist_ok=exist_ok)
            return True
        except (OSError, PermissionError) as e:
            print(f"Error creating directory {path}: {e}")
            return False
    
    @staticmethod
    def create_directories(paths: List[str], parents: bool = True, exist_ok: bool = True) -> Dict[str, bool]:
        """
        Create multiple local directories.
        
        Parameters
        ----------
        paths : list
            List of directory paths to create
        parents : bool
            Create parent directories if they don't exist
        exist_ok : bool
            Don't raise error if directory exists
            
        Returns
        -------
        dict
            Dictionary mapping paths to success status
        """
        results = {}
        for path in paths:
            results[path] = S3Manager.create_directory(path, parents, exist_ok)
        return results
    
    @staticmethod
    def ensure_data_directories(
        base_path: str,
        subdirs: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Create standard data directory structure.
        
        Parameters
        ----------
        base_path : str
            Base directory path
        subdirs : list, optional
            List of subdirectories to create
            Default: ['GEOS_CF', 'OPENAQ', 'FORECASTS', 'MODELS', 'logs']
            
        Returns
        -------
        dict
            Dictionary mapping created paths to success status
        """
        if subdirs is None:
            subdirs = ['GEOS_CF', 'OPENAQ', 'FORECASTS', 'MODELS', 'logs']
        
        paths = [os.path.join(base_path, subdir) for subdir in subdirs]
        return S3Manager.create_directories(paths)
    
    
    # S3 Connectivity & Access Checks
    
    
    def check_access(self, prefix: str = '', bucket: Optional[str] = None) -> bool:
        """
        Check access to a specific S3 bucket and prefix.
        
        Parameters
        ----------
        prefix : str
            S3 prefix (folder path)
        bucket : str, optional
            S3 bucket name (uses default if not provided)
            
        Returns
        -------
        bool
            True if access is confirmed, False otherwise
        """
        bucket = bucket or self.bucket_name
        if not bucket:
            print("No bucket specified.")
            return False
            
        try:
            response = self.client.list_objects_v2(
                Bucket=bucket, 
                Prefix=prefix, 
                MaxKeys=1
            )
            if "Contents" in response:
                print(f"Access confirmed for s3://{bucket}/{prefix}")
            else:
                print(f"No files found, but access confirmed for s3://{bucket}/{prefix}")
            return True
        except ClientError as e:
            print(f"Access denied or error for s3://{bucket}/{prefix}: {e}")
            return False
        except NoCredentialsError:
            print("No AWS credentials found.")
            return False
    
    def check_connectivity(self, prefixes: List[str], bucket: Optional[str] = None) -> Dict[str, bool]:
        """
        Check S3 connectivity for multiple prefixes.
        
        Parameters
        ----------
        prefixes : list
            List of S3 prefixes to check
        bucket : str, optional
            S3 bucket name (uses default if not provided)
            
        Returns
        -------
        dict
            Dictionary mapping prefixes to access status
        """
        results = {}
        for prefix in prefixes:
            results[prefix] = self.check_access(prefix, bucket)
        return results
    
    
    # S3 File Operations - Upload
    
    
    def upload_file(
        self,
        file_path: str,
        s3_key: str,
        bucket: Optional[str] = None,
        extra_args: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Upload a file to S3 bucket.
        
        Parameters
        ----------
        file_path : str
            Local file path
        s3_key : str
            S3 key (path in bucket). If ends with '/', filename is appended.
        bucket : str, optional
            S3 bucket name (uses default if not provided)
        extra_args : dict, optional
            Extra arguments for upload (e.g., ContentType, ACL)
            
        Returns
        -------
        bool
            True if upload successful, False otherwise
        """
        bucket = bucket or self.bucket_name
        if not bucket:
            print("No bucket specified.")
            return False
            
        try:
            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist. Skipping S3 upload.")
                return False
            
            # If s3_key ends with '/', append the filename
            if s3_key.endswith('/'):
                s3_key = f"{s3_key}{os.path.basename(file_path)}"
            
            upload_kwargs = {'Filename': file_path, 'Bucket': bucket, 'Key': s3_key}
            if extra_args:
                upload_kwargs['ExtraArgs'] = extra_args
                
            self.client.upload_file(**upload_kwargs)
            print(f"Successfully uploaded to s3://{bucket}/{s3_key}")
            return True
            
        except (OSError, IOError) as e:
            print(f"File read error uploading {file_path}: {e}")
            return False
        except ClientError as e:
            print(f"S3 error uploading {file_path}: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error uploading {file_path}: {e}")
            return False
    
    def upload_files(
        self,
        file_paths: List[str],
        s3_prefix: str,
        bucket: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        Upload multiple files to S3.
        
        Parameters
        ----------
        file_paths : list
            List of local file paths
        s3_prefix : str
            S3 prefix (folder) for all files
        bucket : str, optional
            S3 bucket name
            
        Returns
        -------
        dict
            Dictionary mapping file paths to upload status
        """
        results = {}
        for file_path in file_paths:
            s3_key = f"{s3_prefix.rstrip('/')}/{os.path.basename(file_path)}"
            results[file_path] = self.upload_file(file_path, s3_key, bucket)
        return results
    
    def upload_dataframe(
        self,
        df: pd.DataFrame,
        s3_key: str,
        bucket: Optional[str] = None,
        file_format: str = 'csv',
        **kwargs
    ) -> bool:
        """
        Upload a pandas DataFrame directly to S3.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to upload
        s3_key : str
            S3 key (path in bucket)
        bucket : str, optional
            S3 bucket name
        file_format : str
            File format ('csv', 'parquet', 'json')
        **kwargs
            Additional arguments for to_csv/to_parquet/to_json
            
        Returns
        -------
        bool
            True if upload successful, False otherwise
        """
        import io
        
        bucket = bucket or self.bucket_name
        if not bucket:
            print("No bucket specified.")
            return False
        
        try:
            buffer = io.BytesIO()
            
            if file_format == 'csv':
                df.to_csv(buffer, index=kwargs.pop('index', False), **kwargs)
                content_type = 'text/csv'
            elif file_format == 'parquet':
                df.to_parquet(buffer, index=kwargs.pop('index', False), **kwargs)
                content_type = 'application/octet-stream'
            elif file_format == 'json':
                json_str = df.to_json(**kwargs)
                buffer = io.BytesIO(json_str.encode('utf-8'))
                content_type = 'application/json'
            else:
                print(f"Unsupported format: {file_format}")
                return False
            
            buffer.seek(0)
            self.client.put_object(
                Bucket=bucket,
                Key=s3_key,
                Body=buffer.getvalue(),
                ContentType=content_type
            )
            print(f"Successfully uploaded DataFrame to s3://{bucket}/{s3_key}")
            return True
            
        except Exception as e:
            print(f"Error uploading DataFrame: {e}")
            return False
    
    
    # S3 File Operations - Download
    
    
    def download_file(
        self,
        s3_key: str,
        local_path: str,
        bucket: Optional[str] = None,
        create_dirs: bool = True
    ) -> bool:
        """
        Download a file from S3.
        
        Parameters
        ----------
        s3_key : str
            S3 key (path in bucket)
        local_path : str
            Local file path to save to
        bucket : str, optional
            S3 bucket name
        create_dirs : bool
            Create parent directories if they don't exist
            
        Returns
        -------
        bool
            True if download successful, False otherwise
        """
        bucket = bucket or self.bucket_name
        if not bucket:
            print("No bucket specified.")
            return False
        
        try:
            if create_dirs:
                parent_dir = os.path.dirname(local_path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
            
            self.client.download_file(bucket, s3_key, local_path)
            print(f"Successfully downloaded s3://{bucket}/{s3_key} to {local_path}")
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                print(f"File not found: s3://{bucket}/{s3_key}")
            else:
                print(f"S3 error downloading {s3_key}: {e}")
            return False
        except Exception as e:
            print(f"Error downloading {s3_key}: {e}")
            return False
    
    def read_to_memory(
        self,
        s3_key: str,
        bucket: Optional[str] = None,
        **kwargs
    ) -> Optional[Union[pd.DataFrame, Dict[str, Any], bytes, str]]:
        """
        Read any file from S3 directly into memory based on extension.
        
        Supported formats:
        - .csv -> pd.DataFrame
        - .json -> dict (or DataFrame with as_dataframe=True in kwargs)
        - .parquet -> pd.DataFrame
        - .txt, .log, .md, .yaml, .yml -> str
        - .pkl, .pickle -> unpickled object
        - other -> bytes
        
        Parameters
        ----------
        s3_key : str
            S3 key (path in bucket)
        bucket : str, optional
            S3 bucket name
        **kwargs
            Additional arguments passed to the reader:
            - For CSV: parse_dates, dtype, etc.
            - For JSON: as_dataframe=True to return DataFrame
            - For Parquet: columns, filters, etc.
            
        Returns
        -------
        pd.DataFrame, dict, str, bytes, or None
            File content in appropriate format, None if failed
            
        Examples
        --------
        >>> manager = S3Manager(bucket_name='my-bucket')
        >>> df = manager.read_to_memory('data/file.csv')
        >>> config = manager.read_to_memory('config/settings.json')
        >>> df = manager.read_to_memory('data/file.parquet')
        >>> text = manager.read_to_memory('logs/output.txt')
        """
        import io
        
        bucket = bucket or self.bucket_name
        if not bucket:
            print("No bucket specified.")
            return None
        
        # Get file extension
        ext = os.path.splitext(s3_key)[1].lower()
        
        try:
            response = self.client.get_object(Bucket=bucket, Key=s3_key)
            body = response['Body'].read()
            
            # CSV
            if ext == '.csv':
                parse_dates = kwargs.pop('parse_dates', [])
                df = pd.read_csv(io.BytesIO(body), parse_dates=parse_dates, **kwargs)
                print(f"CSV loaded from s3://{bucket}/{s3_key}")
                return df
            
            # JSON
            elif ext == '.json':
                as_dataframe = kwargs.pop('as_dataframe', False)
                content = body.decode('utf-8')
                if as_dataframe:
                    df = pd.read_json(content, **kwargs)
                    print(f"JSON loaded as DataFrame from s3://{bucket}/{s3_key}")
                    return df
                else:
                    data = json.loads(content)
                    print(f"JSON loaded from s3://{bucket}/{s3_key}")
                    return data
            
            # Parquet
            elif ext == '.parquet':
                df = pd.read_parquet(io.BytesIO(body), **kwargs)
                print(f"Parquet loaded from s3://{bucket}/{s3_key}")
                return df
            
            # Text files
            elif ext in ['.txt', '.log', '.md', '.yaml', '.yml', '.html', '.xml']:
                encoding = kwargs.get('encoding', 'utf-8')
                text = body.decode(encoding)
                print(f"Text file loaded from s3://{bucket}/{s3_key}")
                return text
            
            # Pickle
            elif ext in ['.pkl', '.pickle']:
                import pickle
                obj = pickle.loads(body)
                print(f"Pickle loaded from s3://{bucket}/{s3_key}")
                return obj
            
            # Excel
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(io.BytesIO(body), **kwargs)
                print(f"Excel loaded from s3://{bucket}/{s3_key}")
                return df
            
            # Default: return raw bytes
            else:
                print(f"File loaded as bytes from s3://{bucket}/{s3_key}")
                return body
                
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                print(f"File not found: s3://{bucket}/{s3_key}")
            else:
                print(f"Error reading from S3: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Invalid JSON format: {e}")
            return None
        except Exception as e:
            print(f"Error reading from S3: {e}")
            return None
        
    
    def download_csv(
        self,
        s3_key: str,
        bucket: Optional[str] = None,
        parse_dates: Optional[List[str]] = None,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Download and read a CSV file from S3 as DataFrame.
        
        Parameters
        ----------
        s3_key : str
            S3 key (path in bucket)
        bucket : str, optional
            S3 bucket name
        parse_dates : list, optional
            List of columns to parse as dates
        **kwargs
            Additional arguments for pd.read_csv
            
        Returns
        -------
        pd.DataFrame or None
            DataFrame if successful, None otherwise
        """
        bucket = bucket or self.bucket_name
        if not bucket:
            print("No bucket specified.")
            return None
        
        try:
            response = self.client.get_object(Bucket=bucket, Key=s3_key)
            df = pd.read_csv(
                response['Body'],
                parse_dates=parse_dates or [],
                **kwargs
            )
            print(f"CSV loaded from s3://{bucket}/{s3_key}")
            return df
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                print(f"File not found: s3://{bucket}/{s3_key}")
            else:
                print(f"Error reading CSV from S3: {e}")
            return None
        except Exception as e:
            print(f"Error reading CSV from S3: {e}")
            return None
    
    def read_file(
        self,
        s3_prefix: str,
        location_name: str,
        bucket: Optional[str] = None,
        file_extension: str = '.csv'
    ) -> Optional[pd.DataFrame]:
        """
        Read model DataFrame from S3 using location name pattern.
        
        Parameters
        ----------
        s3_prefix : str
            S3 prefix/folder (e.g., "snwg_forecast_working_files/GEOS_CF/")
        location_name : str
            Location identifier for the filename
        bucket : str, optional
            S3 bucket name
        file_extension : str
            File extension (default: '.csv')
            
        Returns
        -------
        pd.DataFrame or None
            DataFrame if successful, None otherwise
        """
        s3_key = f"{s3_prefix.rstrip('/')}/{location_name}{file_extension}"
        
        if file_extension == '.csv':
            return self.download_csv(s3_key, bucket, parse_dates=['time'])
        else:
            # For other formats, download to temp and read
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp:
                if self.download_file(s3_key, tmp.name, bucket):
                    try:
                        return pd.read_csv(tmp.name) if file_extension == '.csv' else None
                    finally:
                        os.unlink(tmp.name)
                return None
    
    
    # S3 Metadata & Status
    
    
    def file_exists(
        self,
        s3_key: str,
        bucket: Optional[str] = None
    ) -> bool:
        """
        Check if a file exists in S3.
        
        Parameters
        ----------
        s3_key : str
            S3 key (path in bucket)
        bucket : str, optional
            S3 bucket name
            
        Returns
        -------
        bool
            True if exists, False otherwise
        """
        bucket = bucket or self.bucket_name
        if not bucket:
            return False
        
        try:
            self.client.head_object(Bucket=bucket, Key=s3_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            return False
        except Exception:
            return False
    
    def check_file_status(
        self,
        s3_prefix: str,
        location_name: str,
        bucket: Optional[str] = None,
        file_extension: str = '.csv'
    ) -> bool:
        """
        Check if model CSV exists in S3.
        
        Parameters
        ----------
        s3_prefix : str
            S3 prefix/folder
        location_name : str
            Location identifier for the filename
        bucket : str, optional
            S3 bucket name
        file_extension : str
            File extension (default: '.csv')
            
        Returns
        -------
        bool
            True if exists, False otherwise
        """
        s3_key = f"{s3_prefix.rstrip('/')}/{location_name}{file_extension}"
        return self.file_exists(s3_key, bucket)
    
    def get_file_metadata(
        self,
        s3_key: str,
        bucket: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a file in S3.
        
        Parameters
        ----------
        s3_key : str
            S3 key (path in bucket)
        bucket : str, optional
            S3 bucket name
            
        Returns
        -------
        dict or None
            Metadata dictionary with keys:
            - 'LastModified': datetime of last modification
            - 'ContentLength': file size in bytes
            - 'ContentType': MIME type
            - 'ETag': entity tag
            - 'Metadata': user-defined metadata
        """
        bucket = bucket or self.bucket_name
        if not bucket:
            return None
        
        try:
            response = self.client.head_object(Bucket=bucket, Key=s3_key)
            return {
                'LastModified': response.get('LastModified'),
                'ContentLength': response.get('ContentLength'),
                'ContentType': response.get('ContentType'),
                'ETag': response.get('ETag', '').strip('"'),
                'Metadata': response.get('Metadata', {})
            }
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                print(f"File not found: s3://{bucket}/{s3_key}")
            else:
                print(f"Error getting metadata: {e}")
            return None
        except Exception as e:
            print(f"Error getting metadata: {e}")
            return None
    
    def get_last_modified(
        self,
        s3_key: str,
        bucket: Optional[str] = None
    ) -> Optional[datetime]:
        """
        Get last modified timestamp for a file in S3.
        
        Parameters
        ----------
        s3_key : str
            S3 key (path in bucket)
        bucket : str, optional
            S3 bucket name
            
        Returns
        -------
        datetime or None
            Last modified datetime, or None if not found
        """
        metadata = self.get_file_metadata(s3_key, bucket)
        return metadata.get('LastModified') if metadata else None
    
    def get_file_size(
        self,
        s3_key: str,
        bucket: Optional[str] = None,
        human_readable: bool = False
    ) -> Optional[Union[int, str]]:
        """
        Get file size for a file in S3.
        
        Parameters
        ----------
        s3_key : str
            S3 key (path in bucket)
        bucket : str, optional
            S3 bucket name
        human_readable : bool
            Return human-readable string (e.g., '1.5 MB')
            
        Returns
        -------
        int, str, or None
            File size in bytes (or human-readable string), or None if not found
        """
        metadata = self.get_file_metadata(s3_key, bucket)
        if not metadata:
            return None
        
        size_bytes = metadata.get('ContentLength', 0)
        
        if human_readable:
            return self._format_bytes(size_bytes)
        return size_bytes
    
    @staticmethod
    def _format_bytes(size_bytes: int) -> str:
        """Format bytes into human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
    
    def is_file_recent(
        self,
        s3_key: str,
        hours_threshold: int = 5,
        bucket: Optional[str] = None
    ) -> bool:
        """
        Check if an S3 file was modified within the last N hours.
        
        Parameters
        ----------
        s3_key : str
            S3 key (path in bucket)
        hours_threshold : int
            Number of hours to check (default: 5)
        bucket : str, optional
            S3 bucket name
            
        Returns
        -------
        bool
            True if file exists and was modified within the threshold
        """
        last_modified = self.get_last_modified(s3_key, bucket)
        if not last_modified:
            return False
        
        # Make datetime comparison timezone-aware
        from datetime import timezone
        now = datetime.now(timezone.utc)
        threshold = now - timedelta(hours=hours_threshold)
        
        return last_modified > threshold
    
    
    # S3 Listing Operations
    
    
    def list_objects(
        self,
        prefix: str = '',
        bucket: Optional[str] = None,
        max_keys: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        List objects in an S3 bucket with given prefix.
        
        Parameters
        ----------
        prefix : str
            S3 prefix to filter objects
        bucket : str, optional
            S3 bucket name
        max_keys : int
            Maximum number of keys to return
            
        Returns
        -------
        list
            List of object dictionaries with Key, Size, LastModified
        """
        bucket = bucket or self.bucket_name
        if not bucket:
            return []
        
        try:
            objects = []
            paginator = self.client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(
                Bucket=bucket,
                Prefix=prefix,
                PaginationConfig={'MaxItems': max_keys}
            ):
                for obj in page.get('Contents', []):
                    objects.append({
                        'Key': obj['Key'],
                        'Size': obj['Size'],
                        'LastModified': obj['LastModified'],
                        'ETag': obj['ETag'].strip('"')
                    })
            
            return objects
            
        except ClientError as e:
            print(f"Error listing objects: {e}")
            return []
    
    def list_files(
        self,
        prefix: str = '',
        bucket: Optional[str] = None,
        extension: Optional[str] = None
    ) -> List[str]:
        """
        List file keys in an S3 bucket with given prefix.
        
        Parameters
        ----------
        prefix : str
            S3 prefix to filter objects
        bucket : str, optional
            S3 bucket name
        extension : str, optional
            Filter by file extension (e.g., '.csv', '.json')
            
        Returns
        -------
        list
            List of S3 keys (file paths)
        """
        objects = self.list_objects(prefix, bucket)
        keys = [obj['Key'] for obj in objects]
        
        if extension:
            keys = [k for k in keys if k.endswith(extension)]
        
        return keys
    
    
    # S3 Delete Operations
    
    
    def delete_file(
        self,
        s3_key: str,
        bucket: Optional[str] = None
    ) -> bool:
        """
        Delete a file from S3.
        
        Parameters
        ----------
        s3_key : str
            S3 key (path in bucket)
        bucket : str, optional
            S3 bucket name
            
        Returns
        -------
        bool
            True if deletion successful, False otherwise
        """
        bucket = bucket or self.bucket_name
        if not bucket:
            print("No bucket specified.")
            return False
        
        try:
            self.client.delete_object(Bucket=bucket, Key=s3_key)
            print(f"Successfully deleted s3://{bucket}/{s3_key}")
            return True
        except ClientError as e:
            print(f"Error deleting {s3_key}: {e}")
            return False
    
    def delete_files(
        self,
        s3_keys: List[str],
        bucket: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        Delete multiple files from S3.
        
        Parameters
        ----------
        s3_keys : list
            List of S3 keys to delete
        bucket : str, optional
            S3 bucket name
            
        Returns
        -------
        dict
            Dictionary mapping keys to deletion status
        """
        results = {}
        for s3_key in s3_keys:
            results[s3_key] = self.delete_file(s3_key, bucket)
        return results


# Convenience Functions (Standalone Usage)

def check_s3_access(bucket: str, prefix: str = '') -> bool:
    """
    Check access to a specific S3 bucket and prefix.
    
    Parameters
    ----------
    bucket : str
        S3 bucket name
    prefix : str
        S3 prefix (folder path)
        
    Returns
    -------
    bool
        True if access is confirmed, False otherwise
    """
    manager = S3Manager(bucket_name=bucket)
    return manager.check_access(prefix)


def check_s3_connectivity(bucket: str, prefixes: List[str]) -> Dict[str, bool]:
    """
    Check S3 connectivity for multiple prefixes.
    
    Parameters
    ----------
    bucket : str
        S3 bucket name
    prefixes : list
        List of S3 prefixes to check
        
    Returns
    -------
    dict
        Dictionary mapping prefixes to access status
    """
    manager = S3Manager(bucket_name=bucket)
    return manager.check_connectivity(prefixes)


def upload_to_s3(
    file_path: str,
    s3_client: boto3.client,
    s3_bucket: str,
    s3_key: str
) -> bool:
    """
    Upload file to S3 bucket with verification.
    
    This is a compatibility wrapper for the original funcs.py function signature.
    
    Parameters
    ----------
    file_path : str
        Local file path
    s3_client : boto3.client
        S3 client instance (not used, for compatibility)
    s3_bucket : str
        S3 bucket URI (s3:// prefix optional)
    s3_key : str
        S3 prefix/folder for the file
        
    Returns
    -------
    bool
        True if upload successful, False otherwise
    """
    manager = S3Manager(bucket_name=s3_bucket)
    manager._client = s3_client  # Use provided client
    
    # Append filename to key if it doesn't contain it
    if not s3_key.endswith(os.path.basename(file_path)):
        full_key = f"{s3_key.rstrip('/')}/{os.path.basename(file_path)}"
    else:
        full_key = s3_key
    
    return manager.upload_file(file_path, full_key)


def read_s3_file(
    s3_client: boto3.client,
    bucket_name: str,
    s3_prefix: str,
    location_name: str
) -> Optional[pd.DataFrame]:
    """
    Read model DataFrame from S3.
    
    This is a compatibility wrapper for the original funcs.py function signature.
    
    Parameters
    ----------
    s3_client : boto3.client
        S3 client
    bucket_name : str
        S3 bucket name (without s3:// prefix)
    s3_prefix : str
        S3 prefix/folder (e.g., "snwg_forecast_working_files/GEOS_CF/")
    location_name : str
        Location identifier for the filename
        
    Returns
    -------
    pd.DataFrame or None
        DataFrame if successful, None otherwise
    """
    manager = S3Manager(bucket_name=bucket_name)
    manager._client = s3_client  # Use provided client
    return manager.read_file(s3_prefix, location_name)


def check_s3_file_status(
    s3_client: boto3.client,
    bucket_name: str,
    s3_prefix: str,
    location_name: str
) -> bool:
    """
    Check if model CSV exists in S3.
    
    This is a compatibility wrapper for the original funcs.py function signature.
    
    Parameters
    ----------
    s3_client : boto3.client
        S3 client
    bucket_name : str
        S3 bucket name (without s3:// prefix)
    s3_prefix : str
        S3 prefix/folder (e.g., "snwg_forecast_working_files/GEOS_CF/")
    location_name : str
        Location identifier for the filename
        
    Returns
    -------
    bool
        True if exists, False otherwise
    """
    manager = S3Manager(bucket_name=bucket_name)
    manager._client = s3_client  # Use provided client
    return manager.check_file_status(s3_prefix, location_name)


def is_forecast_recent(file_path: str, hours_threshold: int = 5) -> bool:
    """
    Check if a local forecast file was generated within the last N hours.
    
    Parameters
    ----------
    file_path : str
        Path to the forecast file
    hours_threshold : int
        Number of hours to check (default: 5)
        
    Returns
    -------
    bool
        True if file exists and was modified within the threshold, False otherwise
    """
    try:
        if not os.path.exists(file_path):
            return False
        
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        time_threshold = datetime.now() - timedelta(hours=hours_threshold)
        
        return file_mod_time > time_threshold
    except (OSError, ValueError, OverflowError) as e:
        print(f"Error checking forecast recency for {file_path}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error in is_forecast_recent: {e}")
        return False


def create_directory(path: str) -> bool:
    """
    Create a local directory (convenience function).
    
    Parameters
    ----------
    path : str
        Directory path to create
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    return S3Manager.create_directory(path)


def ensure_data_directories(base_path: str, subdirs: Optional[List[str]] = None) -> Dict[str, bool]:
    """
    Create standard data directory structure (convenience function).
    
    Parameters
    ----------
    base_path : str
        Base directory path
    subdirs : list, optional
        List of subdirectories to create
        
    Returns
    -------
    dict
        Dictionary mapping created paths to success status
    """
    return S3Manager.ensure_data_directories(base_path, subdirs)
