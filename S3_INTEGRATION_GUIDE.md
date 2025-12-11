# S3 Integration Guide for `_read_model()` Method

## Overview
The `_read_model()` method in `MLpred/mlpred.py` has been updated to support AWS S3 bucket operations for storing and retrieving GEOS-CF model data. The changes are the following:

1. **S3 Save Functionality**: Automatically save downloaded CSV data to S3 bucket
2. **S3 Read Functionality**: Check S3 for existing data before downloading
3. **Upload Verification**: Verify successful uploads by checking object metadata
4. **Incremental Updates**: Only fetch new forecast data if historical data exists
5. **Local Fallback**: Optional local saving as a secondary storage option

## New Parameters

### `_read_model()` Method Signature
```python
def _read_model(
    self,
    ilon,
    ilat,
    start,
    end,
    resample=None,
    source=None,
    template=None,
    collections=None,
    remove_outlier=0,
    gases=DEFAULT_GASES,
    s3_bucket=None,              # NEW: S3 bucket name
    s3_prefix="geos_cf_data/",   # NEW: S3 path prefix
    s3_region="us-east-1",       # NEW: AWS region
    save_local=False,             # NEW: Save locally as well
    local_dir="GEOS_CF",          # NEW: Local directory
    **kwargs,
):
```


## Usage Examples

### Example 1: S3-Only (Recommended for Production)
```python
# Initialize site
site = ObsSite(location_id="12345", model_source="s3")

# Read model data with S3 storage
site.read_mod(
    source="s3",
    start=datetime(2023, 1, 1),
    end=datetime(2023, 12, 31),
    s3_bucket="my-data-bucket",
    s3_prefix="geos_cf_data/",
    s3_region="us-east-1",
    save_local=False  # Only use S3
)
```

### Example 2: S3 with Local Backup
```python
site.read_mod(
    source="s3",
    start=datetime(2023, 1, 1),
    end=datetime(2023, 12, 31),
    s3_bucket="my-data-bucket",
    s3_prefix="geos_cf_data/",
    s3_region="us-east-1",
    save_local=True,        # Also save locally
    local_dir="local_cache"
)
```

### Example 3: Local-Only (No S3)
```python
site.read_mod(
    source="s3",
    start=datetime(2023, 1, 1),
    end=datetime(2023, 12, 31),
    save_local=True,
    local_dir="GEOS_CF"
)
```

### Example 4: With Historical Data Check
```python
# If historical CSV exists in S3, only fetches new forecast data
# If not found or incomplete, downloads full dataset
site.read_mod(
    source="s3",
    start=datetime(2018, 1, 1),
    end=datetime.today(),
    s3_bucket="my-data-bucket",
    s3_prefix="geos_cf_data/",
    s3_region="us-east-1"
)
```

## Workflow & Behavior

### First Run (No Cached Data)
```
1. Check S3 for existing CSV → Not found
2. Download all historical data (2018-present)
3. Download replay data
4. Download forecast data
5. Combine all data, remove duplicates, sort by time
6. Upload to S3 with verification
7. (Optional) Save locally if save_local=True
8. Filter by requested date range and return
```

### Subsequent Runs (With Cached Data)
```
1. Check S3 for existing CSV → Found ✓
2. Check data completeness (has data from 2018?)
3. If complete: Fetch only new forecast data
4. Combine cached data + forecast data
5. Remove duplicates, sort by time
6. Upload updated CSV to S3 with verification
7. (Optional) Update local copy if save_local=True
8. Filter by requested date range and return
```

### Data Not Found in S3
```
1. Check S3 → Not found
2. Check local cache (if save_local=True) → Found
3. Use local data, fetch missing forecasts
4. Upload updated data to S3
```

## Key Features

### 1. Automatic S3 Client Initialization
- Attempts to create boto3 S3 client on method call
- Gracefully falls back to local/no storage if connection fails
- Prints status messages for debugging

### 2. CSV Storage Format
- **Filename**: `loc_{lat}_{lon}.csv` (e.g., `loc_40_50_m122_50.csv`)
- **Path in S3**: `s3://bucket-name/{s3_prefix}/loc_{lat}_{lon}.csv`
- **Format**: Standard pandas CSV with date parsing

### 3. Upload Verification
- Verifies successful upload using `head_object` API call
- Reports file size in bytes
- Returns `True` on success, `False` on failure

### 4. Incremental Updates
- Checks if cached data spans from 2018-01-01 (base date)
- If yes: Only fetches forecast data (faster)
- If no: Downloads complete dataset

### 5. Error Handling
- Connection errors are caught and reported
- Missing data is handled gracefully (no duplicates)
- Invalid dates/data gaps trigger full re-download

## New Helper Method

The S3 save operations now use the existing `write_to_s3()` helper function from `funcs.py`:

### `funcs.write_to_s3(data, s3_client, bucket, s3_key, data_format='json')`
Saves data directly to S3 without creating local files.

**Parameters:**
- `data`: dict or DataFrame to save
- `s3_client`: Initialized boto3 S3 client
- `bucket`: S3 bucket name (without s3:// prefix)
- `s3_key`: S3 object key/path within bucket
- `data_format`: Format to save ('json' or 'csv')

**Returns:**
- `True`: Save successful
- `False`: Save failed

This function is used internally by `_read_model()` when saving CSV data to S3.
