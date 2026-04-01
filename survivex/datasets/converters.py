# survivex/datasets/converters.py
"""
Converters for loading datasets from other survival analysis libraries
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
import warnings


# survivex/datasets/converters.py
"""
Converters for loading datasets from other survival analysis libraries
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
import warnings


def detect_and_convert_event_column(df: pd.DataFrame, 
                                   event_col: str = None) -> Tuple[pd.DataFrame, str]:
    """
    Detect and convert event column to standard 0/1 format
    """
    # If event column not specified, try to detect it
    if event_col is None:
        event_keywords = ['event', 'status', 'death', 'fail', 'censor', 'observed', 
                         'arrest', 'relapse', 'recur', 'died', 'failed']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in event_keywords):
                event_col = col
                print(f"   Auto-detected event column: '{event_col}'")
                break
    
    if event_col is None or event_col not in df.columns:
        raise ValueError(f"Could not find event column. Columns available: {list(df.columns)}")
    
    # Get unique values to determine conversion needed
    unique_vals = df[event_col].dropna().unique()
    original_event_col = event_col
    
    # Determine the conversion needed
    if set(unique_vals).issubset({0, 1}):
        # Already in 0/1 format
        df['event'] = df[event_col].astype(int)
        print(f"   Event column already in 0/1 format")
        
    elif set(unique_vals).issubset({True, False}):
        # Boolean format
        df['event'] = df[event_col].astype(int)
        print(f"   Converted boolean event values to 0/1")
        
    elif set(unique_vals).issubset({1, 2}):
        # Common format: 1=censored, 2=event (like lung dataset)
        df['event'] = (df[event_col] == 2).astype(int)
        print(f"   Converted status (1=censored, 2=event) to event (0/1)")
        
    elif set(unique_vals).issubset({0, 1, 2}):
        # Might be competing risks or 0=censored, 1=event1, 2=event2
        # Default: treat any non-zero as event
        df['event'] = (df[event_col] > 0).astype(int)
        print(f"   Converted multi-value status to binary (0=censored, >0=event)")
        print(f"   Warning: Original had values {sorted(unique_vals)} - may be competing risks")
        
    elif set(unique_vals).issubset({'Dead', 'Alive', 'dead', 'alive'}):
        # Text format
        df['event'] = df[event_col].str.lower().map({'dead': 1, 'alive': 0})
        print(f"   Converted text values (Dead/Alive) to 0/1")
        
    elif set(unique_vals).issubset({'Yes', 'No', 'yes', 'no'}):
        # Yes/No format
        df['event'] = df[event_col].str.capitalize().map({'Yes': 1, 'No': 0})
        print(f"   Converted Yes/No values to 0/1")
        
    else:
        # Try numeric conversion
        numeric_vals = pd.to_numeric(df[event_col], errors='coerce')
        if not numeric_vals.isna().all():
            # Check if it looks like survival times (all positive, wide range)
            if numeric_vals.min() >= 0 and numeric_vals.max() > 2:
                print(f"   Warning: Event column '{event_col}' has values {unique_vals[:5]}...")
                print(f"   This might be a time column, not an event indicator")
            
            # For now, treat 0 as censored, anything else as event
            df['event'] = (numeric_vals != 0).astype(int)
            print(f"   Converted numeric values (0=censored, non-zero=event)")
        else:
            raise ValueError(
                f"Cannot interpret event column '{event_col}' with values: {unique_vals[:10]}"
            )
    
    # Keep original column if it's not named 'event'
    if original_event_col != 'event' and 'event' != original_event_col:
        df[f'original_{original_event_col}'] = df[original_event_col].copy()
    
    return df, original_event_col


def detect_and_convert_time_column(df: pd.DataFrame, 
                                  time_col: str = None) -> Tuple[pd.DataFrame, str]:
    """
    Detect and convert time column to standard format
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    time_col : str, optional
        Known time column name
        
    Returns
    -------
    tuple of (DataFrame with standardized 'time' column, original time column name)
    """
    # If time column not specified, try to detect it
    if time_col is None:
        time_keywords = ['time', 'duration', 'days', 'months', 'years', 'week', 
                        'survival', 't', 'stop', 'end', 'followup', 'fu']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in time_keywords):
                # Check if it's numeric
                if pd.api.types.is_numeric_dtype(df[col]) or pd.to_numeric(df[col], errors='coerce').notna().any():
                    time_col = col
                    print(f"   Auto-detected time column: '{time_col}'")
                    break
    
    if time_col is None or time_col not in df.columns:
        raise ValueError(f"Could not find time column. Columns available: {list(df.columns)}")
    
    # Convert to numeric and create standardized 'time' column
    df['time'] = pd.to_numeric(df[time_col], errors='coerce')
    
    # Keep original column if it's not named 'time'
    if time_col != 'time':
        df[f'original_{time_col}'] = df[time_col].copy()
    
    return df, time_col


def load_from_lifelines(dataset_name: str, 
                       standardize: bool = True) -> pd.DataFrame:
    """
    Load datasets from lifelines library with automatic standardization
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    standardize : bool
        Whether to standardize column names and values
        
    Returns
    -------
    pd.DataFrame with standardized column names and values
    """
    try:
        import lifelines.datasets as ld
    except ImportError:
        raise ImportError(
            "lifelines not installed. Install with: pip install lifelines"
        )
    
    # Dataset mapping - now we don't need to specify exact conversions
    dataset_loaders = {
        'lung': ld.load_lung,
        'rossi': ld.load_rossi,
        'waltons': ld.load_waltons,
        'kidney_transplant': ld.load_kidney_transplant,
        'stanford_heart_transplant': ld.load_stanford_heart_transplants,
        'dd': ld.load_dd,
        'regression': ld.load_regression_dataset,
    }
    
    # Add aliases for convenience
    dataset_loaders.update({
        'kidney': dataset_loaders['kidney_transplant'],
        'stanford': dataset_loaders['stanford_heart_transplant'],
    })
    
    if dataset_name not in dataset_loaders:
        available = ', '.join(dataset_loaders.keys())
        raise ValueError(
            f"Unknown lifelines dataset: {dataset_name}. "
            f"Available: {available}"
        )
    
    # Load the dataset
    print(f"Loading '{dataset_name}' from lifelines...")
    df = dataset_loaders[dataset_name]()
    print(f"   Original shape: {df.shape}")
    print(f"   Original columns: {list(df.columns)}")
    
    if standardize:
        # Auto-detect and convert time column
        df, time_col = detect_and_convert_time_column(df)
        
        # Auto-detect and convert event column
        df, event_col = detect_and_convert_event_column(df)
        
        print(f"   Standardized: time='{time_col}', event='{event_col}'")
        
        # Show summary
        event_rate = df['event'].mean()
        print(f"   Events: {df['event'].sum()}/{len(df)} ({event_rate:.1%})")
        print(f"   Time range: [{df['time'].min():.1f}, {df['time'].max():.1f}]")
    
    return df


def load_from_sksurv(dataset_name: str,
                    standardize: bool = True) -> pd.DataFrame:
    """
    Load datasets from scikit-survival library
    """
    try:
        from sksurv import datasets
    except ImportError:
        raise ImportError(
            "scikit-survival not installed. Install with: pip install scikit-survival"
        )
    
    print(f"Loading '{dataset_name}' from scikit-survival...")
    
    # Load based on dataset name
    if dataset_name == 'veterans':
        X, y = datasets.load_veterans_lung_cancer()
    elif dataset_name == 'gbsg2':
        X, y = datasets.load_gbsg2()
    elif dataset_name == 'whas500':
        X, y = datasets.load_whas500()
    elif dataset_name == 'flchain':
        X, y = datasets.load_flchain()
    else:
        raise ValueError(
            f"Unknown sksurv dataset: {dataset_name}. "
            f"Available: veterans, gbsg2, whas500, flchain"
        )
    
    # Convert structured array to DataFrame
    df = X.copy()
    
    # Extract time and event from structured array y
    # sksurv uses structured arrays with 'event' and 'time' fields
    df['event'] = y['event'].astype(int)
    df['time'] = y['time']
    
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    if standardize:
        event_rate = df['event'].mean()
        print(f"   Events: {df['event'].sum()}/{len(df)} ({event_rate:.1%})")
    
    return df


# survivex/datasets/converters.py - Fix auto_detect_format function

def auto_detect_format(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Auto-detect the format of survival data
    
    Returns
    -------
    Dict with detected format information
    """
    detected = {
        'format': 'standard',
        'time_cols': [],
        'event_cols': [],
        'time_col_best': None,
        'event_col_best': None,
        'possible_subject_col': None,
        'possible_competing_risks': False,
        'possible_recurrent': False,
        'date_cols': []
    }
    
    # Look for time-related columns
    time_keywords = ['time', 'duration', 'days', 'months', 'years', 'week', 
                    'survival', 't', 'stop', 'end', 'followup', 'fu']
    
    # PRIORITIZE exact matches first
    for col in df.columns:
        col_lower = col.lower()
        
        # Exact match for 'time' should be highest priority
        if col_lower == 'time':
            detected['time_cols'].insert(0, col)  # Insert at beginning
            detected['time_col_best'] = col
            break
    
    # Then look for other time-related columns
    if detected['time_col_best'] is None:
        for col in df.columns:
            col_lower = col.lower()
            
            # Skip columns that are clearly not time (like 'inst' for institution)
            if col_lower in ['inst', 'institution', 'id', 'patient_id']:
                continue
                
            # Check for time keywords
            if any(keyword in col_lower for keyword in time_keywords):
                if pd.api.types.is_numeric_dtype(df[col]):
                    detected['time_cols'].append(col)
                    if detected['time_col_best'] is None:
                        detected['time_col_best'] = col
        
        # Check for date columns
        for col in df.columns:
            try:
                pd.to_datetime(df[col])
                detected['date_cols'].append(col)
            except:
                pass
    
    # Look for event-related columns - same priority approach
    event_keywords = ['event', 'status', 'death', 'fail', 'censor', 'observed', 
                     'arrest', 'relapse', 'recur', 'died', 'failed', 'alive', 'dead']
    
    # Exact matches first
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['event', 'status']:
            detected['event_cols'].insert(0, col)
            detected['event_col_best'] = col
            break
    
    # Then keyword matches
    if detected['event_col_best'] is None:
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in event_keywords):
                detected['event_cols'].append(col)
                if detected['event_col_best'] is None:
                    detected['event_col_best'] = col
    
    # If no time column found but we have date columns, suggest date range
    if not detected['time_cols'] and len(detected['date_cols']) >= 2:
        detected['format'] = 'date_range'
        detected['suggested_time_cols'] = detected['date_cols'][:2]
    
    # Check for subject/ID column (recurrent events)
    id_keywords = ['id', 'subject', 'patient', 'person', 'individual', 'user', 'customer']
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in id_keywords):
            if df[col].nunique() < len(df):
                detected['possible_subject_col'] = col
                detected['possible_recurrent'] = True
                break
    
    # Check for competing risks (event column with multiple values)
    if detected['event_col_best']:
        col = detected['event_col_best']
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) > 2:
            if all(isinstance(v, (int, float)) for v in unique_vals):
                if min(unique_vals) >= 0 and max(unique_vals) <= 10:
                    detected['possible_competing_risks'] = True
                    detected['format'] = 'competing_risks'
    
    # Check for counting process format
    if 'start' in df.columns and 'stop' in df.columns:
        detected['format'] = 'counting_process'
        detected['time_col_best'] = 'stop'
    
    return detected

def standardize_column_names(df: pd.DataFrame, 
                            time_col: str = None,
                            event_col: str = None,
                            auto_detect: bool = True) -> pd.DataFrame:
    """
    Standardize column names and values to common format
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    time_col : str, optional
        Known time column
    event_col : str, optional  
        Known event column
    auto_detect : bool
        Whether to auto-detect columns if not specified
        
    Returns
    -------
    pd.DataFrame with standardized 'time' and 'event' columns
    """
    df = df.copy()
    
    # Auto-detect if needed
    if auto_detect and (time_col is None or event_col is None):
        detected = auto_detect_format(df)
        
        if time_col is None:
            time_col = detected.get('time_col_best')
            if time_col:
                print(f"🔍 Auto-detected time column: '{time_col}'")
        
        if event_col is None:
            event_col = detected.get('event_col_best')
            if event_col:
                print(f"🔍 Auto-detected event column: '{event_col}'")
    
    # Standardize time column
    if time_col:
        df, _ = detect_and_convert_time_column(df, time_col)
    
    # Standardize event column
    if event_col:
        df, _ = detect_and_convert_event_column(df, event_col)
    
    return df


def load_from_pycox(dataset_name: str) -> pd.DataFrame:
    """
    Load datasets from pycox library
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset. Options:
        - 'metabric': Molecular Taxonomy of Breast Cancer
        - 'gbsg': German Breast Cancer Study Group
        - 'support': Study to Understand Prognoses Preferences Outcomes
        - 'flchain': Serum free light chain
        - 'kkbox': KKBox music streaming churn
        - 'nwtco': National Wilms Tumor Study
        
    Returns
    -------
    pd.DataFrame with standardized column names
    """
    try:
        from pycox import datasets
    except ImportError:
        raise ImportError(
            "pycox not installed. Install with: pip install pycox"
        )
    
    # Load dataset
    dataset_loaders = {
        'metabric': datasets.metabric,
        'gbsg': datasets.gbsg,
        'support': datasets.support,
        'flchain': datasets.flchain,
        'kkbox': datasets.kkbox,
        'nwtco': datasets.nwtco
    }
    
    if dataset_name not in dataset_loaders:
        available = ', '.join(dataset_loaders.keys())
        raise ValueError(
            f"Unknown pycox dataset: {dataset_name}. "
            f"Available: {available}"
        )
    
    # Load and process
    df = dataset_loaders[dataset_name].read_df()
    
    # pycox datasets usually have 'duration' and 'event' columns
    if 'duration' in df.columns:
        df = df.rename(columns={'duration': 'time'})
    
    print(f" Loaded '{dataset_name}' from pycox:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    return df


def convert_competing_risks(df: pd.DataFrame, 
                           event_col: str,
                           event_types: Dict[Any, int]) -> pd.DataFrame:
    """
    Convert competing risks data to standard format
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with competing risks
    event_col : str
        Column containing event types
    event_types : dict
        Mapping of event types to integers
        
    Returns
    -------
    pd.DataFrame with additional columns for each event type
    """
    df = df.copy()
    
    # Create binary columns for each event type
    for event_name, event_code in event_types.items():
        df[f'event_{event_name}'] = (df[event_col] == event_code).astype(int)
    
    # Keep original event column as 'event_type'
    df['event_type'] = df[event_col]
    
    # Create overall event indicator (any event occurred)
    df['event'] = (df[event_col] != 0).astype(int)
    
    return df


def convert_recurrent_events(df: pd.DataFrame,
                            subject_col: str,
                            time_col: str,
                            event_col: str) -> pd.DataFrame:
    """
    Convert recurrent events data to counting process format
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with recurrent events
    subject_col : str
        Column identifying subjects
    time_col : str
        Time column
    event_col : str
        Event column
        
    Returns
    -------
    pd.DataFrame in counting process format with (start, stop, event) columns
    """
    df = df.copy()
    df = df.sort_values([subject_col, time_col])
    
    # Create counting process format
    result = []
    
    for subject in df[subject_col].unique():
        subject_data = df[df[subject_col] == subject].copy()
        subject_data = subject_data.sort_values(time_col)
        
        start_time = 0
        for idx, row in subject_data.iterrows():
            result.append({
                subject_col: subject,
                'start': start_time,
                'stop': row[time_col],
                'event': row[event_col],
                **{col: row[col] for col in subject_data.columns 
                   if col not in [subject_col, time_col, event_col]}
            })
            start_time = row[time_col]
    
    result_df = pd.DataFrame(result)
    
    print(f" Converted to counting process format:")
    print(f"   Original: {len(df)} rows")
    print(f"   Converted: {len(result_df)} intervals")
    print(f"   Subjects: {df[subject_col].nunique()}")
    
    return result_df


# survivex/datasets/converters.py - Simplified auto_detect_format

def auto_detect_format(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Auto-detect the format of survival data using ONLY predefined column names
    """
    # Define acceptable column names
    ACCEPTABLE_TIME_COLS = [
        'time', 'Time', 'TIME',
        'duration', 'Duration', 
        'survival_time', 'survival_days', 'survival_months',
        'days', 'months', 'years',
        'T', 't',
        'stop', 'end',
        'followup', 'follow_up'
    ]
    
    ACCEPTABLE_EVENT_COLS = [
        'event', 'Event', 'EVENT',
        'status', 'Status', 
        'death', 'Death',
        'fail', 'failure', 'failed',
        'censor', 'censored',
        'observed',
        'E', 'e',
        'arrest', 'relapse'
    ]
    
    detected = {
        'format': 'standard',
        'time_cols': [],
        'event_cols': [],
        'time_col_best': None,
        'event_col_best': None,
        'possible_subject_col': None,
        'possible_competing_risks': False,
        'possible_recurrent': False
    }
    
    # Find time columns - ONLY from acceptable list
    for col in df.columns:
        if col in ACCEPTABLE_TIME_COLS:
            detected['time_cols'].append(col)
            # Prioritize 'time' if it exists
            if col in ['time', 'Time', 'TIME']:
                detected['time_col_best'] = col
            elif detected['time_col_best'] is None:
                detected['time_col_best'] = col
    
    # Find event columns - ONLY from acceptable list
    for col in df.columns:
        if col in ACCEPTABLE_EVENT_COLS:
            detected['event_cols'].append(col)
            # Prioritize 'event' or 'status'
            if col in ['event', 'Event', 'EVENT']:
                detected['event_col_best'] = col
            elif col in ['status', 'Status'] and detected['event_col_best'] is None:
                detected['event_col_best'] = col
            elif detected['event_col_best'] is None:
                detected['event_col_best'] = col
    
    # If we couldn't find required columns, raise clear error
    if not detected['time_cols']:
        raise ValueError(
            f"Could not find time column. Your columns: {list(df.columns)}\n"
            f"Acceptable time column names: {ACCEPTABLE_TIME_COLS}\n"
            f"Please rename your time column or specify it explicitly with time_col='your_column_name'"
        )
    
    if not detected['event_cols']:
        raise ValueError(
            f"Could not find event column. Your columns: {list(df.columns)}\n"
            f"Acceptable event column names: {ACCEPTABLE_EVENT_COLS}\n"
            f"Please rename your event column or specify it explicitly with event_col='your_column_name'"
        )
    
    return detected


def standardize_column_names(df: pd.DataFrame, 
                            mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Standardize column names to common format
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    mapping : dict, optional
        Custom column name mapping
        
    Returns
    -------
    pd.DataFrame with standardized column names
    """
    df = df.copy()
    
    # Default mappings
    default_mapping = {
        # Time columns
        'T': 'time',
        't': 'time',
        'duration': 'time',
        'survival_time': 'time',
        'survival': 'time',
        'week': 'time',
        'days': 'time',
        'months': 'time',
        
        # Event columns
        'E': 'event',
        'e': 'event',
        'status': 'event',
        'death': 'event',
        'observed': 'event',
        'arrest': 'event',
        'fail': 'event',
        'failed': 'event',
        'censored': 'event',
    }
    
    # Apply custom mapping if provided
    if mapping:
        default_mapping.update(mapping)
    
    # Rename columns
    df = df.rename(columns=default_mapping)
    
    return df


# Example usage function
def load_benchmark_dataset(source: str, dataset: str) -> pd.DataFrame:
    """
    Load a benchmark dataset from any supported library
    
    Parameters
    ----------
    source : str
        Source library ('lifelines', 'sksurv', 'pycox')
    dataset : str
        Name of the dataset
        
    Returns
    -------
    pd.DataFrame with standardized format
    """
    if source == 'lifelines':
        return load_from_lifelines(dataset)
    elif source == 'sksurv':
        return load_from_sksurv(dataset)
    elif source == 'pycox':
        return load_from_pycox(dataset)
    else:
        raise ValueError(f"Unknown source: {source}")