# survivex/datasets/loaders.py
"""
Main dataset loader with integration of validators and converters
"""
import pandas as pd
import numpy as np
from typing import Union, List, Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime, date

from ..core.data import SurvivalData
from .validators import validate_survival_data, SurvivalDataValidator
from .converters import (
    load_from_lifelines, 
    load_from_sksurv,
    load_from_pycox,
    auto_detect_format,
    standardize_column_names
)


def load_survival_dataset(
    data: Union[str, pd.DataFrame, dict],
    time_col: Union[str, List[str]] = None,
    event_col: str = None,
    feature_cols: Optional[List[str]] = None,
    # Data source
    source: str = 'auto',  # 'auto', 'file', 'lifelines', 'sksurv', 'pycox'
    dataset_name: str = None,  # For library datasets
    # Event handling
    event_mapping: Optional[Dict[Any, int]] = None,
    # Missing values
    handle_missing: str = 'warn',  # 'warn', 'drop', 'impute', 'raise'
    impute_strategy: str = 'median',  # for numeric
    categorical_fill: str = 'mode',  # for categorical
    # Validation
    validate: bool = True,
    auto_fix: bool = False,
    # Categorical handling
    handle_categorical: str = 'encode',  # 'encode', 'drop', 'dummy'
    # Verbosity
    verbose: bool = False,
    **kwargs
) -> SurvivalData:
    """
    Universal survival dataset loader with automatic normalization.
    
    Parameters
    ----------
    data : str, DataFrame, dict, or tuple
        Input data source. Can be:
        - File path (CSV, Excel)
        - pandas DataFrame
        - Dictionary
        - Tuple of (source, dataset_name) for library datasets
    time_col : str or list of str, optional
        - Single string: column name for duration values
        - List of 2 strings: [start_date_col, end_date_col] for date calculation
        - None: auto-detect
    event_col : str, optional
        Column name for event indicators. None to auto-detect
    feature_cols : list, optional
        Feature column names. If None, auto-detects
    source : str
        Data source type ('auto', 'file', 'lifelines', 'sksurv', 'pycox')
    dataset_name : str
        Name of dataset when loading from library
    event_mapping : dict, optional
        Custom mapping for event values to 0/1
    handle_missing : str
        How to handle missing values ('warn', 'drop', 'impute', 'raise')
    impute_strategy : str
        Strategy for numeric imputation ('mean', 'median', 'forward_fill')
    categorical_fill : str
        Strategy for categorical imputation ('mode', 'missing_indicator')
    validate : bool
        Whether to validate the data
    auto_fix : bool
        Whether to automatically fix common issues
    verbose : bool
        Whether to print progress messages
        
    Returns
    -------
    SurvivalData
        Loaded and processed survival data
        
    Examples
    --------
    >>> # Load from CSV
    >>> data = load_survival_dataset('lung_cancer.csv', 'time', 'status')
    
    >>> # Load from lifelines
    >>> data = load_survival_dataset(source='lifelines', dataset_name='lung')
    
    >>> # Auto-detect format
    >>> data = load_survival_dataset('data.csv')
    """
    
    # Step 1: Load the raw data
    df = _load_raw_data(data, source, dataset_name, verbose, **kwargs)
    
    # Step 2: Process time column FIRST if it's a date range
    # This needs to happen BEFORE validation
    if isinstance(time_col, list):
        if verbose:
            print(f"📅 Processing date range columns: {time_col}")
        df = _process_time_column(df, time_col)
        # After processing, we now have a 'time' column
        actual_time_col = 'time'
    else:
        actual_time_col = time_col
    
    # Step 3: Auto-detect format if columns not specified
    if actual_time_col is None or event_col is None:
        detected = auto_detect_format(df)
        if verbose:
            print(f"\n🔍 Auto-detected format: {detected['format']}")
            
        if actual_time_col is None and detected['time_cols']:
            actual_time_col = detected['time_cols'][0]
            if verbose:
                print(f"   Using time column: '{actual_time_col}'")
                
        if event_col is None and detected['event_cols']:
            event_col = detected['event_cols'][0]
            if verbose:
                print(f"   Using event column: '{event_col}'")
    
    # Step 4: Validate (now with the correct time column)
    if validate:
        validator = SurvivalDataValidator(verbose=verbose)
        # Use actual_time_col which is now always a string
        validation_result = validator.validate_all(df, actual_time_col, event_col, feature_cols)
        
        if not validation_result['valid']:
            if auto_fix:
                df = _auto_fix_issues(df, validation_result['issues'], verbose)
            else:
                raise ValueError(
                    f"Validation failed with {len(validation_result['issues'])} issues. "
                    f"Set auto_fix=True to attempt automatic fixes."
                )
    
    # Step 5: Handle missing values
    df = _handle_missing_values(
        df, handle_missing, impute_strategy, categorical_fill, verbose
    )
    
    # Step 6: Process time column if not already done
    if not isinstance(time_col, list):
        df = _process_time_column(df, actual_time_col)
    
    # Step 7: Process event column
    df = _process_event_column(df, event_col, event_mapping)
    
    # Step 8: Select and validate features
    if isinstance(time_col, list):
        exclude_cols = set(time_col) | {event_col, 'time', 'event'}
    else:
        exclude_cols = {actual_time_col, event_col, 'time', 'event'}
    
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        if verbose:
            print(f"\n📊 Auto-detected {len(feature_cols)} feature columns")
    
    # Step 8.5: Handle categorical features
    df = _handle_categorical_features(df, feature_cols, handle_categorical, verbose)
    
    # Update feature_cols after categorical handling
    if handle_categorical == 'dummy':
        feature_cols = [col for col in df.columns if col not in {'time', 'event'}]
    
    # Step 9: Final validation
    final_cols = ['time', 'event'] + feature_cols
    df_final = df[final_cols].copy()
    
    if validate:
        final_validation = validate_survival_data(
            df_final, 
            'time',  # Always 'time' after processing
            'event',  # Always 'event' after processing
            feature_cols, 
            verbose=False
        )
        if not final_validation['valid']:
            if verbose:
                print(f"❌ Final validation issues: {final_validation['issues']}")
            if final_validation['issues']:
                critical_issues = [i for i in final_validation['issues'] 
                                 if 'Warning' not in i and 'warning' not in i.lower()]
                if critical_issues:
                    raise ValueError(f"Final validation failed: {critical_issues}")
    
    # Step 10: Create SurvivalData object
    if verbose:
        print(f"\n✅ Successfully loaded dataset:")
        print(f"   Shape: {df_final.shape}")
        print(f"   Events: {df_final['event'].sum()}/{len(df_final)} "
              f"({df_final['event'].mean():.1%})")
    
    return SurvivalData.from_pandas(
        df_final,
        time_col='time',
        event_col='event',
        feature_cols=feature_cols
    )


def _handle_categorical_features(df, feature_cols, handle_categorical, verbose):
    """Handle categorical features in the dataset"""
    
    if handle_categorical == 'drop':
        # Drop all non-numeric features
        numeric_features = []
        dropped_features = []
        
        for col in feature_cols:
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_features.append(col)
                else:
                    dropped_features.append(col)
                    df = df.drop(columns=[col])
        
        if dropped_features and verbose:
            print(f"   Dropped {len(dropped_features)} categorical features: {dropped_features}")
        
    elif handle_categorical == 'encode':
        # Label encode categorical features
        for col in feature_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    from sklearn.preprocessing import LabelEncoder
                except ImportError:
                    raise ImportError(
                        "scikit-learn required for categorical encoding. "
                        "Install with: pip install scikit-learn\n"
                        "Or use handle_categorical='drop' to skip categorical features."
                    )
                if verbose:
                    print(f"   Encoding categorical feature '{col}'")
                
                le = LabelEncoder()
                # Handle missing values
                mask = df[col].notna()
                df.loc[mask, col] = le.fit_transform(df.loc[mask, col])
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    elif handle_categorical == 'dummy':
        # One-hot encode categorical features
        categorical_cols = [col for col in feature_cols 
                           if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])]
        
        if categorical_cols:
            if verbose:
                print(f"   Creating dummy variables for: {categorical_cols}")
            
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df


def _load_raw_data(data, source, dataset_name, verbose, **kwargs):
    """Load raw data from various sources"""
    
    # Auto-detect source type
    if source == 'auto':
        if isinstance(data, str):
            if Path(data).exists():
                source = 'file'
            else:
                # Might be a dataset name
                source = 'lifelines'
                dataset_name = data
        elif isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        elif isinstance(data, tuple) and len(data) == 2:
            source, dataset_name = data
    
    # Load based on source
    if source == 'file':
        file_path = Path(data)
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(data, **kwargs)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(data, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        if verbose:
            print(f" Loaded from {file_path.name}")
    
    elif source == 'lifelines':
        df = load_from_lifelines(dataset_name)
        
    elif source == 'sksurv':
        df = load_from_sksurv(dataset_name)
        
    elif source == 'pycox':
        df = load_from_pycox(dataset_name)
        
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
        
    elif isinstance(data, dict):
        df = pd.DataFrame(data)
        
    else:
        raise ValueError(f"Cannot load data from source: {source}")
    
    return df


def _handle_missing_values(df, handle_missing, impute_strategy, categorical_fill, verbose):
    """Handle missing values in the dataset"""
    
    missing_summary = df.isnull().sum()
    if not missing_summary.any():
        return df
    
    if verbose:
        print("\n⚠️ Missing Values Detected:")
        print(missing_summary[missing_summary > 0])
    
    if handle_missing == 'raise':
        raise ValueError("Missing values found. Set handle_missing='drop' or 'impute'")
        
    elif handle_missing == 'drop':
        original_len = len(df)
        df = df.dropna()
        if verbose:
            print(f"   Dropped {original_len - len(df)} rows with missing values")
            
    elif handle_missing == 'impute':
        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    if impute_strategy == 'median':
                        df[col].fillna(df[col].median(), inplace=True)
                    elif impute_strategy == 'mean':
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif impute_strategy == 'forward_fill':
                        df[col].fillna(method='ffill', inplace=True)
                else:
                    if categorical_fill == 'mode' and len(df[col].mode()) > 0:
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    else:
                        df[col].fillna('missing', inplace=True)
        
        if verbose:
            print(f"   Imputed missing values using {impute_strategy}/{categorical_fill}")
    
    return df


def _process_time_column(df, time_col):
    """Process time column(s) to create standardized 'time' column"""
    
    if isinstance(time_col, list):
        if len(time_col) != 2:
            raise ValueError("time_col list must contain exactly 2 columns [start, end]")
        
        start_col, end_col = time_col
        if start_col not in df.columns or end_col not in df.columns:
            raise ValueError(f"Time columns not found: {time_col}")
        
        # Calculate duration from dates
        start_dates = pd.to_datetime(df[start_col])
        end_dates = pd.to_datetime(df[end_col])
        df['time'] = (end_dates - start_dates).dt.days
        
    else:
        if time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found")
        
        df['time'] = pd.to_numeric(df[time_col], errors='coerce')
    
    # Validate time values
    if (df['time'] <= 0).any():
        invalid_count = (df['time'] <= 0).sum()
        df = df[df['time'] > 0].copy()
        print(f"⚠️ Removed {invalid_count} rows with non-positive time values")
    
    return df


# In survivex/datasets/loaders.py, update the _process_event_column function:

def _process_event_column(df, event_col, event_mapping):
    """Process event column to create standardized 'event' column"""
    
    if event_col not in df.columns:
        # Try to find a column that might be the event column
        event_keywords = ['event', 'status', 'death', 'fail', 'censor', 'observed']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in event_keywords):
                print(f"⚠️ Event column '{event_col}' not found, using '{col}' instead")
                event_col = col
                break
        else:
            raise ValueError(f"Event column '{event_col}' not found and couldn't auto-detect")
    
    # Get unique values to understand the format
    unique_vals = df[event_col].dropna().unique()
    print(f"🔍 Event column '{event_col}' has values: {unique_vals}")
    
    # Normalize event values
    if event_mapping:
        df['event'] = df[event_col].map(event_mapping)
    else:
        # Auto-convert based on unique values
        unique_set = set(unique_vals)
        
        if unique_set.issubset({0, 1}):
            df['event'] = df[event_col].astype(int)
        elif unique_set.issubset({True, False}):
            df['event'] = df[event_col].astype(int)
        elif unique_set.issubset({1, 2}):
            # Common format: 1=censored, 2=event
            df['event'] = (df[event_col] == 2).astype(int)
            print(f"   Converted status (1=censored, 2=event) to event (0=censored, 1=event)")
        elif unique_set.issubset({0, 1, 2}):
            # Might be competing risks
            df['event'] = (df[event_col] > 0).astype(int)
            print(f"   Converted multi-value to binary (0=censored, >0=event)")
        elif unique_set.issubset({'Dead', 'Alive', 'dead', 'alive'}):
            df['event'] = df[event_col].str.lower().map({'dead': 1, 'alive': 0})
        elif unique_set.issubset({'Yes', 'No', 'yes', 'no'}):
            df['event'] = df[event_col].str.capitalize().map({'Yes': 1, 'No': 0})
        elif unique_set.issubset({'death', 'alive'}):
            df['event'] = df[event_col].map({'death': 1, 'alive': 0})
        else:
            # Try to convert to numeric
            df['event'] = pd.to_numeric(df[event_col], errors='coerce')
            
            # Check if conversion worked
            if df['event'].isna().all():
                raise ValueError(
                    f"Cannot convert event values to 0/1. Found: {unique_vals}. "
                    f"Please provide event_mapping."
                )
            
            # If numeric but not 0/1, apply a rule
            event_vals = df['event'].dropna().unique()
            if not set(event_vals).issubset({0, 1}):
                print(f"   Event values {event_vals} not in 0/1 format")
                if min(event_vals) >= 0:
                    # Assume 0 is censored, anything else is event
                    df['event'] = (df['event'] != 0).astype(int)
                    print(f"   Converted to: 0=censored, non-zero=event")
    
    # Ensure event is 0/1 integer
    df['event'] = df['event'].astype(int)
    
    # Validate
    if not set(df['event'].dropna().unique()).issubset({0, 1}):
        raise ValueError(f"Event column must be 0/1 after conversion. Got: {df['event'].unique()}")
    
    return df


def _auto_fix_issues(df, issues, verbose):
    """Attempt to automatically fix common issues"""
    
    if verbose:
        print("\n🔧 Attempting to auto-fix issues...")
    
    for issue in issues:
        if "non-positive time values" in issue:
            df = df[df['time'] > 0].copy()
            if verbose:
                print(f"   ✓ Removed rows with non-positive times")
                
        elif "Missing" in issue and "time values" in issue:
            df = df.dropna(subset=['time'])
            if verbose:
                print(f"   ✓ Removed rows with missing time values")
                
        elif "Missing" in issue and "event values" in issue:
            df = df.dropna(subset=['event'])
            if verbose:
                print(f"   ✓ Removed rows with missing event values")
    
    return df