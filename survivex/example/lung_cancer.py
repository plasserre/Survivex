import pandas as pd
import numpy as np
import torch
from survivex.datasets.loaders import load_survival_dataset
from survivex.datasets.validators import validate_survival_data
from survivex.datasets.converters import auto_detect_format, load_from_lifelines


import pandas as pd
import numpy as np
from survivex.datasets.loaders import load_survival_dataset
from survivex.datasets.validators import validate_survival_data
from survivex.datasets.converters import (
    load_from_lifelines, 
    standardize_column_names,
    auto_detect_format
)


# test_lung_cancer_variations updated

def test_lung_cancer_variations():
    """Test lung cancer dataset with different column handling"""
    print("\n" + "="*60)
    print(" TESTING LUNG CANCER DATASET VARIATIONS")
    print("="*60)
    
    # Test 1: Load with automatic standardization
    print("\n1️⃣ Load from lifelines with auto-standardization...")
    df1 = load_from_lifelines('lung', standardize=True)
    print(f"Columns after standardization: {list(df1.columns)[:10]}...")
    
    # Test 2: Load raw and let loader handle it
    print("\n2️⃣ Load raw and let loader auto-detect...")
    from lifelines.datasets import load_lung
    df_raw = load_lung()
    print(f"Raw columns: {list(df_raw.columns)}")
    
    # Let loader auto-detect and convert WITH auto_fix
    print(df_raw.head())
    data2 = load_survival_dataset(
        df_raw,
        time_col=None,  # Auto-detect
        event_col=None,  # Auto-detect
        auto_fix=True,   # AUTO-FIX any issues
        verbose=True
    )
    print(f"Result: {data2}")
    
    # Test 3: Explicitly specify the correct columns
    print("\n3️⃣ Explicitly specify correct columns...")
    data3 = load_survival_dataset(
        df_raw,
        time_col='time',     # Use the actual time column
        event_col='status',  # Use the actual status column
        auto_fix=True,       # Auto-fix missing values
        verbose=True
    )
    print(f"Result: {data3}")
    
    print("\n All lung cancer variation tests passed!")
    return data2


def test_event_format_conversions():
    """Test various event column formats"""
    print("\n" + "="*60)
    print("🔄 TESTING EVENT FORMAT CONVERSIONS")
    print("="*60)
    
    test_cases = [
        # (name, event_values, description)
        ("Binary 0/1", [0, 1, 0, 1, 1], "Standard format"),
        ("Binary 1/2", [1, 2, 1, 2, 2], "Status format (1=censored, 2=event)"),
        ("Boolean", [True, False, True, False, True], "Boolean format"),
        ("Text Dead/Alive", ['Dead', 'Alive', 'Dead', 'Alive', 'Dead'], "Text format"),
        ("Text Yes/No", ['Yes', 'No', 'Yes', 'No', 'Yes'], "Yes/No format"),
        ("Multi-value", [0, 1, 2, 0, 1], "Competing risks format"),
    ]
    
    for name, event_values, description in test_cases:
        print(f"\n📊 Testing: {name} - {description}")
        print(f"   Input values: {event_values}")
        
        df = pd.DataFrame({
            'time': [100, 200, 150, 300, 250],
            'my_event_col': event_values,
            'feature1': [1, 2, 3, 4, 5]
        })
        
        try:
            data = load_survival_dataset(
                df,
                time_col='time',
                event_col='my_event_col',
                verbose=False
            )
            
            # Check the converted values
            converted_events = data.to_pandas()['event'].tolist()
            print(f"Converted to: {converted_events}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n✅ Event format conversion tests completed!")


def test_auto_detection():
    """Test auto-detection of columns"""
    print("\n" + "="*60)
    print("🔍 TESTING AUTO-DETECTION")
    print("="*60)
    
    # Test different column naming conventions
    test_datasets = [
        {
            'name': 'Standard naming',
            'data': pd.DataFrame({
                'time': [100, 200, 150],
                'event': [1, 0, 1],
                'age': [65, 70, 55]
            })
        },
        {
            'name': 'Alternative naming',
            'data': pd.DataFrame({
                'survival_time': [100, 200, 150],
                'death_status': [1, 2, 1],  # 1=censored, 2=death
                'patient_age': [65, 70, 55]
            })
        },
        {
            'name': 'Medical naming',
            'data': pd.DataFrame({
                'followup_days': [100, 200, 150],
                'alive': ['No', 'Yes', 'No'],  # No = dead
                'bmi': [25.5, 28.0, 24.0]
            })
        }
    ]
    
    for test in test_datasets:
        print(f"\n📊 Testing: {test['name']}")
        print(f"   Columns: {list(test['data'].columns)}")
        
        # Auto-detect format
        detected = auto_detect_format(test['data'])
        print(f"   Detected time columns: {detected['time_cols']}")
        print(f"   Detected event columns: {detected['event_cols']}")
        print(f"   Best guess - Time: '{detected['time_col_best']}', Event: '{detected['event_col_best']}'")
        
        # Try loading with auto-detection
        try:
            data = load_survival_dataset(
                test['data'],
                time_col=None,
                event_col=None,
                verbose=False
            )
            print(f"   ✅ Successfully loaded with auto-detection")
        except Exception as e:
            print(f"   ⚠️ Manual specification needed: {e}")
    
    print("\n✅ Auto-detection tests completed!")



def get_lung_cancer_from_lifelines():
    try:
        from lifelines.datasets import load_lung
        print("Loading lung cancer data from lifelines lib")
        df = load_lung()
        print(F"Lodead dataset shape: {df.shape}")
        print(f"Available Columns {list(df.columns)}")
        return df
    
    except ImportError:
        print("lifelines not installed. Install with: pip install lifelines")
        return None


def test_lung_cancer():
    print("Lung Cancer dataset loading....")
    print("-"*50)

    df = get_lung_cancer_from_lifelines()
    if df is None:
        print("Couldn't load the dataset")
        return
    print("loading data was successful.")

    print("-"*50)

    print("\n Testing validation...")
    validation = validate_survival_data(df, 'time', 'event')
    print("-"*50)


    print("Preparing data for survival analysis")
    survival_data = load_survival_dataset(df,'time', 'status')
    
    survival_df = survival_data.to_pandas()
    print(survival_df.head())
    #you can also pass the feature columns to the loader




if __name__ == "__main__":
    test_lung_cancer()
    test_lung_cancer_variations()
