import torch
import pandas as pd
import numpy as np
from survivex.core.data import SurvivalData
import pandas as pd
import numpy as np
from survivex.datasets.loaders import load_survival_dataset
from survivex.datasets.validators import validate_survival_data
from survivex.datasets.converters import auto_detect_format, load_from_lifelines

from survivex.example.lung_cancer import test_lung_cancer, test_lung_cancer_variations


def test_basic_creation():
    """Test basic SurvivalData creation"""
    print("🧪 Test 1: Basic creation")
    
    # Simple data
    times = [1, 3, 5, 7, 9]
    events = [1, 1, 0, 1, 0]
    
    data = SurvivalData(time=times, event=events)
    
    print(f" Created: {data}")
    print(f"   Length: {len(data)}")
    print(f"   Times: {data.time}")
    print(f"   Events: {data.event}")
    
    assert len(data) == 5
    assert data.event.sum() == 3  # 3 events
    print(" Basic creation test passed!\n")


def test_with_covariates():
    """Test SurvivalData with covariates"""
    print("🧪 Test 2: With covariates")
    
    times = [1, 3, 5, 7, 9]
    events = [1, 1, 0, 1, 0]
    X = [[65, 1], [70, 0], [55, 1], [60, 0], [75, 1]]  # age, sex
    feature_names = ['age', 'sex']
    
    data = SurvivalData(time=times, event=events, X=X, feature_names=feature_names)
    
    print(f" Created: {data}")
    print(f"   Features: {data.feature_names}")
    print(f"   X shape: {data.X.shape}")
    
    assert data.X.shape == (5, 2)
    assert data.feature_names == ['age', 'sex']
    print(" Covariates test passed!\n")


def test_pandas_conversion():
    """Test pandas conversion"""
    print("🧪 Test 3: Pandas conversion")
    
    times = [1, 3, 5, 7]
    events = [1, 1, 0, 1]
    X = [[65, 1], [70, 0], [55, 1], [60, 0]]
    feature_names = ['age', 'sex']
    
    data = SurvivalData(time=times, event=events, X=X, feature_names=feature_names)
    df = data.to_pandas()
    
    print(" Converted to pandas:")
    print(df)
    
    assert 'time' in df.columns
    assert 'event' in df.columns
    assert 'age' in df.columns
    assert 'sex' in df.columns
    assert len(df) == 4
    print(" Pandas conversion test passed!\n")


def test_from_pandas():
    """Test creating from pandas"""
    print("🧪 Test 4: From pandas DataFrame")
    
    df = pd.DataFrame({
        'time': [1, 3, 5, 7, 9],
        'event': [1, 1, 0, 1, 0],
        'age': [65, 70, 55, 60, 75],
        'sex': [1, 0, 1, 0, 1]
    })
    
    data = SurvivalData.from_pandas(df, feature_cols=['age', 'sex'])
    
    print(f" Created from pandas: {data}")
    print("Original DataFrame:")
    print(df.head(3))
    print("\nConverted back:")
    print(data.to_pandas().head(3))
    
    assert len(data) == 5
    assert data.feature_names == ['age', 'sex']
    print(" From pandas test passed!\n")


def test_validation():
    """Test data validation"""
    print("🧪 Test 5: Data validation")
    
    # Test mismatched lengths
    try:
        SurvivalData(time=[1, 2, 3], event=[1, 0])  # Different lengths
        assert False, "Should have raised error"
    except ValueError as e:
        print(f" Caught expected error: {e}")
    
    # Test negative times
    try:
        SurvivalData(time=[-1, 2, 3], event=[1, 0, 1])  # Negative time
        assert False, "Should have raised error"
    except ValueError as e:
        print(f" Caught expected error: {e}")
    
    # Test invalid events
    try:
        SurvivalData(time=[1, 2, 3], event=[1, 2, 0])  # Event = 2 (invalid)
        assert False, "Should have raised error"
    except ValueError as e:
        print(f" Caught expected error: {e}")
    
    print(" Validation tests passed!\n")


def test_tensor_types():
    """Test different input types"""
    print("🧪 Test 6: Different input types")
    
    # Test with lists
    data1 = SurvivalData(time=[1, 2, 3], event=[1, 0, 1])
    
    # Test with numpy arrays
    data2 = SurvivalData(time=np.array([1, 2, 3]), event=np.array([1, 0, 1]))
    
    # Test with torch tensors
    data3 = SurvivalData(time=torch.tensor([1, 2, 3]), event=torch.tensor([1, 0, 1]))
    
    # All should be equivalent
    assert torch.equal(data1.time, data2.time)
    assert torch.equal(data2.time, data3.time)
    assert torch.equal(data1.event, data2.event)
    
    print(" Different input types work correctly!")
    print(f"   List input: {data1}")
    print(f"   NumPy input: {data2}")  
    print(f"   Torch input: {data3}")
    print(" Tensor types test passed!\n")


# test_dataset_module.py
"""
Manual test script for the dataset module
"""





# test_custom_data.py
def test_custom_data():
    """Test with custom data formats"""
    print("\n" + "="*60)
    print("🧪 TESTING CUSTOM DATA FORMATS")
    print("="*60)
    
    # Test 1: Dictionary input with categorical
    print("\n1️⃣ Testing dictionary input...")
    data_dict = {
        'survival_days': [100, 200, 150, 300, 250],
        'death': ['Yes', 'No', 'Yes', 'Yes', 'No'],
        'age': [65, 70, 55, 60, 75],
        'treatment': ['A', 'B', 'A', 'B', 'A']
    }
    
    data1 = load_survival_dataset(
        data_dict,
        time_col='survival_days',
        event_col='death',
        handle_categorical='encode',  # Encode the treatment column
        verbose=True
    )
    print(f"Result: {data1}")
    
    # Test 2: Date range input (with updated freq)
    print("\n2️⃣ Testing date range input...")
    data_dates = {
        'start_date': pd.date_range('2020-01-01', periods=5, freq='ME'),  # Month End
        'end_date': pd.date_range('2020-06-01', periods=5, freq='ME'),    # Month End
        'status': [1, 0, 1, 1, 0],
        'score': [3.5, 4.2, 2.8, 3.9, 4.5]
    }
    
    data2 = load_survival_dataset(
        data_dates,
        time_col=['start_date', 'end_date'],
        event_col='status',
        verbose=True
    )
    print(f"Result: {data2}")
    
    print("\n All custom data tests completed!")
    return data1, data2


def test_converters_directly():
    """Test converters to understand the data structure"""
    print("\n" + "="*60)
    print("🔍 TESTING CONVERTERS DIRECTLY")
    print("="*60)
    
    # Test lifelines lung dataset
    print("\n📚 Testing lifelines lung dataset:")
    from lifelines.datasets import load_lung
    df_raw = load_lung()
    print(f"Raw columns: {list(df_raw.columns)}")
    print(f"Raw shape: {df_raw.shape}")
    print(f"First row:\n{df_raw.iloc[0]}")
    
    # Test our converter
    df_converted = load_from_lifelines('lung')
    print(f"\nConverted columns: {list(df_converted.columns)}")
    print(f"Converted shape: {df_converted.shape}")
    print(f"First row:\n{df_converted.iloc[0]}")
    
    # Check what's in time and event columns
    print(f"\nTime column ('time'): {df_converted['time'].iloc[:5].values}")
    print(f"Event column ('event'): {df_converted['event'].iloc[:5].values}")
    print(f"Event unique values: {df_converted['event'].unique()}")



def run_all_tests():
    """Run all tests"""
    print("🚀 Running Step 1 Tests: SurvivalData Class")
    print("=" * 50)
    
    test_basic_creation()
    test_with_covariates() 
    test_pandas_conversion()
    test_from_pandas()
    test_validation()
    test_tensor_types()
    data1, data2 = test_custom_data()
    test_lung_cancer()
    
    print("🎉 ALL STEP 1 TESTS PASSED!")
    print("=" * 50)
  


if __name__ == "__main__":
    run_all_tests()
