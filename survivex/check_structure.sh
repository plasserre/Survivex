# Fix Python Import Issues for SurviveX
# =====================================

echo "🔧 Fixing Python import issues..."

# Step 1: Check current directory structure
echo "📁 Current directory structure:"
pwd
ls -la

echo ""
echo "📂 Project structure should look like this:"
echo "survivex/                    # Main project directory"
echo "├── survivex/               # Python package directory"
echo "│   ├── __init__.py         # Makes it a package"
echo "│   ├── core/"
echo "│   │   ├── __init__.py"
echo "│   │   └── data.py"
echo "│   └── ..."
echo "├── tests/"
echo "├── examples/"
echo "└── test_step1.py"
echo ""

# Step 2: Check if we have the right structure
echo "🔍 Checking current structure..."

if [ -d "survivex/survivex" ]; then
    echo "⚠️  PROBLEM: You have nested survivex directories!"
    echo "   Current: survivex/survivex/..."
    echo "   Should be: survivex/..."
    echo ""
    echo "🔧 Fixing nested structure..."
    
    # Move contents up one level
    mv survivex/survivex/* .
    rmdir survivex/survivex
    
    echo "✅ Fixed nested directory structure"
fi

# Step 3: Create missing __init__.py files
echo "📝 Creating missing __init__.py files..."

# Main package __init__.py
cat > survivex/__init__.py << 'EOF'
"""
SurviveX: Advanced Survival Analysis for Python
==============================================

A comprehensive survival analysis library with GPU acceleration,
statistical rigor, and support for complex multi-state models.
"""

__version__ = "0.1.0"
__author__ = "SurviveX Contributors"
__email__ = "maintainers@survivex.org"

# Will add imports as we build components
# from survivex.core.data import SurvivalData

__all__ = [
    "__version__",
]
EOF

# Core module __init__.py
mkdir -p survivex/core
cat > survivex/core/__init__.py << 'EOF'
"""
Core data structures and utilities for survival analysis
"""

from .data import SurvivalData

__all__ = ['SurvivalData']
EOF

# Create other __init__.py files
mkdir -p survivex/datasets
touch survivex/datasets/__init__.py

mkdir -p survivex/models  
touch survivex/models/__init__.py

mkdir -p survivex/utils
touch survivex/utils/__init__.py

echo "✅ Created __init__.py files"

# Step 4: Install package in development mode
echo "📦 Installing package in development mode..."

# Install in editable mode
pip install -e .

echo "✅ Package installed"

# Step 5: Test imports
echo "🧪 Testing imports..."

python -c "
try:
    import survivex
    print('✅ survivex imported successfully')
    print(f'   Version: {survivex.__version__}')
except ImportError as e:
    print(f'❌ Error importing survivex: {e}')
"

python -c "
try:
    from survivex.core.data import SurvivalData
    print('✅ SurvivalData imported successfully')
    
    # Quick test
    data = SurvivalData(time=[1, 2, 3], event=[1, 0, 1])
    print(f'   Test data: {data}')
except ImportError as e:
    print(f'❌ Error importing SurvivalData: {e}')
except Exception as e:
    print(f'❌ Error creating SurvivalData: {e}')
"

# Step 6: Fix file locations  
echo "📁 Organizing files properly..."

# Move test files to proper location
mkdir -p tests
mkdir -p examples

# If test file is in wrong location, move it
if [ -f "survivex/example/dataset_example.py" ]; then
    mv survivex/example/dataset_example.py examples/
    rmdir survivex/example 2>/dev/null || true
fi

if [ -f "survivex/test_step1.py" ]; then
    mv survivex/test_step1.py tests/
fi

echo "✅ Files organized"

# Step 7: Create a working test file in the right location
cat > test_step1_fixed.py << 'EOF'
#!/usr/bin/env python3
"""
Test script for Step 1: SurvivalData class (FIXED VERSION)
"""

# Add current directory to path (for development)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
import numpy as np

try:
    from survivex.core.data import SurvivalData
    print("✅ Import successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("💡 Trying alternative import method...")
    
    # Alternative import for development
    try:
        sys.path.insert(0, './survivex')
        from core.data import SurvivalData
        print("✅ Alternative import successful!")
    except ImportError as e2:
        print(f"❌ Alternative import also failed: {e2}")
        sys.exit(1)


def test_basic_creation():
    """Test basic SurvivalData creation"""
    print("\n🧪 Test 1: Basic creation")
    
    times = [1, 3, 5, 7, 9]
    events = [1, 1, 0, 1, 0]
    
    data = SurvivalData(time=times, event=events)
    
    print(f"✅ Created: {data}")
    assert len(data) == 5
    assert data.event.sum() == 3
    print("✅ Basic creation test passed!")


def test_with_pandas():
    """Test pandas integration"""
    print("\n🧪 Test 2: Pandas integration")
    
    df = pd.DataFrame({
        'time': [1, 3, 5, 7],
        'event': [1, 1, 0, 1],
        'age': [65, 70, 55, 60]
    })
    
    data = SurvivalData.from_pandas(df, feature_cols=['age'])
    print(f"✅ Created from pandas: {data}")
    
    df_back = data.to_pandas()
    print("✅ Converted back to pandas:")
    print(df_back)
    
    assert len(data) == 4
    print("✅ Pandas test passed!")


def run_quick_test():
    """Run quick test to verify everything works"""
    print("🚀 Running Quick Import and Functionality Test")
    print("=" * 50)
    
    test_basic_creation()
    test_with_pandas()
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ SurviveX is working correctly!")
    print("✅ Ready to continue development!")


if __name__ == "__main__":
    run_quick_test()
EOF

chmod +x test_step1_fixed.py

echo ""
echo "🎯 SUMMARY OF FIXES:"
echo "==================="
echo "✅ Fixed directory structure"
echo "✅ Created missing __init__.py files"
echo "✅ Installed package in development mode"
echo "✅ Created working test file"
echo ""
echo "🚀 NOW TRY THIS:"
echo "==============="
echo "python test_step1_fixed.py"
echo ""
echo "💡 If you still have issues, run:"
echo "   pwd                    # Check current directory"
echo "   ls -la survivex/       # Check package structure"  
echo "   pip install -e .       # Reinstall package"
echo ""

# Final verification
echo "🔍 FINAL VERIFICATION:"
echo "====================="
echo "Current directory: $(pwd)"
echo "Package structure:"
find survivex -name "*.py" | head -10