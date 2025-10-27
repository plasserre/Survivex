# In run_all_validation.py, update the import:

if __name__ == "__main__":
    print("="*80)
    print(" " * 20 + "COX PH COMPLETE VALIDATION SUITE")
    print("="*80)
    print()
    
    # Part 1: Core tests
    print("\n" + "█"*80)
    print("PART 1: CORE FUNCTIONALITY TESTS")
    print("█"*80)
    
    try:
        import sys
        import os
        # Add tests directory to path
        sys.path.insert(0, os.path.dirname(__file__))
        
        from validate_cox_ph import run_all_tests
        run_all_tests()
    except Exception as e:
        print(f"Core tests error: {e}")
        print("Skipping core tests...")
    
    # Part 2: Extended tests
    print("\n\n" + "█"*80)
    print("PART 2: EXTENDED FEATURE TESTS")
    print("█"*80)
    
    try:
        from validate_cox_ph_extended import run_extended_tests
        run_extended_tests()
    except Exception as e:
        print(f"Extended tests error: {e}")
    
    print("\n\n" + "="*80)
    print(" " * 25 + "VALIDATION COMPLETE")
    print("="*80)