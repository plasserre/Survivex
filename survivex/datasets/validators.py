# survivex/datasets/validators.py
"""
Data validation utilities for survival analysis datasets
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class SurvivalDataValidator:
    """Comprehensive validation for survival datasets"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.issues = []
        self.warnings = []
        
    def validate_all(self, 
                     df: pd.DataFrame, 
                     time_col: str = 'time',
                     event_col: str = 'event',
                     feature_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run all validation checks
        
        Returns
        -------
        Dict with keys:
            - valid: bool
            - issues: List of critical issues
            - warnings: List of warnings
            - summary: Dict of data statistics
        """
        self.issues = []
        self.warnings = []
        
        # Check required columns exist
        self._validate_columns_exist(df, time_col, event_col)
        
        # Validate time values
        self._validate_time_values(df, time_col)
        
        # Validate event values
        self._validate_event_values(df, event_col)
        
        # Check for missing values
        self._check_missing_values(df, time_col, event_col, feature_cols)
        
        # Check data quality
        self._check_data_quality(df, time_col, event_col)
        
        # Check statistical properties
        summary = self._compute_summary_statistics(df, time_col, event_col)
        
        # Print report if verbose
        if self.verbose:
            self._print_validation_report(summary)
        
        return {
            'valid': len(self.issues) == 0,
            'issues': self.issues,
            'warnings': self.warnings,
            'summary': summary
        }
    
    def _validate_columns_exist(self, df: pd.DataFrame, time_col: str, event_col: str):
        """Check if required columns exist"""
        if time_col not in df.columns:
            self.issues.append(f"Time column '{time_col}' not found in data")
        if event_col not in df.columns:
            self.issues.append(f"Event column '{event_col}' not found in data")
    
    def _validate_time_values(self, df: pd.DataFrame, time_col: str):
        """Validate time column values"""
        if time_col not in df.columns:
            return
            
        times = pd.to_numeric(df[time_col], errors='coerce')
        
        # Check for non-numeric values
        if times.isna().any():
            non_numeric_count = times.isna().sum()
            self.issues.append(f"Found {non_numeric_count} non-numeric time values")
        
        # Check for negative or zero times
        if (times <= 0).any():
            invalid_count = (times <= 0).sum()
            self.issues.append(f"Found {invalid_count} non-positive time values (times must be > 0)")
        
        # Check for infinite values
        if np.isinf(times).any():
            self.issues.append(f"Found {np.isinf(times).sum()} infinite time values")
        
        # Warn about potential outliers
        if len(times) > 0 and not times.isna().all():
            q99 = times.quantile(0.99)
            q95 = times.quantile(0.95)
            max_time = times.max()
            
            if max_time > 10 * q99:
                self.warnings.append(
                    f"Potential outlier: max time {max_time:.1f} is >10x the 99th percentile ({q99:.1f})"
                )
            elif max_time > 5 * q95:
                self.warnings.append(
                    f"Large time range: max time {max_time:.1f} is >5x the 95th percentile ({q95:.1f})"
                )
    
    def _validate_event_values(self, df: pd.DataFrame, event_col: str):
        """Validate event column values"""
        if event_col not in df.columns:
            return
            
        events = df[event_col]
        unique_vals = events.dropna().unique()
        
        # Check if values can be converted to 0/1
        valid_binary = set(unique_vals).issubset({0, 1, True, False, 'Yes', 'No', 'Dead', 'Alive'})
        
        if not valid_binary:
            # Check if it might be competing risks (multiple event types)
            if all(isinstance(v, (int, float)) for v in unique_vals):
                if min(unique_vals) >= 0 and max(unique_vals) <= 10:
                    self.warnings.append(
                        f"Event column has values {sorted(unique_vals)} - might be competing risks?"
                    )
                else:
                    self.issues.append(
                        f"Event values must be binary (0/1) or convertible. Found: {unique_vals}"
                    )
            else:
                self.issues.append(
                    f"Cannot interpret event values: {unique_vals}"
                )
    
    def _check_missing_values(self, df: pd.DataFrame, time_col: str, event_col: str, 
                              feature_cols: Optional[List[str]] = None):
        """Check for missing values"""
        # Critical columns
        if time_col in df.columns:
            time_missing = df[time_col].isna().sum()
            if time_missing > 0:
                self.issues.append(f"Missing {time_missing} time values (critical)")
        
        if event_col in df.columns:
            event_missing = df[event_col].isna().sum()
            if event_missing > 0:
                self.issues.append(f"Missing {event_missing} event values (critical)")
        
        # Feature columns
        if feature_cols:
            for col in feature_cols:
                if col in df.columns:
                    missing = df[col].isna().sum()
                    if missing > 0:
                        pct = missing / len(df) * 100
                        if pct > 50:
                            self.warnings.append(f"Column '{col}' has {pct:.1f}% missing values")
                        elif pct > 20:
                            self.warnings.append(f"Column '{col}' has {pct:.1f}% missing values")
    
    def _check_data_quality(self, df: pd.DataFrame, time_col: str, event_col: str):
        """Check general data quality issues"""
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            self.warnings.append(f"Found {duplicates} duplicate rows")
        
        # Check for very small dataset
        if len(df) < 30:
            self.warnings.append(f"Small dataset ({len(df)} samples) - results may be unreliable")
        
        # Check event rate
        if event_col in df.columns and time_col in df.columns:
            events = pd.to_numeric(df[event_col], errors='coerce')
            if not events.isna().all():
                event_rate = events.mean()
                
                if event_rate < 0.05:
                    self.warnings.append(f"Very low event rate ({event_rate:.1%}) - may cause estimation issues")
                elif event_rate < 0.10:
                    self.warnings.append(f"Low event rate ({event_rate:.1%})")
                elif event_rate > 0.95:
                    self.warnings.append(f"Very high event rate ({event_rate:.1%}) - limited censoring")
                elif event_rate > 0.90:
                    self.warnings.append(f"High event rate ({event_rate:.1%})")
    
    def _compute_summary_statistics(self, df: pd.DataFrame, time_col: str, event_col: str) -> Dict:
        """Compute summary statistics"""
        summary = {
            'n_samples': len(df),
            'n_features': len(df.columns) - 2,  # Exclude time and event
            'n_duplicates': df.duplicated().sum(),
        }
        
        if time_col in df.columns:
            times = pd.to_numeric(df[time_col], errors='coerce')
            summary.update({
                'time_min': times.min(),
                'time_max': times.max(),
                'time_median': times.median(),
                'time_mean': times.mean(),
                'time_std': times.std(),
            })
        
        if event_col in df.columns:
            events = pd.to_numeric(df[event_col], errors='coerce')
            if not events.isna().all():
                summary.update({
                    'n_events': int(events.sum()),
                    'n_censored': int((1 - events).sum()),
                    'event_rate': events.mean(),
                })
        
        return summary
    
    def _print_validation_report(self, summary: Dict):
        """Print validation report"""
        print("\n" + "="*60)
        print("SURVIVAL DATA VALIDATION REPORT")
        print("="*60)
        
        # Data summary
        print("\n Data Summary:")
        print(f"  • Samples: {summary.get('n_samples', 'N/A')}")
        print(f"  • Features: {summary.get('n_features', 'N/A')}")
        print(f"  • Events: {summary.get('n_events', 'N/A')} ({summary.get('event_rate', 0):.1%})")
        print(f"  • Censored: {summary.get('n_censored', 'N/A')}")
        
        if 'time_min' in summary:
            print(f"\n Time Statistics:")
            print(f"  • Range: [{summary['time_min']:.1f}, {summary['time_max']:.1f}]")
            print(f"  • Median: {summary['time_median']:.1f}")
            print(f"  • Mean ± SD: {summary['time_mean']:.1f} ± {summary['time_std']:.1f}")
        
        # Issues and warnings
        if self.issues:
            print(f"\nXX Critical Issues ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  • {issue}")
        else:
            print("\n No critical issues found")
        
        if self.warnings:
            print(f"\n⚠️ Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
        else:
            print("\n No warnings")
        
        print("\n" + "="*60)
        
        if self.issues:
            print(" VALIDATION FAILED - Fix critical issues before proceeding")
        else:
            print(" VALIDATION PASSED - Data is ready for analysis")
        print("="*60 + "\n")


def validate_survival_data(df: pd.DataFrame, 
                          time_col: str = 'time',
                          event_col: str = 'event',
                          feature_cols: Optional[List[str]] = None,
                          verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function to validate survival data
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to validate
    time_col : str
        Name of time column
    event_col : str
        Name of event column  
    feature_cols : list, optional
        Names of feature columns
    verbose : bool
        Whether to print validation report
        
    Returns
    -------
    Dict with validation results
    """
    validator = SurvivalDataValidator(verbose=verbose)
    return validator.validate_all(df, time_col, event_col, feature_cols)