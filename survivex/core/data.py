

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import torch


@dataclass
class SurvivalData:
    """
    Core survival data structure.
    
    Parameters
    ----------
    time : array-like
        Observed survival times
    event : array-like  
        Event indicators (1=event, 0=censored)
    X : array-like, optional
        Covariate matrix
    feature_names : list, optional
        Names of features/covariates
    
    Examples
    --------
    >>> # Simple example
    >>> times = [1, 3, 5, 7, 9]
    >>> events = [1, 1, 0, 1, 0]  # 1=event, 0=censored
    >>> data = SurvivalData(time=times, event=events)
    >>> print(f"Dataset size: {len(data)}")
    >>> print(f"Events: {data.event.sum()}/{len(data)}")
    
    >>> # With covariates
    >>> X = [[65, 1], [70, 0], [55, 1], [60, 0], [75, 1]]  # age, sex
    >>> data = SurvivalData(time=times, event=events, X=X, 
    ...                     feature_names=['age', 'sex'])
    >>> df = data.to_pandas()
    >>> print(df.head())
    """
    time: torch.Tensor
    event: torch.Tensor
    X: Optional[torch.Tensor] = None
    feature_names: Optional[List[str]] = None
    


    def __post_init__(self):
        """Convert inputs to tensors and validate"""
        # Convert to tensors
        self.time = self._to_tensor(self.time, torch.float64)
        self.event = self._to_tensor(self.event, torch.int64)
        
        if self.X is not None:
            self.X = self._to_tensor(self.X, torch.float64)
            
        # Validate data
        self._validate()
    
    def _to_tensor(self, data, dtype):
        """Convert data to tensor"""
        if not isinstance(data, torch.Tensor):
            if isinstance(data, (pd.Series, pd.DataFrame)):
                data = data.values
            data = torch.tensor(data, dtype=dtype)
        return data.to(dtype)
    
    def _validate(self):
        """Validate data consistency"""
        if len(self.time) != len(self.event):
            raise ValueError(f"time and event must have same length: {len(self.time)} vs {len(self.event)}")
        
        if torch.any(self.time <= 0):
            raise ValueError("All survival times must be positive")
            
        if self.X is not None and len(self.X) != len(self.time):
            raise ValueError(f"Covariates X must match time/event length: {len(self.X)} vs {len(self.time)}")
        
        # Check events are 0 or 1
        if not torch.all((self.event == 0) | (self.event == 1)):
            raise ValueError("Event indicators must be 0 (censored) or 1 (event)")
    
    def __len__(self):
        """Return number of observations"""
        return len(self.time)
    
    def __repr__(self):
        """String representation"""
        n_events = self.event.sum().item()
        event_rate = n_events / len(self) * 100
        
        repr_str = f"SurvivalData(n_obs={len(self)}, n_events={n_events}, event_rate={event_rate:.1f}%"
        
        if self.X is not None:
            repr_str += f", n_features={self.X.shape[1]}"
        
        repr_str += ")"
        return repr_str
    
    def to_pandas(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        df = pd.DataFrame({
            'time': self.time.numpy(),
            'event': self.event.numpy()
        })
        
        if self.X is not None:
            if self.feature_names:
                for i, name in enumerate(self.feature_names):
                    df[name] = self.X[:, i].numpy()
            else:
                for i in range(self.X.shape[1]):
                    df[f'X{i}'] = self.X[:, i].numpy()
        
        return df
    




    # survivex/core/data.py - Update the from_pandas method

    @classmethod
    def from_pandas(cls, 
                df: pd.DataFrame,
                time_col: str = 'time',
                event_col: str = 'event',
                feature_cols: Optional[List[str]] = None) -> 'SurvivalData':
        """
        Create SurvivalData from pandas DataFrame
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        time_col : str
            Name of time column
        event_col : str
            Name of event column
        feature_cols : list, optional
            Names of feature columns
            
        Returns
        -------
        SurvivalData object
        """
        # Extract time and event
        time = df[time_col].values
        event = df[event_col].values
        
        # Extract features
        if feature_cols:
            # Filter to only numeric features
            numeric_features = []
            for col in feature_cols:
                if col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        numeric_features.append(col)
                    else:
                        print(f"⚠️ Skipping non-numeric feature '{col}' - encode it first")
            
            if numeric_features:
                X = df[numeric_features].values
                feature_names = numeric_features
            else:
                X = None
                feature_names = None
        else:
            # Auto-detect numeric features
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col not in [time_col, event_col]]
            
            if feature_cols:
                X = df[feature_cols].values
                feature_names = feature_cols
            else:
                X = None
                feature_names = None
        
        return cls(time=time, event=event, X=X, feature_names=feature_names)