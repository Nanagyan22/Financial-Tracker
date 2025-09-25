import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from fuzzywuzzy import fuzz, process
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Comprehensive data preprocessing pipeline."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.entity_mappings = {}
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values with appropriate strategies.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        # Numeric columns - use median and handle infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Replace infinite values with NaN first
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            if df[col].isnull().sum() > 0:
                if col not in self.imputers:
                    self.imputers[col] = SimpleImputer(strategy='median')
                    df[col] = self.imputers[col].fit_transform(df[[col]]).flatten()
                else:
                    df[col] = self.imputers[col].transform(df[[col]]).flatten()
        
        # Categorical columns - use mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                if col not in self.imputers:
                    self.imputers[col] = SimpleImputer(strategy='most_frequent')
                    df[col] = self.imputers[col].fit_transform(df[[col]]).flatten()
                else:
                    df[col] = self.imputers[col].transform(df[[col]]).flatten()
        
        return df
    
    def standardize_entity_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize entity names using fuzzy matching.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized entity names
        """
        if 'entity_name' not in df.columns:
            return df
        
        df = df.copy()
        
        # Get unique entity names
        unique_entities = df['entity_name'].dropna().unique()
        
        # Create standardized mapping
        standardized_entities = {}
        processed_entities = set()
        
        for entity in unique_entities:
            if entity in processed_entities:
                continue
                
            # Find similar entities
            similar_entities = [entity]
            
            for other_entity in unique_entities:
                if other_entity != entity and other_entity not in processed_entities:
                    similarity = fuzz.token_sort_ratio(entity.lower(), other_entity.lower())
                    if similarity >= 85:  # High similarity threshold
                        similar_entities.append(other_entity)
            
            # Choose the most common or longest name as standard
            standard_name = max(similar_entities, key=len)
            
            for similar_entity in similar_entities:
                standardized_entities[similar_entity] = standard_name
                processed_entities.add(similar_entity)
        
        # Apply standardization
        df['entity_name'] = df['entity_name'].map(standardized_entities).fillna(df['entity_name'])
        self.entity_mappings = standardized_entities
        
        return df
    
    def create_entity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on entity characteristics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional entity features
        """
        df = df.copy()
        
        if 'entity_name' not in df.columns:
            return df
        
        # Entity name length (proxy for complexity/size)
        df['entity_name_length'] = df['entity_name'].str.len()
        
        # Entity type indicators based on name keywords
        type_keywords = {
            'university': ['university', 'college', 'school'],
            'hospital': ['hospital', 'clinic', 'health'],
            'commission': ['commission', 'authority', 'board'],
            'company': ['company', 'limited', 'ltd', 'corporation'],
            'ministry': ['ministry', 'department'],
            'fund': ['fund', 'trust']
        }
        
        for entity_type, keywords in type_keywords.items():
            df[f'is_{entity_type}'] = df['entity_name'].str.lower().str.contains(
                '|'.join(keywords), na=False
            ).astype(int)
        
        return df
    
    def scale_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numeric features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with scaled numeric features
        """
        df = df.copy()
        
        # Identify numeric columns to scale (exclude binary flags and IDs)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_scale = []
        
        for col in numeric_cols:
            # Skip binary flags, year, and ID columns
            if (col.startswith('is_') or 
                col in ['year', 'audit_flag', 'entity_id'] or
                df[col].nunique() <= 2):
                continue
            cols_to_scale.append(col)
        
        # Scale identified columns
        for col in cols_to_scale:
            if col not in self.scalers:
                self.scalers[col] = StandardScaler()
                df[col] = self.scalers[col].fit_transform(df[[col]]).flatten()
            else:
                df[col] = self.scalers[col].transform(df[[col]]).flatten()
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical features
        """
        df = df.copy()
        
        # Get categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col == 'entity_name':
                # Special handling for entity names - create entity ID
                if 'entity_name' not in self.encoders:
                    self.encoders['entity_name'] = LabelEncoder()
                    df['entity_id'] = self.encoders['entity_name'].fit_transform(df[col])
                else:
                    df['entity_id'] = self.encoders['entity_name'].transform(df[col])
                continue
            
            # For other categorical columns with few unique values, use one-hot encoding
            unique_values = df[col].nunique()
            
            if unique_values <= 10:  # One-hot encode if few categories
                if col not in self.encoders:
                    self.encoders[col] = OneHotEncoder(sparse_output=False, drop='first')
                    encoded = self.encoders[col].fit_transform(df[[col]])
                    
                    # Get feature names
                    feature_names = [f"{col}_{cat}" for cat in self.encoders[col].categories_[0][1:]]
                    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
                    df = pd.concat([df, encoded_df], axis=1)
                else:
                    encoded = self.encoders[col].transform(df[[col]])
                    feature_names = [f"{col}_{cat}" for cat in self.encoders[col].categories_[0][1:]]
                    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
                    df = pd.concat([df, encoded_df], axis=1)
                
                # Drop original column
                df = df.drop(columns=[col])
            
            else:  # Label encode if many categories
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[col] = self.encoders[col].fit_transform(df[col])
                else:
                    df[col] = self.encoders[col].transform(df[col])
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features for time series analysis.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with lag features
        """
        if 'year' not in df.columns or 'entity_name' not in df.columns:
            return df
        
        df = df.copy()
        df = df.sort_values(['entity_name', 'year'])
        
        # Financial columns for lag features
        financial_cols = ['revenue', 'expenditure', 'total_assets', 'total_liabilities']
        
        for col in financial_cols:
            if col in df.columns:
                # Previous year value
                df[f'{col}_lag1'] = df.groupby('entity_name')[col].shift(1)
                
                # Year-over-year change
                df[f'{col}_yoy_change'] = df[col] - df[f'{col}_lag1']
                df[f'{col}_yoy_pct_change'] = (df[col] - df[f'{col}_lag1']) / df[f'{col}_lag1']
                
                # Replace infinite values
                df[f'{col}_yoy_pct_change'] = df[f'{col}_yoy_pct_change'].replace([np.inf, -np.inf], 0)
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Detect and handle outliers.
        
        Args:
            df: Input DataFrame
            method: Method for outlier detection ('iqr', 'zscore')
            
        Returns:
            DataFrame with outlier flags
        """
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_flags = []
        
        for col in numeric_cols:
            if df[col].nunique() <= 2:  # Skip binary columns
                continue
                
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                df[f'{col}_outlier'] = outliers.astype(int)
                outlier_flags.append(f'{col}_outlier')
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > 3
                df[f'{col}_outlier'] = outliers.astype(int)
                outlier_flags.append(f'{col}_outlier')
        
        # Overall outlier flag
        if outlier_flags:
            df['total_outlier_count'] = df[outlier_flags].sum(axis=1)
        
        return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main preprocessing pipeline.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    print("Starting data preprocessing...")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Step 1: Handle missing values
    print("Handling missing values...")
    df = preprocessor.handle_missing_values(df)
    
    # Step 2: Standardize entity names
    print("Standardizing entity names...")
    df = preprocessor.standardize_entity_names(df)
    
    # Step 3: Create entity features
    print("Creating entity features...")
    df = preprocessor.create_entity_features(df)
    
    # Step 4: Create lag features
    print("Creating lag features...")
    df = preprocessor.create_lag_features(df)
    
    # Step 5: Detect outliers
    print("Detecting outliers...")
    df = preprocessor.detect_outliers(df)
    
    # Step 6: Encode categorical features
    print("Encoding categorical features...")
    df = preprocessor.encode_categorical_features(df)
    
    # Step 7: Scale numeric features
    print("Scaling numeric features...")
    df = preprocessor.scale_numeric_features(df)
    
    # Final cleanup
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    print(f"Preprocessing complete. Final shape: {df.shape}")
    
    return df
