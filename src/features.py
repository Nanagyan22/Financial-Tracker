import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class FinancialFeatureEngineer:
    """Engineer financial and risk-related features."""
    
    def __init__(self):
        self.feature_definitions = {}
    
    def create_financial_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive financial ratios.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with financial ratio features
        """
        df = df.copy()
        
        # Liquidity Ratios
        if 'cash' in df.columns and 'total_liabilities' in df.columns:
            df['cash_ratio'] = df['cash'] / (df['total_liabilities'] + 1e-6)
        
        if 'total_assets' in df.columns and 'total_liabilities' in df.columns:
            df['current_ratio'] = df['total_assets'] / (df['total_liabilities'] + 1e-6)
            df['debt_ratio'] = df['total_liabilities'] / (df['total_assets'] + 1e-6)
        
        # Efficiency Ratios
        if 'revenue' in df.columns and 'total_assets' in df.columns:
            df['asset_turnover'] = df['revenue'] / (df['total_assets'] + 1e-6)
        
        if 'revenue' in df.columns and 'expenditure' in df.columns:
            df['expense_ratio'] = df['expenditure'] / (df['revenue'] + 1e-6)
            df['operating_margin'] = (df['revenue'] - df['expenditure']) / (df['revenue'] + 1e-6)
        
        # Size Indicators
        if 'total_assets' in df.columns:
            df['log_total_assets'] = np.log1p(df['total_assets'])
        
        if 'revenue' in df.columns:
            df['log_revenue'] = np.log1p(df['revenue'])
        
        # Financial Health Indicators
        if 'equity' in df.columns and 'total_assets' in df.columns:
            df['equity_ratio'] = df['equity'] / (df['total_assets'] + 1e-6)
        
        return df
    
    def create_procurement_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create procurement-related risk indicators.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with procurement features
        """
        df = df.copy()
        
        # Contract analysis
        if 'contract_count' in df.columns and 'single_source_contracts' in df.columns:
            df['single_source_ratio'] = df['single_source_contracts'] / (df['contract_count'] + 1e-6)
        
        if 'contract_value' in df.columns and 'contract_count' in df.columns:
            df['avg_contract_value'] = df['contract_value'] / (df['contract_count'] + 1e-6)
        
        if 'contract_value' in df.columns and 'total_assets' in df.columns:
            df['contract_to_assets_ratio'] = df['contract_value'] / (df['total_assets'] + 1e-6)
        
        if 'contract_value' in df.columns and 'expenditure' in df.columns:
            df['contract_to_expenditure_ratio'] = df['contract_value'] / (df['expenditure'] + 1e-6)
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with temporal features
        """
        if 'year' not in df.columns:
            return df
        
        df = df.copy()
        
        # Year-based features
        df['years_since_2020'] = df['year'] - 2020
        df['is_recent_year'] = (df['year'] >= 2022).astype(int)
        
        # Financial growth indicators (if lag features exist)
        growth_cols = [col for col in df.columns if 'yoy_pct_change' in col]
        
        if growth_cols:
            # Volatility indicators
            for entity in df['entity_name'].unique() if 'entity_name' in df.columns else []:
                entity_data = df[df['entity_name'] == entity] if 'entity_name' in df.columns else df
                
                for col in growth_cols:
                    if col in entity_data.columns:
                        volatility = entity_data[col].std()
                        df.loc[df['entity_name'] == entity, f'{col}_volatility'] = volatility
        
        return df
    
    def create_hr_payroll_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create HR and payroll-related features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with HR features
        """
        df = df.copy()
        
        # Employee efficiency
        if 'revenue' in df.columns and 'employee_count' in df.columns:
            df['revenue_per_employee'] = df['revenue'] / (df['employee_count'] + 1e-6)
        
        if 'expenditure' in df.columns and 'employee_count' in df.columns:
            df['expenditure_per_employee'] = df['expenditure'] / (df['employee_count'] + 1e-6)
        
        # Payroll analysis
        if 'payroll_expense' in df.columns and 'employee_count' in df.columns:
            df['avg_salary'] = df['payroll_expense'] / (df['employee_count'] + 1e-6)
        
        if 'payroll_expense' in df.columns and 'expenditure' in df.columns:
            df['payroll_to_expenditure_ratio'] = df['payroll_expense'] / (df['expenditure'] + 1e-6)
        
        return df
    
    def create_risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive risk indicators.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with risk features
        """
        df = df.copy()
        
        # Financial stress indicators
        if 'debt_ratio' in df.columns:
            df['high_debt_risk'] = (df['debt_ratio'] > 0.8).astype(int)
        
        if 'expense_ratio' in df.columns:
            df['overspending_risk'] = (df['expense_ratio'] > 1.0).astype(int)
        
        if 'single_source_ratio' in df.columns:
            df['procurement_risk'] = (df['single_source_ratio'] > 0.5).astype(int)
        
        # Composite risk score
        risk_indicators = [col for col in df.columns if col.endswith('_risk')]
        if risk_indicators:
            df['composite_risk_score'] = df[risk_indicators].sum(axis=1)
        
        # Size-based risk (very small or very large entities might have different risk profiles)
        if 'total_assets' in df.columns:
            asset_percentiles = df['total_assets'].quantile([0.1, 0.9])
            df['size_risk'] = ((df['total_assets'] < asset_percentiles[0.1]) | 
                              (df['total_assets'] > asset_percentiles[0.9])).astype(int)
        
        return df
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features for anomaly detection.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with statistical features
        """
        df = df.copy()
        
        # Financial metrics relative to peer group
        if 'entity_type' in df.columns:
            for entity_type in df['entity_type'].unique():
                type_mask = df['entity_type'] == entity_type
                type_data = df[type_mask]
                
                # Calculate z-scores within entity type
                numeric_cols = ['revenue', 'expenditure', 'total_assets']
                for col in numeric_cols:
                    if col in df.columns and len(type_data) > 1:
                        mean_val = type_data[col].mean()
                        std_val = type_data[col].std()
                        
                        if std_val > 0:
                            df.loc[type_mask, f'{col}_zscore_within_type'] = (
                                (df.loc[type_mask, col] - mean_val) / std_val
                            )
        
        # Overall percentile rankings
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['year', 'audit_flag'] and df[col].nunique() > 10:
                df[f'{col}_percentile'] = df[col].rank(pct=True)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between key variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        # Size x Performance interactions
        if 'log_total_assets' in df.columns and 'operating_margin' in df.columns:
            df['size_performance_interaction'] = df['log_total_assets'] * df['operating_margin']
        
        # Risk x Size interactions
        if 'composite_risk_score' in df.columns and 'log_total_assets' in df.columns:
            df['risk_size_interaction'] = df['composite_risk_score'] * df['log_total_assets']
        
        # Efficiency x Scale interactions
        if 'asset_turnover' in df.columns and 'log_revenue' in df.columns:
            df['efficiency_scale_interaction'] = df['asset_turnover'] * df['log_revenue']
        
        return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main feature engineering pipeline.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    print("Starting feature engineering...")
    
    # Initialize feature engineer
    engineer = FinancialFeatureEngineer()
    
    # Create financial ratios
    print("Creating financial ratios...")
    df = engineer.create_financial_ratios(df)
    
    # Create procurement indicators
    print("Creating procurement indicators...")
    df = engineer.create_procurement_indicators(df)
    
    # Create temporal features
    print("Creating temporal features...")
    df = engineer.create_temporal_features(df)
    
    # Create HR/payroll features
    print("Creating HR/payroll features...")
    df = engineer.create_hr_payroll_features(df)
    
    # Create risk indicators
    print("Creating risk indicators...")
    df = engineer.create_risk_indicators(df)
    
    # Create statistical features
    print("Creating statistical features...")
    df = engineer.create_statistical_features(df)
    
    # Create interaction features
    print("Creating interaction features...")
    df = engineer.create_interaction_features(df)
    
    # Final cleanup
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    print(f"Feature engineering complete. Final shape: {df.shape}")
    
    return df
