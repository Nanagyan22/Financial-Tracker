import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

def save_model(model, model_name: str, models_dir: str = "models") -> str:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model object
        model_name: Name for the model file
        models_dir: Directory to save models
        
    Returns:
        Path to saved model file
    """
    models_path = Path(models_dir)
    models_path.mkdir(exist_ok=True)
    
    file_path = models_path / f"{model_name}.pkl"
    joblib.dump(model, file_path)
    
    return str(file_path)

def load_model(model_path: str):
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Loaded model object
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return joblib.load(model_path)

def save_data(data: pd.DataFrame, filename: str, data_dir: str = "data/processed") -> str:
    """
    Save DataFrame to parquet format.
    
    Args:
        data: DataFrame to save
        filename: Name of the file
        data_dir: Directory to save data
        
    Returns:
        Path to saved file
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    if not filename.endswith('.parquet'):
        filename += '.parquet'
    
    file_path = data_path / filename
    data.to_parquet(file_path, index=False)
    
    return str(file_path)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load DataFrame from parquet file.
    
    Args:
        file_path: Path to parquet file
        
    Returns:
        Loaded DataFrame
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    return pd.read_parquet(file_path)

def save_json(data: Dict, filename: str, data_dir: str = "data/processed") -> str:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filename: Name of the file
        data_dir: Directory to save data
        
    Returns:
        Path to saved file
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    if not filename.endswith('.json'):
        filename += '.json'
    
    file_path = data_path / filename
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    return str(file_path)

def load_json(file_path: str) -> Dict:
    """
    Load dictionary from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        return json.load(f)

def get_model_metrics(models_results: Dict) -> pd.DataFrame:
    """
    Extract model metrics into a comparison DataFrame.
    
    Args:
        models_results: Dictionary with model results
        
    Returns:
        DataFrame with model metrics
    """
    metrics_data = []
    
    for model_name, results in models_results.items():
        metrics = {
            'Model': model_name,
            'ROC_AUC': results.get('roc_auc', 0),
            'PR_AUC': results.get('pr_auc', 0),
            'F1_Score': results.get('f1_score', 0),
            'Precision': results.get('precision', 0),
            'Recall': results.get('recall', 0)
        }
        
        # Add Precision@k metrics if available
        if 'precision_at_k' in results:
            for k in [5, 10, 20]:
                key = f'precision_at_{k}pct'
                if key in results['precision_at_k']:
                    metrics[f'P@{k}%'] = results['precision_at_k'][key]
        
        metrics_data.append(metrics)
    
    return pd.DataFrame(metrics_data)

def create_entity_lookup(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Create a lookup dictionary for entity information.
    
    Args:
        df: DataFrame with entity data
        
    Returns:
        Dictionary mapping entity names to their information
    """
    if 'entity_name' not in df.columns:
        return {}
    
    entity_lookup = {}
    
    for entity in df['entity_name'].unique():
        entity_data = df[df['entity_name'] == entity].iloc[-1]  # Most recent record
        
        entity_info = {
            'latest_year': entity_data.get('year', 'Unknown'),
            'audit_flag': entity_data.get('audit_flag', 0),
            'revenue': entity_data.get('revenue', 0),
            'expenditure': entity_data.get('expenditure', 0),
            'total_assets': entity_data.get('total_assets', 0),
            'entity_type': entity_data.get('entity_type', 'Unknown'),
            'sector': entity_data.get('sector', 'Unknown'),
            'region': entity_data.get('region', 'Unknown')
        }
        
        entity_lookup[entity] = entity_info
    
    return entity_lookup

def calculate_risk_score(entity_data: pd.Series, model_results: Dict) -> Dict:
    """
    Calculate risk score for an entity using the best model.
    
    Args:
        entity_data: Series with entity data
        model_results: Dictionary with trained models
        
    Returns:
        Dictionary with risk assessment
    """
    try:
        # Get best model
        best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['roc_auc'])
        best_model = model_results[best_model_name]['model']
        
        # Prepare features (exclude non-feature columns)
        exclude_cols = [
            'audit_flag', 'entity_name', 'detected_issues', 'severity', 
            'evidence_count', 'irregularity_score'
        ]
        
        feature_cols = [col for col in entity_data.index if col not in exclude_cols]
        features = entity_data[feature_cols].select_dtypes(include=[np.number])
        
        # Make prediction
        risk_probability = best_model.predict_proba([features])[0][1]
        risk_category = 'High' if risk_probability > 0.7 else 'Medium' if risk_probability > 0.3 else 'Low'
        
        return {
            'risk_score': risk_probability,
            'risk_category': risk_category,
            'model_used': best_model_name,
            'confidence': max(risk_probability, 1 - risk_probability)
        }
        
    except Exception as e:
        return {
            'risk_score': 0.5,
            'risk_category': 'Unclassified',
            'model_used': 'None',
            'confidence': 0.5,
            'error': str(e)
        }

def format_currency(amount: float, currency: str = 'GHS') -> str:
    """
    Format currency amounts for display.
    
    Args:
        amount: Amount to format
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    if pd.isna(amount) or amount == 0:
        return f"{currency} 0"
    
    if amount >= 1e9:
        return f"{currency} {amount/1e9:.2f}B"
    elif amount >= 1e6:
        return f"{currency} {amount/1e6:.2f}M"
    elif amount >= 1e3:
        return f"{currency} {amount/1e3:.2f}K"
    else:
        return f"{currency} {amount:,.2f}"

def get_entity_history(df: pd.DataFrame, entity_name: str) -> pd.DataFrame:
    """
    Get historical data for a specific entity.
    
    Args:
        df: Full DataFrame
        entity_name: Name of the entity
        
    Returns:
        DataFrame with entity's historical data
    """
    if 'entity_name' not in df.columns:
        return pd.DataFrame()
    
    entity_data = df[df['entity_name'] == entity_name].copy()
    
    if 'year' in entity_data.columns:
        entity_data = entity_data.sort_values('year')
    
    return entity_data

def detect_concept_drift(current_data: pd.DataFrame, reference_data: pd.DataFrame, 
                        threshold: float = 0.05) -> Dict:
    """
    Detect concept drift using Kolmogorov-Smirnov test.
    
    Args:
        current_data: Current period data
        reference_data: Reference period data
        threshold: P-value threshold for drift detection
        
    Returns:
        Dictionary with drift detection results
    """
    from scipy.stats import ks_2samp
    
    drift_results = {}
    numeric_cols = current_data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col in reference_data.columns:
            try:
                # Perform KS test
                statistic, p_value = ks_2samp(
                    reference_data[col].dropna(), 
                    current_data[col].dropna()
                )
                
                drift_detected = p_value < threshold
                
                drift_results[col] = {
                    'ks_statistic': statistic,
                    'p_value': p_value,
                    'drift_detected': drift_detected,
                    'drift_severity': 'High' if p_value < 0.01 else 'Medium' if p_value < 0.05 else 'Low'
                }
                
            except Exception as e:
                drift_results[col] = {
                    'error': str(e),
                    'drift_detected': False
                }
    
    # Overall drift summary
    total_features = len(drift_results)
    drifted_features = sum(1 for r in drift_results.values() 
                          if r.get('drift_detected', False))
    
    drift_results['summary'] = {
        'total_features': total_features,
        'drifted_features': drifted_features,
        'drift_percentage': drifted_features / total_features if total_features > 0 else 0,
        'overall_drift': drifted_features / total_features > 0.3 if total_features > 0 else False
    }
    
    return drift_results

def validate_data_quality(df: pd.DataFrame) -> Dict:
    """
    Validate data quality and return assessment.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary with data quality assessment
    """
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns)
    }
    
    # Missing value percentage
    quality_report['missing_percentage'] = {
        col: (missing / len(df)) * 100 
        for col, missing in quality_report['missing_values'].items()
    }
    
    # Overall quality score
    missing_score = 1 - (sum(quality_report['missing_values'].values()) / 
                        (len(df) * len(df.columns)))
    duplicate_score = 1 - (quality_report['duplicate_rows'] / len(df))
    
    quality_report['quality_score'] = (missing_score + duplicate_score) / 2
    quality_report['quality_grade'] = (
        'Excellent' if quality_report['quality_score'] > 0.9 else
        'Good' if quality_report['quality_score'] > 0.7 else
        'Fair' if quality_report['quality_score'] > 0.5 else
        'Poor'
    )
    
    return quality_report

def create_backup(file_path: str, backup_dir: str = "backups") -> str:
    """
    Create backup of a file.
    
    Args:
        file_path: Path to file to backup
        backup_dir: Directory for backups
        
    Returns:
        Path to backup file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    backup_path = Path(backup_dir)
    backup_path.mkdir(exist_ok=True)
    
    file_name = Path(file_path).name
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_path / f"{timestamp}_{file_name}"
    
    # Copy file
    import shutil
    shutil.copy2(file_path, backup_file)
    
    return str(backup_file)
