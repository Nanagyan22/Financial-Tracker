import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score, 
    precision_score, recall_score, classification_report
)

# Optional ML packages - imported only when needed
XGB_AVAILABLE = False
LIGHTGBM_AVAILABLE = False

def try_import_xgb():
    """Try to import XGBoost only when needed."""
    global XGB_AVAILABLE
    try:
        import xgboost as xgb
        XGB_AVAILABLE = True
        return xgb
    except ImportError:
        XGB_AVAILABLE = False
        return None

def try_import_lgb():
    """Try to import LightGBM only when needed."""
    global LIGHTGBM_AVAILABLE
    try:
        import lightgbm as lgb
        LIGHTGBM_AVAILABLE = True
        return lgb
    except ImportError:
        LIGHTGBM_AVAILABLE = False
        return None

import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import text features
try:
    from text_features import create_text_features_pipeline
    TEXT_FEATURES_AVAILABLE = True
except ImportError:
    TEXT_FEATURES_AVAILABLE = False

class ModelTrainer:
    """Train and evaluate multiple ML models."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'audit_flag') -> tuple:
        """
        Prepare data for training.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Remove non-feature columns
        exclude_cols = [
            target_col, 'entity_name', 'detected_issues', 'severity', 
            'evidence_count', 'irregularity_score'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle any remaining non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        print(f"Features shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, X.columns.tolist()
    
    def create_time_aware_split(self, df: pd.DataFrame, test_size: float = 0.2) -> tuple:
        """
        Create time-aware train/test split.
        
        Args:
            df: Input DataFrame
            test_size: Proportion for test set
            
        Returns:
            Tuple of (train_idx, test_idx)
        """
        if 'year' not in df.columns:
            # Fallback to random split
            return train_test_split(df.index, test_size=test_size, random_state=42, 
                                   stratify=df['audit_flag'])
        
        # Sort by year
        df_sorted = df.sort_values('year')
        
        # Calculate split point
        split_point = int(len(df_sorted) * (1 - test_size))
        
        train_idx = df_sorted.index[:split_point]
        test_idx = df_sorted.index[split_point:]
        
        return train_idx, test_idx
    
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """Train Logistic Regression model."""
        print("Training Logistic Regression...")
        
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        return {
            'model': model,
            'name': 'Logistic Regression',
            'type': 'linear'
        }
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """Train Random Forest model."""
        print("Training Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        model.fit(X_train, y_train)
        
        return {
            'model': model,
            'name': 'Random Forest',
            'type': 'ensemble'
        }
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """Train XGBoost model."""
        xgb = try_import_xgb()
        if xgb is None:
            raise ImportError("XGBoost not available")
            
        print("Training XGBoost...")
        
        # Calculate scale_pos_weight for imbalanced data
        pos_count = (y_train == 1).sum()
        neg_count = (y_train == 0).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        
        model = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8
        )
        
        model.fit(X_train, y_train)
        
        return {
            'model': model,
            'name': 'XGBoost',
            'type': 'boosting'
        }
    
    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """Train LightGBM model."""
        lgb = try_import_lgb()
        if lgb is None:
            raise ImportError("LightGBM not available due to system dependencies")
            
        print("Training LightGBM...")
        
        model = lgb.LGBMClassifier(
            random_state=42,
            class_weight='balanced',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        return {
            'model': model,
            'name': 'LightGBM',
            'type': 'boosting'
        }
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Classification metrics
        f1 = f1_score(y_test, y_pred)
        precision_score_val = precision_score(y_test, y_pred, zero_division=0)
        recall_score_val = recall_score(y_test, y_pred)
        
        # Precision at k
        precision_at_k = self.calculate_precision_at_k(y_test, y_pred_proba)
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'f1_score': f1,
            'precision': precision_score_val,
            'recall': recall_score_val,
            'precision_at_k': precision_at_k,
            'predictions': y_pred_proba,
            'binary_predictions': y_pred
        }
    
    def calculate_precision_at_k(self, y_true, y_scores, k_values=[0.05, 0.1, 0.2]) -> dict:
        """Calculate precision at top k% of predictions."""
        precision_at_k = {}
        
        for k in k_values:
            # Get top k% threshold
            threshold = np.percentile(y_scores, (1 - k) * 100)
            
            # Predictions at this threshold
            y_pred_k = (y_scores >= threshold).astype(int)
            
            # Calculate precision
            if y_pred_k.sum() > 0:
                precision_k = (y_true & y_pred_k).sum() / y_pred_k.sum()
            else:
                precision_k = 0
            
            precision_at_k[f'precision_at_{int(k*100)}pct'] = precision_k
        
        return precision_at_k
    
    def get_feature_importance(self, model, feature_names: list) -> dict:
        """Extract feature importance from model."""
        importance_dict = {}
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance_dict = dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            # Linear models
            importance_dict = dict(zip(feature_names, np.abs(model.coef_[0])))
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

def train_models(df: pd.DataFrame) -> dict:
    """
    Main function to train all models.
    
    Args:
        df: Preprocessed DataFrame with features
        
    Returns:
        Dictionary with all trained models and results
    """
    print("Starting model training pipeline...")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Import and apply preprocessing
    from preprocess import DataPreprocessor
    preprocessor = DataPreprocessor()
    
    # Handle missing values and clean data
    df_clean = preprocessor.handle_missing_values(df)
    
    # Add text features if available
    if TEXT_FEATURES_AVAILABLE:
        try:
            print("🔤 Adding TF-IDF text features to dataset...")
            df_clean = create_text_features_pipeline(df_clean, text_column='audit_notes')
            print(f"✅ Enhanced dataset shape with text features: {df_clean.shape}")
        except Exception as e:
            print(f"⚠️ Could not add text features: {e}")
            print("Continuing without text features...")
    
    # Prepare data
    X, y, feature_names = trainer.prepare_data(df_clean)
    
    # Time-aware split
    train_idx, test_idx = trainer.create_time_aware_split(df_clean)
    
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Train positive rate: {y_train.mean():.3f}")
    print(f"Test positive rate: {y_test.mean():.3f}")
    
    # Train models
    models_to_train = [
        trainer.train_logistic_regression,
        trainer.train_random_forest,
    ]
    
    # Try to add optional models (will be tested at runtime)
    try:
        xgb_test = try_import_xgb()
        if xgb_test is not None:
            models_to_train.append(trainer.train_xgboost)
            print("XGBoost will be included in training")
    except Exception:
        print("XGBoost not available, skipping")
    
    try:
        lgb_test = try_import_lgb()
        if lgb_test is not None:
            models_to_train.append(trainer.train_lightgbm)
            print("LightGBM will be included in training")
    except Exception:
        print("LightGBM not available due to system dependencies, skipping")
    
    results = {}
    
    for train_func in models_to_train:
        try:
            # Train model
            model_info = train_func(X_train, y_train)
            model = model_info['model']
            model_name = model_info['name']
            
            # Evaluate model
            evaluation = trainer.evaluate_model(model, X_test, y_test)
            
            # Get feature importance
            feature_importance = trainer.get_feature_importance(model, feature_names)
            
            # Store results
            results[model_name] = {
                'model': model,
                'model_info': model_info,
                'evaluation': evaluation,
                'feature_importance': feature_importance,
                **evaluation  # Flatten evaluation metrics
            }
            
            print(f"{model_name} - ROC AUC: {evaluation['roc_auc']:.3f}, PR AUC: {evaluation['pr_auc']:.3f}")
            
        except Exception as e:
            print(f"Error training {train_func.__name__}: {e}")
    
    # Save best model
    if results:
        best_model_name = max(results.keys(), key=lambda k: results[k]['roc_auc'])
        best_model = results[best_model_name]['model']
        
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save best model and results
        joblib.dump(best_model, models_dir / "best_model.pkl")
        joblib.dump(feature_names, models_dir / "feature_names.pkl")
        joblib.dump(results, models_dir / "model_results.pkl")
        
        # Create and save explainer
        try:
            from explain import ModelExplainer
            explainer = ModelExplainer()
            explainer.create_explainer(best_model, X_train, best_model_name)
            joblib.dump(explainer, models_dir / "explainer.pkl")
            print("✅ Explainer saved to models/explainer.pkl")
        except Exception as e:
            print(f"⚠️ Could not save explainer: {e}")
        
        print(f"Best model: {best_model_name} (ROC AUC: {results[best_model_name]['roc_auc']:.3f})")
        print(f"Saved to models/best_model.pkl")
        print(f"Saved model results to models/model_results.pkl")
    
    return results
