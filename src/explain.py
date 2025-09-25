import pandas as pd
import numpy as np
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available, using basic feature importance")
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
import joblib
from typing import Dict, List, Optional, Union

class ModelExplainer:
    """Generate SHAP explanations for model predictions."""
    
    def __init__(self):
        self.explainers = {}
        self.shap_values = {}
        self.expected_values = {}
        
    def create_explainer(self, model, X_train: pd.DataFrame, model_name: str):
        """
        Create explainer for a model (SHAP if available, fallback otherwise).
        
        Args:
            model: Trained model
            X_train: Training data
            model_name: Name of the model
        """
        print(f"Creating explainer for {model_name}...")
        
        if SHAP_AVAILABLE:
            try:
                # Choose appropriate explainer based on model type
                if hasattr(model, 'tree_'):
                    # Single tree models
                    explainer = shap.TreeExplainer(model)
                elif hasattr(model, 'estimators_'):
                    # Ensemble tree models (Random Forest, etc.)
                    explainer = shap.TreeExplainer(model)
                elif hasattr(model, 'get_booster'):
                    # XGBoost
                    explainer = shap.TreeExplainer(model)
                elif model_name.lower() == 'lightgbm':
                    # LightGBM
                    explainer = shap.TreeExplainer(model)
                else:
                    # Linear models or other types - use KernelExplainer
                    background = shap.sample(X_train, min(100, len(X_train)))
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                
                self.explainers[model_name] = explainer
                print(f"✅ SHAP explainer created for {model_name}")
                return
                
            except Exception as e:
                print(f"❌ Error creating SHAP explainer for {model_name}: {e}")
        
        # Fallback: Store model and training data for basic feature importance
        self.explainers[model_name] = {
            'type': 'basic',
            'model': model,
            'X_train': X_train,
            'feature_names': list(X_train.columns)
        }
        print(f"✅ Basic feature importance explainer created for {model_name}")
    
    def calculate_feature_importance(self, model_name: str, X_data: pd.DataFrame, 
                                   max_samples: int = 500) -> np.ndarray:
        """
        Calculate feature importance/SHAP values for the given data.
        
        Args:
            model_name: Name of the model
            X_data: Data to explain
            max_samples: Maximum number of samples to explain
            
        Returns:
            Feature importance/SHAP values array
        """
        if model_name not in self.explainers:
            print(f"No explainer found for {model_name}")
            return None
        
        explainer = self.explainers[model_name]
        
        # Limit samples for performance
        if len(X_data) > max_samples:
            sample_idx = np.random.choice(len(X_data), max_samples, replace=False)
            X_sample = X_data.iloc[sample_idx]
        else:
            X_sample = X_data
        
        # Check if using basic explainer
        if isinstance(explainer, dict) and explainer.get('type') == 'basic':
            return self._calculate_basic_importance(explainer, X_sample)
        
        # SHAP explainer
        if not SHAP_AVAILABLE:
            return None
            
        try:
            print(f"Calculating SHAP values for {len(X_sample)} samples...")
            
            if hasattr(explainer, 'shap_values'):
                # Tree explainers
                shap_values = explainer.shap_values(X_sample)
                
                # Handle multiclass output (take positive class)
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]  # Positive class
                    
            else:
                # Kernel explainer
                shap_values = explainer.shap_values(X_sample)
                if len(shap_values.shape) == 3:  # Multiclass
                    shap_values = shap_values[:, :, 1]  # Positive class
            
            # Store expected value
            if hasattr(explainer, 'expected_value'):
                expected_value = explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1:
                    expected_value = expected_value[1]  # Positive class
                self.expected_values[model_name] = expected_value
            
            self.shap_values[model_name] = shap_values
            print(f"✅ SHAP values calculated for {model_name}")
            
            return shap_values
            
        except Exception as e:
            print(f"❌ Error calculating SHAP values for {model_name}: {e}")
            return None
    
    def _calculate_basic_importance(self, explainer_dict: dict, X_sample: pd.DataFrame) -> np.ndarray:
        """Calculate basic feature importance using model's built-in methods."""
        model = explainer_dict['model']
        feature_names = explainer_dict['feature_names']
        
        try:
            # Get feature importance from model
            if hasattr(model, 'feature_importances_'):
                # Tree-based models (Random Forest, XGBoost)
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models (Logistic Regression)
                importance = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            else:
                # Uniform importance as last resort
                importance = np.ones(len(feature_names)) / len(feature_names)
            
            # Create mock SHAP values by broadcasting importance to sample size
            shap_values = np.outer(np.ones(len(X_sample)), importance)
            
            print(f"✅ Basic feature importance calculated for {len(X_sample)} samples")
            return shap_values
            
        except Exception as e:
            print(f"❌ Error calculating basic importance: {e}")
            return None
    
    def create_global_importance_plot(self, model_name: str, feature_names: List[str], 
                                    top_n: int = 20) -> go.Figure:
        """
        Create global feature importance plot using SHAP values.
        
        Args:
            model_name: Name of the model
            feature_names: List of feature names
            top_n: Number of top features to show
            
        Returns:
            Plotly figure
        """
        if model_name not in self.shap_values:
            return go.Figure()
        
        shap_values = self.shap_values[model_name]
        
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Create plot
        fig = go.Figure([go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h',
            text=[f'{imp:.3f}' for imp in importance_df['importance']],
            textposition='auto'
        )])
        
        fig.update_layout(
            title=f'Global Feature Importance (SHAP) - {model_name}',
            xaxis_title='Mean |SHAP Value|',
            yaxis_title='Features',
            height=max(400, top_n * 25),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def create_summary_plot_data(self, model_name: str, feature_names: List[str], 
                               X_data: pd.DataFrame, top_n: int = 20) -> Dict:
        """
        Create data for SHAP summary plot.
        
        Args:
            model_name: Name of the model
            feature_names: List of feature names
            X_data: Feature data
            top_n: Number of top features
            
        Returns:
            Dictionary with plot data
        """
        if model_name not in self.shap_values:
            return {}
        
        shap_values = self.shap_values[model_name]
        
        # Get top features by mean absolute SHAP value
        mean_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_shap)[-top_n:][::-1]
        
        # Prepare data for each feature
        plot_data = []
        
        for i, feature_idx in enumerate(top_indices):
            feature_name = feature_names[feature_idx]
            feature_shap = shap_values[:, feature_idx]
            feature_values = X_data.iloc[:len(feature_shap), feature_idx].values
            
            plot_data.append({
                'feature': feature_name,
                'shap_values': feature_shap.tolist(),
                'feature_values': feature_values.tolist(),
                'importance_rank': i + 1
            })
        
        return {
            'model_name': model_name,
            'features': plot_data,
            'expected_value': self.expected_values.get(model_name, 0)
        }
    
    def create_waterfall_plot(self, model_name: str, feature_names: List[str], 
                            instance_idx: int, X_data: pd.DataFrame) -> go.Figure:
        """
        Create waterfall plot for a single prediction.
        
        Args:
            model_name: Name of the model
            feature_names: List of feature names
            instance_idx: Index of the instance to explain
            X_data: Feature data
            
        Returns:
            Plotly figure
        """
        if model_name not in self.shap_values:
            return go.Figure()
        
        shap_values = self.shap_values[model_name]
        expected_value = self.expected_values.get(model_name, 0)
        
        if instance_idx >= len(shap_values):
            return go.Figure()
        
        # Get SHAP values for this instance
        instance_shap = shap_values[instance_idx]
        instance_features = X_data.iloc[instance_idx]
        
        # Sort by absolute SHAP value
        sorted_indices = np.argsort(np.abs(instance_shap))[::-1]
        
        # Take top features for clarity
        top_n = min(15, len(sorted_indices))
        top_indices = sorted_indices[:top_n]
        
        # Prepare waterfall data
        feature_names_subset = [feature_names[i] for i in top_indices]
        shap_values_subset = instance_shap[top_indices]
        
        # Create waterfall plot
        cumulative = expected_value
        x_values = ['Expected Value']
        y_values = [expected_value]
        
        for i, (feature_name, shap_val) in enumerate(zip(feature_names_subset, shap_values_subset)):
            cumulative += shap_val
            x_values.append(f"{feature_name}\n({instance_features[top_indices[i]]:.2f})")
            y_values.append(cumulative)
        
        x_values.append('Prediction')
        y_values.append(cumulative)
        
        # Create the plot
        fig = go.Figure()
        
        # Add bars for each step
        for i in range(1, len(y_values) - 1):
            color = 'red' if shap_values_subset[i-1] < 0 else 'blue'
            fig.add_trace(go.Bar(
                x=[x_values[i]],
                y=[abs(shap_values_subset[i-1])],
                base=[y_values[i-1]] if shap_values_subset[i-1] > 0 else [y_values[i]],
                marker_color=color,
                showlegend=False,
                text=[f'{shap_values_subset[i-1]:+.3f}'],
                textposition='middle center'
            ))
        
        # Add expected value and final prediction
        fig.add_trace(go.Scatter(
            x=[x_values[0], x_values[-1]],
            y=[y_values[0], y_values[-1]],
            mode='markers',
            marker=dict(size=10, color='green'),
            name='Key Values',
            text=[f'Expected: {expected_value:.3f}', f'Prediction: {cumulative:.3f}'],
            textposition='top center'
        ))
        
        fig.update_layout(
            title=f'SHAP Waterfall Plot - Instance {instance_idx} ({model_name})',
            xaxis_title='Features',
            yaxis_title='Output',
            height=600,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_force_plot_data(self, model_name: str, feature_names: List[str], 
                              instance_idx: int, X_data: pd.DataFrame) -> Dict:
        """
        Create data for SHAP force plot.
        
        Args:
            model_name: Name of the model
            feature_names: List of feature names
            instance_idx: Index of the instance
            X_data: Feature data
            
        Returns:
            Dictionary with force plot data
        """
        if model_name not in self.shap_values:
            return {}
        
        shap_values = self.shap_values[model_name]
        expected_value = self.expected_values.get(model_name, 0)
        
        if instance_idx >= len(shap_values):
            return {}
        
        instance_shap = shap_values[instance_idx]
        instance_features = X_data.iloc[instance_idx]
        
        # Prepare force plot data
        force_data = []
        for i, (feature_name, shap_val, feature_val) in enumerate(
            zip(feature_names, instance_shap, instance_features)
        ):
            force_data.append({
                'feature': feature_name,
                'shap_value': float(shap_val),
                'feature_value': float(feature_val),
                'effect': 'positive' if shap_val > 0 else 'negative'
            })
        
        # Sort by absolute SHAP value
        force_data.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        return {
            'model_name': model_name,
            'instance_idx': instance_idx,
            'expected_value': float(expected_value),
            'prediction': float(expected_value + instance_shap.sum()),
            'features': force_data[:20]  # Top 20 features
        }

def generate_explanations(df: pd.DataFrame, models_results: Dict) -> Dict:
    """
    Main function to generate SHAP explanations for all models.
    
    Args:
        df: Processed DataFrame
        models_results: Dictionary with trained models
        
    Returns:
        Dictionary with explanation results
    """
    print("Starting SHAP explanation generation...")
    
    explainer = ModelExplainer()
    
    # Prepare data
    exclude_cols = [
        'audit_flag', 'entity_name', 'detected_issues', 'severity', 
        'evidence_count', 'irregularity_score'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].select_dtypes(include=[np.number])
    feature_names = X.columns.tolist()
    
    # Split data (use same logic as training)
    if 'year' in df.columns:
        df_sorted = df.sort_values('year')
        split_point = int(len(df_sorted) * 0.8)
        X_train = X.iloc[:split_point]
        X_test = X.iloc[split_point:]
    else:
        X_train = X.iloc[:int(len(X) * 0.8)]
        X_test = X.iloc[int(len(X) * 0.8):]
    
    explanations = {}
    
    # Generate explanations for each model
    for model_name, results in models_results.items():
        try:
            model = results['model']
            
            # Create explainer
            explainer.create_explainer(model, X_train, model_name)
            
            # Calculate SHAP values on test set
            shap_values = explainer.calculate_shap_values(model_name, X_test)
            
            if shap_values is not None:
                # Create explanation plots and data
                explanations[model_name] = {
                    'global_importance': explainer.create_global_importance_plot(
                        model_name, feature_names
                    ),
                    'summary_data': explainer.create_summary_plot_data(
                        model_name, feature_names, X_test
                    ),
                    'waterfall_plots': {},
                    'force_plots': {}
                }
                
                # Create waterfall plots for a few sample instances
                sample_indices = [0, len(X_test)//4, len(X_test)//2, 3*len(X_test)//4, -1]
                sample_indices = [i for i in sample_indices if 0 <= i < len(X_test)]
                
                for idx in sample_indices[:3]:  # Limit to 3 samples
                    waterfall_plot = explainer.create_waterfall_plot(
                        model_name, feature_names, idx, X_test
                    )
                    force_plot_data = explainer.create_force_plot_data(
                        model_name, feature_names, idx, X_test
                    )
                    
                    explanations[model_name]['waterfall_plots'][idx] = waterfall_plot
                    explanations[model_name]['force_plots'][idx] = force_plot_data
                
                print(f"✅ Explanations generated for {model_name}")
        
        except Exception as e:
            print(f"❌ Error generating explanations for {model_name}: {e}")
    
    # Save explanations
    explanations_dir = Path("models")
    explanations_dir.mkdir(exist_ok=True)
    
    # Save explainer objects
    joblib.dump(explainer, explanations_dir / "explainer.pkl")
    
    print("SHAP explanation generation completed!")
    
    return explanations
