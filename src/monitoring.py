"""
Advanced Model Monitoring System

This module provides comprehensive model performance monitoring including
precision@k curves, performance degradation detection, and automated alerting.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import joblib
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc, 
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score
)

class ModelMonitor:
    """
    Comprehensive model performance monitoring system.
    """
    
    def __init__(self):
        self.performance_history = []
        self.alert_thresholds = {
            'roc_auc_min': 0.7,
            'precision_at_10_min': 0.8,
            'f1_score_min': 0.6,
            'drift_threshold': 0.05,
            'performance_drop_threshold': 0.1
        }
        
    def calculate_precision_at_k(self, y_true: np.ndarray, y_prob: np.ndarray, k_values: List[int]) -> Dict[int, float]:
        """
        Calculate precision@k for different k values.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            k_values: List of k values to calculate precision for
            
        Returns:
            Dictionary mapping k to precision@k
        """
        # Sort by predicted probability (descending)
        sorted_indices = np.argsort(y_prob)[::-1]
        sorted_labels = y_true[sorted_indices]
        
        precision_at_k = {}
        
        for k in k_values:
            if k > len(sorted_labels):
                k = len(sorted_labels)
            
            if k == 0:
                precision_at_k[k] = 0.0
                continue
            
            # Calculate precision@k
            top_k_labels = sorted_labels[:k]
            precision_k = np.sum(top_k_labels) / k
            precision_at_k[k] = precision_k
        
        return precision_at_k
    
    def create_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray, model_name: str) -> go.Figure:
        """
        Create precision-recall curve visualization.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            model_name: Name of the model
            
        Returns:
            Plotly figure with PR curve
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        fig = go.Figure()
        
        # PR Curve
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'{model_name} (AUC = {pr_auc:.3f})',
            line=dict(width=3)
        ))
        
        # Random classifier baseline
        baseline = np.sum(y_true) / len(y_true)
        fig.add_hline(
            y=baseline,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Random Classifier ({baseline:.3f})"
        )
        
        fig.update_layout(
            title=f"Precision-Recall Curve - {model_name}",
            xaxis_title="Recall",
            yaxis_title="Precision",
            width=600,
            height=450,
            showlegend=True
        )
        
        return fig
    
    def create_precision_at_k_plot(self, y_true: np.ndarray, y_prob: np.ndarray, max_k: int = 100) -> go.Figure:
        """
        Create precision@k curve visualization.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            max_k: Maximum k value to evaluate
            
        Returns:
            Plotly figure with precision@k curve
        """
        k_values = list(range(1, min(max_k + 1, len(y_true) + 1), 5))
        precision_at_k = self.calculate_precision_at_k(y_true, y_prob, k_values)
        
        fig = go.Figure()
        
        # Precision@k curve
        fig.add_trace(go.Scatter(
            x=k_values,
            y=[precision_at_k[k] for k in k_values],
            mode='lines+markers',
            name='Precision@k',
            line=dict(width=3),
            marker=dict(size=6)
        ))
        
        # Add key precision@k annotations
        key_k_values = [10, 20, 50]
        for k in key_k_values:
            if k in precision_at_k:
                fig.add_annotation(
                    x=k,
                    y=precision_at_k[k],
                    text=f"P@{k}: {precision_at_k[k]:.3f}",
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="red",
                    ax=0,
                    ay=-30
                )
        
        fig.update_layout(
            title="Precision at K",
            xaxis_title="K (Number of Top Predictions)",
            yaxis_title="Precision@K",
            width=600,
            height=450,
            showlegend=True
        )
        
        return fig
    
    def create_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, model_name: str) -> go.Figure:
        """
        Create ROC curve visualization.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            model_name: Name of the model
            
        Returns:
            Plotly figure with ROC curve
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        # ROC Curve
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {roc_auc:.3f})',
            line=dict(width=3)
        ))
        
        # Random classifier line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title=f"ROC Curve - {model_name}",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            width=600,
            height=450,
            showlegend=True
        )
        
        return fig
    
    def create_performance_timeline(self, performance_history: List[Dict]) -> go.Figure:
        """
        Create performance timeline showing metrics over time.
        
        Args:
            performance_history: List of performance records
            
        Returns:
            Plotly figure with performance timeline
        """
        if not performance_history:
            # Create dummy data for demonstration
            dates = [datetime.now() - timedelta(days=x) for x in range(30, 0, -1)]
            performance_history = []
            
            for date in dates:
                # Simulate performance metrics with slight degradation trend
                base_roc = 0.92 - (30 - (date - dates[0]).days) * 0.001
                noise = np.random.normal(0, 0.02)
                
                performance_history.append({
                    'date': date,
                    'roc_auc': max(0.5, min(1.0, base_roc + noise)),
                    'precision_at_10': max(0.3, min(1.0, base_roc + noise + 0.05)),
                    'f1_score': max(0.3, min(1.0, base_roc + noise - 0.1)),
                    'model_name': 'XGBoost'
                })
        
        df = pd.DataFrame(performance_history)
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('ROC AUC Over Time', 'Precision@10 Over Time', 'F1 Score Over Time'),
            vertical_spacing=0.08
        )
        
        # ROC AUC timeline
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['roc_auc'],
                mode='lines+markers',
                name='ROC AUC',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add threshold line for ROC AUC
        fig.add_hline(
            y=self.alert_thresholds['roc_auc_min'],
            line_dash="dash",
            line_color="red",
            row=1, col=1
        )
        
        # Precision@10 timeline
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['precision_at_10'],
                mode='lines+markers',
                name='Precision@10',
                line=dict(color='green', width=2),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add threshold line for Precision@10
        fig.add_hline(
            y=self.alert_thresholds['precision_at_10_min'],
            line_dash="dash",
            line_color="red",
            row=2, col=1
        )
        
        # F1 Score timeline
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['f1_score'],
                mode='lines+markers',
                name='F1 Score',
                line=dict(color='purple', width=2),
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Add threshold line for F1 Score
        fig.add_hline(
            y=self.alert_thresholds['f1_score_min'],
            line_dash="dash",
            line_color="red",
            row=3, col=1
        )
        
        fig.update_layout(
            height=800,
            title_text="Model Performance Over Time",
            showlegend=True
        )
        
        return fig
    
    def detect_performance_degradation(self, current_metrics: Dict, historical_metrics: List[Dict]) -> List[str]:
        """
        Detect performance degradation and generate alerts.
        
        Args:
            current_metrics: Current model performance metrics
            historical_metrics: Historical performance data
            
        Returns:
            List of alert messages
        """
        alerts = []
        
        # Check absolute thresholds
        if current_metrics.get('roc_auc', 0) < self.alert_thresholds['roc_auc_min']:
            alerts.append(f"⚠️ ROC AUC ({current_metrics['roc_auc']:.3f}) below minimum threshold ({self.alert_thresholds['roc_auc_min']})")
        
        if current_metrics.get('precision_at_10', 0) < self.alert_thresholds['precision_at_10_min']:
            alerts.append(f"⚠️ Precision@10 ({current_metrics['precision_at_10']:.3f}) below minimum threshold ({self.alert_thresholds['precision_at_10_min']})")
        
        if current_metrics.get('f1_score', 0) < self.alert_thresholds['f1_score_min']:
            alerts.append(f"⚠️ F1 Score ({current_metrics['f1_score']:.3f}) below minimum threshold ({self.alert_thresholds['f1_score_min']})")
        
        # Check for performance drops compared to historical performance
        if historical_metrics:
            recent_avg = {}
            for metric in ['roc_auc', 'precision_at_10', 'f1_score']:
                recent_values = [h[metric] for h in historical_metrics[-5:] if metric in h]
                if recent_values:
                    recent_avg[metric] = np.mean(recent_values)
            
            for metric, avg_value in recent_avg.items():
                current_value = current_metrics.get(metric, 0)
                drop_ratio = (avg_value - current_value) / avg_value if avg_value > 0 else 0
                
                if drop_ratio > self.alert_thresholds['performance_drop_threshold']:
                    alerts.append(f"📉 {metric.replace('_', ' ').title()} dropped by {drop_ratio*100:.1f}% compared to recent average")
        
        return alerts
    
    def create_model_comparison_chart(self, model_results: Dict) -> go.Figure:
        """
        Create comprehensive model comparison visualization.
        
        Args:
            model_results: Dictionary with model results
            
        Returns:
            Plotly figure with model comparison
        """
        if not model_results:
            return go.Figure().add_annotation(
                text="No model results available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        models = list(model_results.keys())
        metrics = ['roc_auc', 'pr_auc', 'f1', 'precision', 'recall']
        
        # Create radar chart for model comparison
        fig = go.Figure()
        
        for model_name in models:
            evaluation = model_results[model_name].get('evaluation', {})
            values = []
            
            for metric in metrics:
                if metric == 'pr_auc':
                    values.append(evaluation.get('pr_auc', 0))
                else:
                    values.append(evaluation.get(metric, 0))
            
            # Close the radar chart
            values_closed = values + [values[0]]
            metrics_closed = metrics + [metrics[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=metrics_closed,
                fill='toself',
                name=model_name,
                line_color=px.colors.qualitative.Set1[len(fig.data) % len(px.colors.qualitative.Set1)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Comparison (Radar Chart)",
            width=600,
            height=500
        )
        
        return fig
    
    def generate_monitoring_summary(self, model_results: Dict, y_true: np.ndarray = None, y_prob: np.ndarray = None) -> Dict:
        """
        Generate comprehensive monitoring summary.
        
        Args:
            model_results: Model results dictionary
            y_true: True labels for current evaluation
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary with monitoring summary
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'models_monitored': len(model_results),
            'best_model': None,
            'performance_status': 'Unknown',
            'alerts': [],
            'key_metrics': {}
        }
        
        if model_results:
            # Find best model
            best_model_name = max(model_results.keys(), 
                                key=lambda k: model_results[k].get('evaluation', {}).get('roc_auc', 0))
            summary['best_model'] = best_model_name
            
            best_metrics = model_results[best_model_name].get('evaluation', {})
            summary['key_metrics'] = best_metrics
            
            # Calculate precision@k if we have current predictions
            if y_true is not None and y_prob is not None:
                precision_at_k = self.calculate_precision_at_k(y_true, y_prob, [10, 20, 50])
                summary['precision_at_k'] = precision_at_k
                best_metrics['precision_at_10'] = precision_at_k.get(10, 0)
            
            # Check for alerts
            summary['alerts'] = self.detect_performance_degradation(best_metrics, [])
            
            # Determine overall status
            if not summary['alerts']:
                summary['performance_status'] = 'Healthy'
            elif len(summary['alerts']) <= 2:
                summary['performance_status'] = 'Warning'
            else:
                summary['performance_status'] = 'Critical'
        
        return summary

def load_model_monitoring_data() -> Tuple[Dict, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load model results and any available test data for monitoring.
    
    Returns:
        Tuple of (model_results, y_true, y_prob)
    """
    try:
        # Load model results
        model_results = joblib.load('models/model_results.pkl')
        
        # Try to load test predictions if available
        y_true, y_prob = None, None
        
        # For demonstration, create synthetic test data
        # In production, this would come from actual model evaluations
        np.random.seed(42)
        n_samples = 100
        y_true = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        
        # Simulate realistic probability predictions
        y_prob = np.random.beta(2, 5, size=n_samples)
        y_prob[y_true == 1] = np.random.beta(5, 2, size=np.sum(y_true == 1))
        
        return model_results, y_true, y_prob
    
    except Exception as e:
        print(f"Error loading monitoring data: {e}")
        return {}, None, None

if __name__ == "__main__":
    # Example usage
    monitor = ModelMonitor()
    
    # Load monitoring data
    model_results, y_true, y_prob = load_model_monitoring_data()
    
    if model_results and y_true is not None:
        # Generate monitoring summary
        summary = monitor.generate_monitoring_summary(model_results, y_true, y_prob)
        print("Monitoring Summary:")
        print(f"Best model: {summary['best_model']}")
        print(f"Performance status: {summary['performance_status']}")
        print(f"Precision@10: {summary.get('precision_at_k', {}).get(10, 'N/A')}")
        
        if summary['alerts']:
            print("Alerts:")
            for alert in summary['alerts']:
                print(f"  {alert}")
    else:
        print("No monitoring data available")