import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import json
import os

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import load_data, format_currency, get_entity_history, calculate_risk_score
import joblib

# Import report generator
try:
    from report_generator import EntityReportGenerator
    REPORT_GENERATOR_AVAILABLE = True
except ImportError:
    REPORT_GENERATOR_AVAILABLE = False

# Import authentication
try:
    from auth import init_authentication, UserRole
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

# Configure page
st.set_page_config(
    page_title="Entity Profiles - Financial Irregularities",
    page_icon="🏢",
    layout="wide"
)

def load_entity_data():
    """Load data for entity analysis."""
    try:
        df = load_data("data/processed/model_dataset.parquet")
        return df
    except:
        return None

def load_model_results():
    """Load model results if available."""
    try:
        import joblib
        model = joblib.load("models/best_model.pkl")
        # Create mock results structure
        return {
            'Best Model': {
                'model': model,
                'roc_auc': 0.85  # Default value
            }
        }
    except:
        return None

def get_entity_list(df):
    """Get sorted list of unique entities."""
    if df is None or 'entity_name' not in df.columns:
        return []
    return sorted(df['entity_name'].unique())

def create_enhanced_risk_assessment(entity_data, model_results):
    """Create comprehensive risk assessment with confidence intervals and trends."""
    risk_assessments = []
    
    # Calculate risk for each year if possible
    for idx, row in entity_data.iterrows():
        assessment = calculate_risk_score(row, model_results or {})
        assessment['year'] = row.get('year', 'Unknown')
        risk_assessments.append(assessment)
    
    latest_assessment = risk_assessments[-1] if risk_assessments else {
        'risk_score': 0.5, 'risk_category': 'Unclassified', 'confidence': 0.5
    }
    
    return {
        'latest': latest_assessment,
        'historical': risk_assessments,
        'trend': 'stable'  # Could be enhanced with actual trend calculation
    }

def create_comprehensive_financial_kpis(entity_data):
    """Create comprehensive financial KPIs dashboard."""
    if len(entity_data) == 0:
        return {}
    
    latest = entity_data.iloc[-1]
    
    kpis = {
        'liquidity': {},
        'efficiency': {},
        'profitability': {},
        'leverage': {},
        'activity': {}
    }
    
    # Liquidity Ratios
    if 'current_assets' in latest and 'current_liabilities' in latest:
        kpis['liquidity']['current_ratio'] = latest['current_assets'] / (latest['current_liabilities'] + 1e-6)
    
    # Efficiency Ratios
    revenue_col = 'total_revenue' if 'total_revenue' in latest else 'revenue' if 'revenue' in latest else None
    expenditure_col = 'total_expenditure' if 'total_expenditure' in latest else 'expenditure' if 'expenditure' in latest else None
    
    if revenue_col and 'total_assets' in latest:
        kpis['efficiency']['asset_turnover'] = latest[revenue_col] / (latest['total_assets'] + 1e-6)
    
    if revenue_col and expenditure_col:
        kpis['efficiency']['expense_ratio'] = latest[expenditure_col] / (latest[revenue_col] + 1e-6)
    
    # Profitability Ratios
    if 'net_profit' in latest and revenue_col:
        kpis['profitability']['profit_margin'] = latest['net_profit'] / (latest[revenue_col] + 1e-6)
    
    if 'net_profit' in latest and 'total_assets' in latest:
        kpis['profitability']['roa'] = latest['net_profit'] / (latest['total_assets'] + 1e-6)
    
    # Leverage Ratios
    if 'total_liabilities' in latest and 'total_assets' in latest:
        kpis['leverage']['debt_ratio'] = latest['total_liabilities'] / (latest['total_assets'] + 1e-6)
    
    if 'total_liabilities' in latest and 'netassets' in latest:
        kpis['leverage']['debt_to_equity'] = latest['total_liabilities'] / (latest['netassets'] + 1e-6)
    
    # Activity Metrics
    kpis['activity']['years_active'] = len(entity_data)
    kpis['activity']['audit_flag_rate'] = entity_data['audit_flag'].mean() if 'audit_flag' in entity_data.columns else 0
    
    return kpis

def create_entity_summary_card(entity_data, risk_assessment):
    """Create enhanced summary card for entity with comprehensive metrics."""
    latest_data = entity_data.iloc[-1]
    enhanced_risk = create_enhanced_risk_assessment(entity_data, {})
    financial_kpis = create_comprehensive_financial_kpis(entity_data)
    
    # Enhanced risk color coding with confidence
    risk_info = enhanced_risk['latest']
    risk_color = {
        'High': '#DC143C',      # Crimson
        'Medium': '#FF8C00',    # Dark Orange 
        'Low': '#228B22',       # Forest Green
        'Unclassified': '#708090'    # Slate Gray
    }.get(risk_info.get('risk_category', 'Unclassified'), '#708090')
    
    confidence = risk_info.get('confidence', 0.5)
    confidence_color = '#228B22' if confidence > 0.8 else '#FF8C00' if confidence > 0.6 else '#DC143C'
    
    # Two rows of metrics for comprehensive display
    st.markdown("### 📊 Entity Overview")
    
    # First row: Core metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="🎯 Risk Score",
            value=f"{risk_info.get('risk_score', 0.5):.3f}",
            delta=f"±{1-confidence:.3f}",
            help="ML-predicted probability of irregularities with confidence interval"
        )
        st.markdown(f"<div style='text-align: center'><span style='color: {risk_color}; font-weight: bold; font-size: 14px'>{risk_info.get('risk_category', 'Unclassified')} Risk</span></div>", 
                   unsafe_allow_html=True)
    
    with col2:
        # Use flexible column naming
        revenue_col = 'total_revenue' if 'total_revenue' in latest_data else 'revenue' if 'revenue' in latest_data else None
        revenue = latest_data.get(revenue_col, 0) if revenue_col else 0
        prev_revenue = entity_data.iloc[-2].get(revenue_col, revenue) if len(entity_data) > 1 and revenue_col else revenue
        revenue_change = ((revenue - prev_revenue) / (prev_revenue + 1e-6)) * 100 if len(entity_data) > 1 and revenue_col else 0
        
        st.metric(
            label="💰 Total Revenue",
            value=format_currency(revenue),
            delta=f"{revenue_change:+.1f}%" if len(entity_data) > 1 and revenue_col else None,
            help="Latest reported total revenue with year-over-year change"
        )
    
    with col3:
        # Use flexible column naming
        expenditure_col = 'total_expenditure' if 'total_expenditure' in latest_data else 'expenditure' if 'expenditure' in latest_data else None
        expenditure = latest_data.get(expenditure_col, 0) if expenditure_col else 0
        prev_expenditure = entity_data.iloc[-2].get(expenditure_col, expenditure) if len(entity_data) > 1 and expenditure_col else expenditure
        exp_change = ((expenditure - prev_expenditure) / (prev_expenditure + 1e-6)) * 100 if len(entity_data) > 1 and expenditure_col else 0
        
        st.metric(
            label="💸 Total Expenditure",
            value=format_currency(expenditure),
            delta=f"{exp_change:+.1f}%" if len(entity_data) > 1 and expenditure_col else None,
            help="Latest reported total expenditure with year-over-year change"
        )
    
    with col4:
        audit_flags = entity_data['audit_flag'].sum() if 'audit_flag' in entity_data.columns else 0
        audit_rate = entity_data['audit_flag'].mean() if 'audit_flag' in entity_data.columns else 0
        audit_status = "🔴 Flagged" if latest_data.get('audit_flag', 0) == 1 else "🟢 Normal"
        
        st.metric(
            label="🚨 Audit Status",
            value=audit_status,
            delta=f"{audit_rate:.1%} rate",
            help=f"Current status with historical audit flag rate ({audit_flags} total flags)"
        )
    
    with col5:
        years_active = len(entity_data)
        latest_year = latest_data.get('year', 'Unknown')
        
        st.metric(
            label="📅 Activity Period",
            value=f"{years_active} years",
            delta=f"Until {latest_year}",
            help="Number of years with available data and latest reporting year"
        )
    
    # Second row: Financial KPIs
    st.markdown("### 📈 Key Financial Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        current_ratio = financial_kpis.get('liquidity', {}).get('current_ratio', None)
        if current_ratio is not None:
            ratio_color = "normal" if 1.0 <= current_ratio <= 3.0 else "inverse"
            st.metric(
                label="💧 Current Ratio",
                value=f"{current_ratio:.2f}",
                delta="Healthy" if 1.0 <= current_ratio <= 3.0 else "Review",
                delta_color=ratio_color,
                help="Current Assets / Current Liabilities (healthy range: 1.0-3.0)"
            )
        else:
            st.metric("💧 Current Ratio", "N/A", help="Data not available")
    
    with col2:
        debt_ratio = financial_kpis.get('leverage', {}).get('debt_ratio', None)
        if debt_ratio is not None:
            ratio_color = "normal" if debt_ratio <= 0.6 else "inverse"
            st.metric(
                label="⚖️ Debt Ratio",
                value=f"{debt_ratio:.2f}",
                delta="Good" if debt_ratio <= 0.6 else "High",
                delta_color=ratio_color,
                help="Total Debt / Total Assets (lower is better, >0.6 is high)"
            )
        else:
            st.metric("⚖️ Debt Ratio", "N/A", help="Data not available")
    
    with col3:
        profit_margin = financial_kpis.get('profitability', {}).get('profit_margin', None)
        if profit_margin is not None:
            margin_pct = profit_margin * 100
            margin_color = "normal" if profit_margin > 0 else "inverse"
            st.metric(
                label="📊 Profit Margin",
                value=f"{margin_pct:.1f}%",
                delta="Profitable" if profit_margin > 0 else "Loss",
                delta_color=margin_color,
                help="Net Profit / Total Revenue (positive indicates profitability)"
            )
        else:
            st.metric("📊 Profit Margin", "N/A", help="Data not available")
    
    with col4:
        asset_turnover = financial_kpis.get('efficiency', {}).get('asset_turnover', None)
        if asset_turnover is not None:
            efficiency_color = "normal" if asset_turnover > 0.5 else "inverse"
            st.metric(
                label="🔄 Asset Turnover",
                value=f"{asset_turnover:.2f}",
                delta="Efficient" if asset_turnover > 0.5 else "Low",
                delta_color=efficiency_color,
                help="Revenue / Total Assets (higher indicates better asset utilization)"
            )
        else:
            st.metric("🔄 Asset Turnover", "N/A", help="Data not available")
    
    with col5:
        expense_ratio = financial_kpis.get('efficiency', {}).get('expense_ratio', None)
        if expense_ratio is not None:
            exp_pct = expense_ratio * 100
            exp_color = "normal" if expense_ratio < 0.9 else "inverse"
            st.metric(
                label="💳 Expense Ratio",
                value=f"{exp_pct:.1f}%",
                delta="Controlled" if expense_ratio < 0.9 else "High",
                delta_color=exp_color,
                help="Total Expenses / Total Revenue (lower indicates better control)"
            )
        else:
            st.metric("💳 Expense Ratio", "N/A", help="Data not available")
    
    # Risk confidence indicator
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style='text-align: center; padding: 10px; background-color: rgba(0,0,0,0.05); border-radius: 10px'>
            <span style='color: {confidence_color}; font-weight: bold'>Model Confidence: {confidence:.1%}</span><br>
            <small>Risk assessment reliability based on data quality and model certainty</small>
        </div>
        """, unsafe_allow_html=True)

def create_financial_trends_plot(entity_data):
    """Create financial trends plot for entity."""
    if 'year' not in entity_data.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No temporal data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title="Financial Trends", height=400)
        return fig
    
    entity_data = entity_data.sort_values('year')
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Revenue and expenditure using flexible column detection
    revenue_col, expenditure_col = get_financial_columns(entity_data.columns)
    
    if revenue_col:
        fig.add_trace(
            go.Scatter(
                x=entity_data['year'],
                y=entity_data[revenue_col],
                mode='lines+markers',
                name='Revenue',
                line=dict(color='blue', width=3)
            ),
            secondary_y=False,
        )
    
    if expenditure_col:
        fig.add_trace(
            go.Scatter(
                x=entity_data['year'],
                y=entity_data[expenditure_col],
                mode='lines+markers',
                name='Expenditure',
                line=dict(color='red', width=3)
            ),
            secondary_y=False,
        )
    
    # Operating margin
    if revenue_col and expenditure_col:
        margin = (entity_data[revenue_col] - entity_data[expenditure_col]) / (entity_data[revenue_col] + 1e-6) * 100
        fig.add_trace(
            go.Scatter(
                x=entity_data['year'],
                y=margin,
                mode='lines+markers',
                name='Operating Margin %',
                line=dict(color='green', width=2, dash='dash')
            ),
            secondary_y=True,
        )
    
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Amount (GHS)", secondary_y=False)
    fig.update_yaxes(title_text="Operating Margin (%)", secondary_y=True)
    
    fig.update_layout(
        title="Financial Performance Trends",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def get_financial_columns(data):
    """Get flexible column names for revenue and expenditure."""
    revenue_col = None
    expenditure_col = None
    
    # Check for revenue columns
    if 'total_revenue' in data:
        revenue_col = 'total_revenue'
    elif 'revenue' in data:
        revenue_col = 'revenue'
    elif 'operational_revenue' in data:
        revenue_col = 'operational_revenue'
    
    # Check for expenditure columns
    if 'total_expenditure' in data:
        expenditure_col = 'total_expenditure'
    elif 'expenditure' in data:
        expenditure_col = 'expenditure'
    elif 'expenses' in data:
        expenditure_col = 'expenses'
    
    return revenue_col, expenditure_col

def load_shap_explainer():
    """Load SHAP explainer if available."""
    try:
        explainer = joblib.load("models/explainer.pkl")
        return explainer
    except:
        return None

def load_model_components():
    """Load model, preprocessor, and feature names for SHAP computation."""
    try:
        # Load trained model (correct artifact name)
        model = joblib.load("models/best_model.pkl")
        
        # Preprocessor is optional - many models don't need it
        preprocessor = None
        try:
            preprocessor = joblib.load("models/preprocessor.pkl")
        except:
            pass  # Continue without preprocessor
        
        # Load feature names if available
        try:
            feature_names = joblib.load("models/feature_names.pkl")
        except:
            # Fallback to common feature names
            feature_names = [
                'total_revenue', 'total_expenditure', 'total_assets', 'total_liabilities',
                'current_assets', 'current_liabilities', 'net_profit', 'debt_to_equity',
                'current_ratio', 'return_on_assets', 'profit_margin', 'asset_turnover'
            ]
        
        return model, preprocessor, feature_names
    except Exception as e:
        return None, None, None

def prepare_entity_features(entity_data, feature_names):
    """Prepare feature vector for SHAP computation."""
    try:
        # Create feature vector matching training format
        features = []
        for feature_name in feature_names:
            value = entity_data.get(feature_name, 0)
            # Handle missing values
            if pd.isna(value):
                value = 0
            features.append(float(value))
        
        return np.array(features).reshape(1, -1)
    except Exception as e:
        return None

def compute_entity_shap_values(model, preprocessor, entity_features, explainer):
    """Compute per-entity feature contributions (SHAP-like analysis)."""
    try:
        # Preprocess features if preprocessor available
        if preprocessor is not None:
            processed_features = preprocessor.transform(entity_features)
        else:
            processed_features = entity_features
        
        # Try to use loaded explainer if available (may be SHAP object)
        if explainer is not None:
            try:
                # Try SHAP explainer methods if SHAP is available
                if hasattr(explainer, 'shap_values'):
                    shap_values = explainer.shap_values(processed_features)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    return shap_values[0] if len(shap_values.shape) > 1 else shap_values
                elif hasattr(explainer, '__call__'):
                    result = explainer(processed_features)
                    if hasattr(result, 'values'):
                        values = result.values
                        return values[0] if len(values.shape) > 1 else values
                    return result
            except Exception as e:
                print(f"Explainer computation failed: {e}")
        
        # Compute per-entity feature contributions using model introspection
        if model is not None:
            try:
                # Get prediction for this entity - check model capabilities first
                if hasattr(model, 'predict_proba'):
                    prediction = model.predict_proba(processed_features)[0]
                    base_prob = 0.5  # Baseline probability
                    entity_prob = prediction[1] if len(prediction) > 1 else prediction[0]
                    prediction_delta = entity_prob - base_prob
                elif hasattr(model, 'decision_function'):
                    decision = model.decision_function(processed_features)[0]
                    # Convert decision to probability-like scale
                    prediction_delta = decision / (abs(decision) + 1.0)
                elif hasattr(model, 'predict'):
                    prediction = model.predict(processed_features)[0]
                    # Simple binary: 1 for positive class, 0 for negative
                    prediction_delta = prediction - 0.5
                else:
                    print("Model has no supported prediction method")
                    return None
                
                # Get feature contributions
                if hasattr(model, 'feature_importances_'):
                    # Tree-based models: Use feature importance weighted by values
                    feature_importances = model.feature_importances_
                    feature_values = processed_features[0]
                    
                    # Normalize feature values to [-1, 1] range for contribution calculation
                    max_abs_value = np.max(np.abs(feature_values)) + 1e-10
                    normalized_values = feature_values / max_abs_value
                    
                    # Calculate per-entity contributions
                    # Scale importance by feature value and prediction confidence
                    contributions = feature_importances * normalized_values * prediction_delta * 2
                    
                    return contributions
                
                elif hasattr(model, 'coef_'):
                    # Linear models: Use coefficients directly
                    if len(model.coef_.shape) > 1:
                        coefficients = model.coef_[0]  # Binary classification
                    else:
                        coefficients = model.coef_
                    
                    # Linear contribution is coef * feature_value
                    feature_values = processed_features[0]
                    contributions = coefficients * feature_values
                    
                    return contributions
                
                else:
                    # Generic model: Approximate using feature value correlation
                    feature_values = processed_features[0]
                    # Simple heuristic: higher values contribute more to higher predictions
                    mean_value = np.mean(np.abs(feature_values)) + 1e-10
                    normalized_values = feature_values / mean_value
                    contributions = normalized_values * prediction_delta
                    
                    return contributions
                
            except Exception as e:
                print(f"Model introspection failed: {e}")
        
        return None
    except Exception as e:
        print(f"Feature contribution computation error: {e}")
        return None

def create_global_feature_importance_plot(explainer, entity_data):
    """Fallback to global feature importance when per-entity SHAP unavailable."""
    try:
        if hasattr(explainer, 'feature_importance_'):
            feature_importance = explainer.feature_importance_
            feature_names = getattr(explainer, 'feature_names_', [f'Feature_{i}' for i in range(len(feature_importance))])
            top_features = sorted(zip(feature_names, feature_importance), key=lambda x: abs(x[1]), reverse=True)[:15]
        else:
            return create_risk_factors_plot(entity_data, None)
        
        if not top_features:
            return create_risk_factors_plot(entity_data, None)
        
        features, values = zip(*top_features)
        colors = ['#17becf' for _ in values]  # Neutral color for global importance
        
        fig = go.Figure([go.Bar(
            x=list(values),
            y=list(features),
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.3)', width=0.5)
            ),
            text=[f'{v:.3f}' for v in values],
            textposition='auto',
            hovertemplate=(
                '<b>%{y}</b><br>' +
                'Global Importance: %{x:.3f}<br>' +
                '<extra></extra>'
            )
        )])
        
        fig.update_layout(
            title={
                'text': "📊 Global Feature Importance<br><sub>Overall model feature importance (SHAP per-entity analysis unavailable)</sub>",
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Feature Importance",
            yaxis_title="Features",
            height=max(500, len(features) * 30),
            yaxis={'categoryorder': 'total ascending'},
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    except:
        return create_risk_factors_plot(entity_data, None)

def create_shap_explanation_plot(entity_data, explainer=None):
    """Create per-entity feature contribution plot for individual entity prediction."""
    if explainer is None:
        explainer = load_shap_explainer()
    
    try:
        # Load model and preprocessor for per-entity analysis
        model, preprocessor, feature_names = load_model_components()
        
        if model is None:
            # Fallback to basic risk factors when no model available
            return create_risk_factors_plot(entity_data, None)
        
        # Prepare feature vector for the latest entity data
        latest_data = entity_data.iloc[-1]
        entity_features = prepare_entity_features(latest_data, feature_names)
        
        if entity_features is None:
            return create_risk_factors_plot(entity_data, None)
        
        # Compute per-entity feature contributions
        contributions = compute_entity_shap_values(model, preprocessor, entity_features, explainer)
        
        if contributions is None:
            # Fallback to basic risk factors if all computation fails
            return create_risk_factors_plot(entity_data, None)
        
        # Validate feature names match contributions before plotting
        if len(feature_names) != len(contributions):
            print(f"Feature name/contribution length mismatch: {len(feature_names)} vs {len(contributions)}")
            return create_risk_factors_plot(entity_data, None)
        
        # Get top features by absolute contribution value
        feature_contribution_pairs = list(zip(feature_names, contributions))
        top_features = sorted(feature_contribution_pairs, key=lambda x: abs(x[1]), reverse=True)[:15]
        
        if not top_features:
            return create_risk_factors_plot(entity_data, None)
        
        features, values = zip(*top_features)
        
        # Color coding: red for positive (increasing risk), blue for negative (decreasing risk)
        colors = ['#DC143C' if v > 0 else '#1E90FF' for v in values]
        
        # Determine if we're using SHAP or model-based contributions
        is_shap = explainer is not None and hasattr(explainer, 'shap_values')
        method_name = "SHAP" if is_shap else "Model-Based"
        
        fig = go.Figure([go.Bar(
            x=list(values),
            y=list(features),
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.3)', width=0.5)
            ),
            text=[f'{v:+.3f}' for v in values],
            textposition='auto',
            hovertemplate=(
                '<b>%{y}</b><br>' +
                f'{method_name} Contribution: %{{x:+.3f}}<br>' +
                'Effect: %{customdata}<br>' +
                'Entity Value: %{customdata2}<br>' +
                '<extra></extra>'
            ),
            customdata=['Increases Risk' if v > 0 else 'Decreases Risk' for v in values],
            customdata2=[f'{latest_data.get(f, "N/A")}' for f in features]
        )])
        
        fig.update_layout(
            title={
                'text': f"🎯 Per-Entity {method_name} Analysis<br><sub>How each feature contributes to THIS entity's risk prediction (red=increases risk, blue=decreases risk)</sub>",
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title=f"{method_name} Contribution (Impact on Risk Score)",
            yaxis_title="Features",
            height=max(500, len(features) * 30),
            yaxis={'categoryorder': 'total ascending'},
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=2
            ),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
        
    except Exception as e:
        # Always fall back to basic risk factors on any error
        return create_risk_factors_plot(entity_data, None)

def create_risk_factors_plot(entity_data, feature_importance=None):
    """Create risk factors analysis plot."""
    if feature_importance is None:
        # Create simplified risk factors based on available data
        risk_factors = {}
        latest_data = entity_data.iloc[-1]
        
        # Financial ratios as risk factors - use flexible columns
        revenue_col, expenditure_col = get_financial_columns(latest_data.index)
        if revenue_col and expenditure_col and revenue_col in latest_data and expenditure_col in latest_data:
            expense_ratio = latest_data[expenditure_col] / (latest_data[revenue_col] + 1e-6)
            risk_factors['High Expense Ratio'] = min(expense_ratio, 2.0) * 0.5
        
        if 'total_assets' in latest_data and 'total_liabilities' in latest_data:
            debt_ratio = latest_data['total_liabilities'] / (latest_data['total_assets'] + 1e-6)
            risk_factors['High Debt Ratio'] = min(debt_ratio, 1.0)
        
        # Add audit history
        if 'audit_flag' in entity_data.columns:
            audit_history = entity_data['audit_flag'].mean()
            risk_factors['Historical Audit Issues'] = audit_history
        
        # Add volatility if multiple years
        if len(entity_data) > 1 and revenue_col:
            revenue_cv = entity_data[revenue_col].std() / (entity_data[revenue_col].mean() + 1e-6)
            risk_factors['Revenue Volatility'] = min(revenue_cv, 1.0)
    else:
        # Use actual feature importance
        risk_factors = dict(list(feature_importance.items())[:10])
    
    if not risk_factors:
        fig = go.Figure()
        fig.add_annotation(
            text="No risk factors data available",
            xref="paper", yref="paper", 
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title="Risk Factors Analysis", height=400)
        return fig
    
    factors = list(risk_factors.keys())
    values = list(risk_factors.values())
    
    # Color code by risk level
    colors = ['red' if v > 0.7 else 'orange' if v > 0.4 else 'green' for v in values]
    
    fig = go.Figure([go.Bar(
        x=values,
        y=factors,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.3f}' for v in values],
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Key Risk Factors",
        xaxis_title="Risk Score",
        yaxis_title="Risk Factor",
        height=max(300, len(factors) * 30),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_peer_comparison_plot(entity_data, df, entity_name):
    """Create enhanced bubble chart peer comparison with flexible column support."""
    if df is None or len(df) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for peer comparison",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title="Peer Comparison Analysis", height=500)
        return fig
    
    # Get flexible column names for the dataset
    revenue_col, expenditure_col = get_financial_columns(df.columns)
    
    if not revenue_col or not expenditure_col:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient financial data columns for comparison",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title="Peer Comparison Analysis", height=500)
        return fig
    
    # Enhanced entity type inference
    def get_entity_type(name):
        name_lower = name.lower()
        if any(word in name_lower for word in ['university', 'college', 'school', 'institute', 'polytechnic']):
            return 'Education'
        elif any(word in name_lower for word in ['hospital', 'clinic', 'health', 'medical']):
            return 'Health'
        elif any(word in name_lower for word in ['commission', 'authority', 'board', 'agency']):
            return 'Regulatory'
        elif any(word in name_lower for word in ['company', 'limited', 'ltd', 'corporation', 'enterprise']):
            return 'State Enterprise'
        elif any(word in name_lower for word in ['ministry', 'department', 'secretariat']):
            return 'Government'
        elif any(word in name_lower for word in ['bank', 'financial', 'insurance']):
            return 'Financial Services'
        else:
            return 'Other'
    
    entity_type = get_entity_type(entity_name)
    
    # Get peer entities of same type
    peer_entities = []
    for other_entity in df['entity_name'].unique():
        if other_entity != entity_name and get_entity_type(other_entity) == entity_type:
            peer_entities.append(other_entity)
    
    # If no peers of same type, use top entities by expenditure
    if len(peer_entities) < 5:
        all_entities = df.groupby('entity_name')[expenditure_col].sum().sort_values(ascending=False)
        peer_entities = [e for e in all_entities.index[:20] if e != entity_name]
    
    peer_entities = peer_entities[:15]  # Limit to 15 peers for readability
    
    # Calculate comprehensive metrics for comparison
    comparison_data = []
    
    # Add current entity
    current_latest = entity_data.iloc[-1]
    comparison_data.append({
        'Entity': entity_name,
        'Type': entity_type,
        'Revenue': current_latest.get(revenue_col, 0),
        'Expenditure': current_latest.get(expenditure_col, 0),
        'Assets': current_latest.get('total_assets', 0),
        'Audit_Flags': entity_data['audit_flag'].sum() if 'audit_flag' in entity_data.columns else 0,
        'Audit_Rate': entity_data['audit_flag'].mean() if 'audit_flag' in entity_data.columns else 0,
        'Years_Active': len(entity_data),
        'Is_Current': True
    })
    
    # Add peer entities  
    for peer in peer_entities:
        peer_df = df[df['entity_name'] == peer]
        if len(peer_df) > 0:
            peer_latest = peer_df.iloc[-1]
            comparison_data.append({
                'Entity': peer,
                'Type': get_entity_type(peer),
                'Revenue': peer_latest.get(revenue_col, 0),
                'Expenditure': peer_latest.get(expenditure_col, 0),
                'Assets': peer_latest.get('total_assets', 0),
                'Audit_Flags': peer_df['audit_flag'].sum() if 'audit_flag' in peer_df.columns else 0,
                'Audit_Rate': peer_df['audit_flag'].mean() if 'audit_flag' in peer_df.columns else 0,
                'Years_Active': len(peer_df),
                'Is_Current': False
            })
    
    if len(comparison_data) <= 1:
        fig = go.Figure()
        fig.add_annotation(
            text="No peer data available for comparison",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title="Peer Comparison Analysis", height=500)
        return fig
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create sophisticated bubble chart: Revenue vs Expenditure, size = Assets, color = Audit Rate
    fig = go.Figure()
    
    # Plot peers as bubble chart
    peer_data = comparison_df[~comparison_df['Is_Current']]
    if len(peer_data) > 0:
        # Calculate bubble sizes (scale based on assets)
        max_assets = comparison_df['Assets'].max()
        min_size, max_size = 10, 40
        sizes = []
        for assets in peer_data['Assets']:
            if max_assets > 0:
                normalized_size = (assets / max_assets) * (max_size - min_size) + min_size
            else:
                normalized_size = min_size
            sizes.append(max(min_size, min(max_size, normalized_size)))
        
        fig.add_trace(go.Scatter(
            x=peer_data['Revenue'],
            y=peer_data['Expenditure'],
            mode='markers',
            marker=dict(
                size=sizes,
                color=peer_data['Audit_Rate'],
                colorscale=[
                    [0, '#2E8B57'],      # Green for low audit rate
                    [0.3, '#FFD700'],    # Gold for medium-low
                    [0.6, '#FF8C00'],    # Orange for medium-high
                    [1, '#DC143C']       # Red for high audit rate
                ],
                colorbar=dict(
                    title="Audit<br>Rate",
                    tickformat=".0%"
                ),
                line=dict(width=1, color='rgba(0,0,0,0.4)'),
                opacity=0.8,
                showscale=True
            ),
            text=[f"{entity}" for entity in peer_data['Entity']],
            name='Peer Entities',
            hovertemplate=(
                '<b>%{text}</b><br>' +
                f'{revenue_col.title()}: GHS %{{x:,.0f}}<br>' +
                f'{expenditure_col.title()}: GHS %{{y:,.0f}}<br>' +
                'Assets: GHS %{customdata[0]:,.0f}<br>' +
                'Audit Rate: %{customdata[1]:.1%}<br>' +
                'Total Flags: %{customdata[2]}<br>' +
                'Years Active: %{customdata[3]}<br>' +
                'Type: %{customdata[4]}<br>' +
                '<extra></extra>'
            ),
            customdata=list(zip(
                peer_data['Assets'], 
                peer_data['Audit_Rate'], 
                peer_data['Audit_Flags'],
                peer_data['Years_Active'],
                peer_data['Type']
            ))
        ))
    
    # Highlight current entity with distinctive star marker
    current_data = comparison_df[comparison_df['Is_Current']].iloc[0]
    current_size = 50  # Fixed large size for prominence
    
    fig.add_trace(go.Scatter(
        x=[current_data['Revenue']],
        y=[current_data['Expenditure']],
        mode='markers',
        marker=dict(
            size=current_size,
            color='#FFD700',  # Gold color
            symbol='star',
            line=dict(width=4, color='#000000'),  # Black border
            opacity=1.0
        ),
        name=f'{entity_name} (Current Entity)',
        hovertemplate=(
            '<b>%{text} ⭐ CURRENT ENTITY</b><br>' +
            f'{revenue_col.title()}: GHS %{{x:,.0f}}<br>' +
            f'{expenditure_col.title()}: GHS %{{y:,.0f}}<br>' +
            'Assets: GHS %{customdata[0]:,.0f}<br>' +
            'Audit Rate: %{customdata[1]:.1%}<br>' +
            'Total Flags: %{customdata[2]}<br>' +
            'Years Active: %{customdata[3]}<br>' +
            'Type: %{customdata[4]}<br>' +
            '<extra></extra>'
        ),
        text=[entity_name],
        customdata=[[
            current_data['Assets'], 
            current_data['Audit_Rate'], 
            current_data['Audit_Flags'],
            current_data['Years_Active'],
            current_data['Type']
        ]]
    ))
    
    # Add reference lines for better context
    if len(peer_data) > 0:
        # Add median lines for peer comparison
        median_revenue = peer_data['Revenue'].median()
        median_expenditure = peer_data['Expenditure'].median()
        
        # Vertical median line (revenue)
        fig.add_vline(
            x=median_revenue, 
            line_dash="dash", 
            line_color="gray", 
            opacity=0.5,
            annotation_text="Peer Median Revenue",
            annotation_position="top"
        )
        
        # Horizontal median line (expenditure)
        fig.add_hline(
            y=median_expenditure, 
            line_dash="dash", 
            line_color="gray", 
            opacity=0.5,
            annotation_text="Peer Median Expenditure",
            annotation_position="right"
        )
    
    fig.update_layout(
        title={
            'text': f"👥 Enhanced Peer Comparison: {entity_name}<br><sub>{revenue_col.title()} vs {expenditure_col.title()} • Bubble size = Assets • Color = Audit risk rate</sub>",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title=f"{revenue_col.replace('_', ' ').title()} (GHS)",
        yaxis_title=f"{expenditure_col.replace('_', ' ').title()} (GHS)",
        height=700,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        plot_bgcolor='rgba(250,250,250,0.8)',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        )
    )
    
    return fig
    
    # Create comparison plot for financial metrics
    fig = go.Figure()
    
    if 'revenue' in available_metrics:
        fig.add_trace(go.Bar(
            name='Revenue (Peers)',
            x=peer_df['entity'],
            y=peer_df['revenue'],
            marker_color='lightblue',
            opacity=0.7
        ))
        
        # Add current entity as highlighted bar
        fig.add_trace(go.Bar(
            name=f'Revenue ({entity_name})',
            x=[entity_name],
            y=[entity_latest.get('revenue', 0)],
            marker_color='blue'
        ))
    
    fig.update_layout(
        title=f"Revenue Comparison: {entity_name} vs Peers",
        xaxis_title="Entity",
        yaxis_title="Revenue (GHS)",
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig

def create_audit_history_plot(entity_data):
    """Create audit history timeline."""
    if 'year' not in entity_data.columns or 'audit_flag' not in entity_data.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No audit history data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title="Audit History", height=300)
        return fig
    
    entity_data = entity_data.sort_values('year')
    
    # Create timeline
    colors = ['red' if flag == 1 else 'green' for flag in entity_data['audit_flag']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=entity_data['year'],
        y=entity_data['audit_flag'],
        mode='markers+lines',
        marker=dict(
            size=15,
            color=colors,
            line=dict(width=2, color='black')
        ),
        line=dict(color='gray', width=2),
        name='Audit Status'
    ))
    
    fig.update_layout(
        title="Audit History Timeline",
        xaxis_title="Year",
        yaxis_title="Audit Status",
        yaxis=dict(
            tickvals=[0, 1],
            ticktext=['Normal', 'Flagged']
        ),
        height=300,
        showlegend=False
    )
    
    return fig

def main():
    # Initialize authentication for user context (optional login)
    if AUTH_AVAILABLE:
        auth_manager = init_authentication()
        if auth_manager.is_authenticated():
            auth_manager.show_user_info()  # Show user info if logged in
    
    st.title("🏢 Entity Risk Profiles")
    st.markdown("Detailed analysis of individual state entities")
    
    # Load data
    df = load_entity_data()
    model_results = load_model_results()
    
    if df is None:
        st.error("""
        **No entity data available.**
        
        Please run the main application first to process your dataset.
        """)
        return
    
    # Entity selection
    entities = get_entity_list(df)
    if not entities:
        st.error("No entities found in the dataset.")
        return
    
    selected_entity = st.selectbox(
        "🔍 Search and Select Entity",
        entities,
        help="Select an entity to view its detailed risk profile"
    )
    
    if not selected_entity:
        st.info("Please select an entity to view its profile.")
        return
    
    # Get entity data
    entity_data = get_entity_history(df, selected_entity)
    
    if len(entity_data) == 0:
        st.error(f"No data found for entity: {selected_entity}")
        return
    
    # Calculate risk assessment
    risk_assessment = calculate_risk_score(entity_data.iloc[-1], model_results or {})
    
    st.markdown(f"## 📊 Profile: {selected_entity}")
    st.markdown("---")
    
    # Summary card
    create_entity_summary_card(entity_data, risk_assessment)
    
    st.markdown("---")
    
    # Main analysis sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Financial Trends", 
        "⚠️ Risk Analysis", 
        "👥 Peer Comparison", 
        "📋 Detailed History"
    ])
    
    with tab1:
        st.plotly_chart(create_financial_trends_plot(entity_data), config={'responsive': True})
        
        # Financial ratios table
        if len(entity_data) > 0:
            st.markdown("### Key Financial Ratios")
            latest_data = entity_data.iloc[-1]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'revenue' in latest_data and 'expenditure' in latest_data:
                    expense_ratio = latest_data['expenditure'] / (latest_data['revenue'] + 1e-6)
                    st.metric("Expense Ratio", f"{expense_ratio:.2f}")
                
                if 'total_assets' in latest_data and 'total_liabilities' in latest_data:
                    debt_ratio = latest_data['total_liabilities'] / (latest_data['total_assets'] + 1e-6)
                    st.metric("Debt Ratio", f"{debt_ratio:.2f}")
            
            with col2:
                if 'revenue' in latest_data and 'total_assets' in latest_data:
                    asset_turnover = latest_data['revenue'] / (latest_data['total_assets'] + 1e-6)
                    st.metric("Asset Turnover", f"{asset_turnover:.2f}")
                
                if 'cash' in latest_data and 'total_liabilities' in latest_data:
                    cash_ratio = latest_data['cash'] / (latest_data['total_liabilities'] + 1e-6)
                    st.metric("Cash Ratio", f"{cash_ratio:.2f}")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # This will now show SHAP if available, or fall back to basic risk factors
            shap_plot = create_shap_explanation_plot(entity_data)
            st.plotly_chart(shap_plot, config={'responsive': True})
        
        with col2:
            st.plotly_chart(create_audit_history_plot(entity_data), config={'responsive': True})
        
        # Risk explanation
        st.markdown("### 🔍 Risk Assessment Explanation")
        
        risk_score = risk_assessment.get('risk_score', 0.5)
        risk_category = risk_assessment.get('risk_category', 'Unclassified')
        
        if risk_score > 0.7:
            st.error(f"""
            **High Risk Entity** (Score: {risk_score:.3f})
            
            This entity shows significant indicators of potential financial irregularities:
            - Risk score above 70% threshold
            - Requires immediate audit attention
            - Enhanced monitoring recommended
            """)
        elif risk_score > 0.3:
            st.warning(f"""
            **Medium Risk Entity** (Score: {risk_score:.3f})
            
            This entity shows moderate risk indicators:
            - Risk score between 30-70%
            - Periodic review recommended
            - Monitor key risk factors
            """)
        else:
            st.success(f"""
            **Low Risk Entity** (Score: {risk_score:.3f})
            
            This entity shows minimal risk indicators:
            - Risk score below 30%
            - Standard audit procedures sufficient
            - Continue routine monitoring
            """)
    
    with tab3:
        st.plotly_chart(create_peer_comparison_plot(entity_data, df, selected_entity), config={'responsive': True})
        
        # Peer ranking
        st.markdown("### 📊 Risk Ranking Among Peers")
        
        # Calculate risk scores for all entities (simplified)
        entity_risks = []
        for entity in df['entity_name'].unique():
            entity_df = df[df['entity_name'] == entity]
            if len(entity_df) > 0:
                latest = entity_df.iloc[-1]
                risk = calculate_risk_score(latest, model_results or {})
                entity_risks.append({
                    'Entity': entity,
                    'Risk Score': risk.get('risk_score', 0.5),
                    'Risk Category': risk.get('risk_category', 'Unclassified')
                })
        
        risk_df = pd.DataFrame(entity_risks).sort_values('Risk Score', ascending=False)
        
        # Find current entity rank
        current_rank = risk_df[risk_df['Entity'] == selected_entity].index[0] + 1
        total_entities = len(risk_df)
        
        st.info(f"**{selected_entity}** ranks **#{current_rank}** out of {total_entities} entities by risk score")
        
        # Show top and bottom entities
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔴 Highest Risk Entities**")
            st.dataframe(risk_df.head(10), hide_index=True)
        
        with col2:
            st.markdown("**🟢 Lowest Risk Entities**")
            st.dataframe(risk_df.tail(10).iloc[::-1], hide_index=True)
    
    with tab4:
        st.markdown("### 📅 Complete Entity History")
        
        # Display full history table
        display_columns = ['year']
        if 'audit_flag' in entity_data.columns:
            display_columns.append('audit_flag')
        if 'revenue' in entity_data.columns:
            display_columns.append('revenue')
        if 'expenditure' in entity_data.columns:
            display_columns.append('expenditure')
        if 'total_assets' in entity_data.columns:
            display_columns.append('total_assets')
        
        available_columns = [col for col in display_columns if col in entity_data.columns]
        
        if available_columns:
            history_display = entity_data[available_columns].copy()
            
            # Format currency columns
            currency_cols = ['revenue', 'expenditure', 'total_assets']
            for col in currency_cols:
                if col in history_display.columns:
                    history_display[col] = history_display[col].apply(lambda x: format_currency(x) if pd.notnull(x) else "N/A")
            
            st.dataframe(history_display.sort_values('year' if 'year' in history_display.columns else history_display.columns[0], ascending=False), 
                        hide_index=True)
        
        # Additional entity information
        if 'detected_issues' in entity_data.columns:
            issues = entity_data['detected_issues'].dropna()
            if len(issues) > 0:
                st.markdown("### ⚠️ Detected Issues")
                for idx, issue in issues.items():
                    if issue and str(issue).strip():
                        year = entity_data.loc[idx, 'year'] if 'year' in entity_data.columns else 'Unknown'
                        st.write(f"**{year}:** {issue}")
    
    # Export options
    st.markdown("---")
    st.markdown("### 📤 Export Entity Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Download Entity Data"):
            csv_data = entity_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"{selected_entity.replace(' ', '_')}_profile_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if REPORT_GENERATOR_AVAILABLE:
            if st.button("📄 Generate PDF Report"):
                with st.spinner("🔄 Generating comprehensive PDF report..."):
                    try:
                        generator = EntityReportGenerator()
                        report_path = generator.generate_entity_report(selected_entity)
                        
                        if os.path.exists(report_path):
                            with open(report_path, 'rb') as file:
                                st.download_button(
                                    label="📥 Download PDF Report",
                                    data=file.read(),
                                    file_name=f"{selected_entity.replace(' ', '_')}_comprehensive_report_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
                                    mime="application/pdf",
                                    key="pdf_download"
                                )
                            st.success("✅ Comprehensive PDF report generated successfully!")
                        else:
                            st.error("❌ Error generating PDF report")
                            
                    except Exception as e:
                        st.error(f"❌ Error generating PDF report: {str(e)}")
                        st.info("💡 Falling back to basic text report...")
                        
                        # Fallback to basic report
                        basic_report = f"""# Entity Risk Profile Report

**Entity:** {selected_entity}
**Report Date:** {pd.Timestamp.now().strftime('%B %d, %Y')}

## Risk Assessment
- **Risk Score:** {risk_assessment.get('risk_score', 0.5):.3f}
- **Risk Category:** {risk_assessment.get('risk_category', 'Unclassified')}

## Financial Summary (Latest Year)
- **Revenue:** {format_currency(entity_data.iloc[-1].get('total_revenue', 0))}
- **Expenditure:** {format_currency(entity_data.iloc[-1].get('total_expenditure', 0))}

## Recommendations
Based on the risk assessment, enhanced monitoring is recommended.
"""
                        st.download_button(
                            label="📄 Download Basic Report",
                            data=basic_report,
                            file_name=f"{selected_entity.replace(' ', '_')}_report_{pd.Timestamp.now().strftime('%Y%m%d')}.md",
                            mime="text/markdown"
                        )
        else:
            if st.button("📋 Generate Basic Report"):
                basic_report = f"""# Entity Risk Profile Report

**Entity:** {selected_entity}
**Report Date:** {pd.Timestamp.now().strftime('%B %d, %Y')}

## Risk Assessment
- **Risk Score:** {risk_assessment.get('risk_score', 0.5):.3f}
- **Risk Category:** {risk_assessment.get('risk_category', 'Unclassified')}

## Financial Summary (Latest Year)  
- **Revenue:** {format_currency(entity_data.iloc[-1].get('total_revenue', 0))}
- **Expenditure:** {format_currency(entity_data.iloc[-1].get('total_expenditure', 0))}

## Recommendations
Based on the risk assessment, monitoring is recommended.
"""
                st.download_button(
                    label="Download Report",
                    data=basic_report,
                    file_name=f"{selected_entity.replace(' ', '_')}_report_{pd.Timestamp.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )

if __name__ == "__main__":
    main()
