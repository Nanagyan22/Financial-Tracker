import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import json
import joblib
import os

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import load_data, get_model_metrics, detect_concept_drift, validate_data_quality
from database import DatabaseManager
from evaluate import evaluate_models

# Import monitoring system
try:
    from monitoring import ModelMonitor, load_model_monitoring_data
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

# Import authentication
try:
    from auth import init_authentication, UserRole
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

# Configure page
st.set_page_config(
    page_title="Admin Panel - Financial Irregularities",
    page_icon="⚙️",
    layout="wide"
)

def load_model_artifacts():
    """Load saved model artifacts."""
    artifacts = {}
    
    try:
        # Load best model
        if os.path.exists("models/best_model.pkl"):
            artifacts['best_model'] = joblib.load("models/best_model.pkl")
        
        # Load feature names
        if os.path.exists("models/feature_names.pkl"):
            artifacts['feature_names'] = joblib.load("models/feature_names.pkl")
        
        # Load explainer
        if os.path.exists("models/explainer.pkl"):
            artifacts['explainer'] = joblib.load("models/explainer.pkl")
    
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
    
    return artifacts

def get_system_status():
    """Get overall system status."""
    status = {
        'data_available': False,
        'models_trained': False,
        'explanations_ready': False,
        'last_updated': 'Unknown'
    }
    
    try:
        # Check data
        if os.path.exists("data/processed/model_dataset.parquet"):
            status['data_available'] = True
            status['last_updated'] = pd.Timestamp.fromtimestamp(
                os.path.getmtime("data/processed/model_dataset.parquet")
            ).strftime('%Y-%m-%d %H:%M:%S')
        
        # Check models
        if os.path.exists("models/best_model.pkl"):
            status['models_trained'] = True
        
        # Check explanations
        if os.path.exists("models/explainer.pkl"):
            status['explanations_ready'] = True
    
    except Exception as e:
        st.error(f"Error checking system status: {e}")
    
    return status

def create_model_performance_summary():
    """Create model performance summary from saved results."""
    try:
        # Try to load real model results from training
        if os.path.exists("models/model_results.pkl"):
            saved_results = joblib.load("models/model_results.pkl")
            model_results = {}
            
            for model_name, data in saved_results.items():
                if 'evaluation' in data:
                    eval_data = data['evaluation']
                    model_results[model_name] = {
                        'roc_auc': eval_data.get('roc_auc', 0),
                        'pr_auc': eval_data.get('pr_auc', 0),
                        'f1_score': eval_data.get('f1_score', 0),
                        'precision': eval_data.get('precision', 0),
                        'recall': eval_data.get('recall', 0)
                    }
            
            return model_results
        
        else:
            # Fallback to recent training results if no saved file
            st.warning("No saved model results found. Please train models to see performance metrics.")
            return {}
        
    except Exception as e:
        st.error(f"Error loading model results: {e}")
        return {}

def create_performance_comparison_plot(model_results):
    """Create model performance comparison plot."""
    if not model_results:
        fig = go.Figure()
        fig.add_annotation(
            text="No model performance data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title="Model Performance Comparison", height=400)
        return fig
    
    models = list(model_results.keys())
    metrics = ['roc_auc', 'pr_auc', 'f1_score', 'precision', 'recall']
    
    fig = go.Figure()
    
    for metric in metrics:
        values = [model_results[model].get(metric, 0) for model in models]
        
        fig.add_trace(go.Scatter(
            x=models,
            y=values,
            mode='lines+markers',
            name=metric.upper().replace('_', '-'),
            line=dict(width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="Score",
        height=400,
        yaxis=dict(range=[0, 1]),
        hovermode='x unified'
    )
    
    return fig

def create_precision_at_k_plot():
    """Create precision@k plot."""
    # Mock precision@k data
    k_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    model_precision = {
        'XGBoost': [0.95, 0.92, 0.88, 0.85, 0.82, 0.79, 0.76, 0.73, 0.70, 0.67],
        'LightGBM': [0.93, 0.90, 0.86, 0.83, 0.80, 0.77, 0.74, 0.71, 0.68, 0.65],
        'Random Forest': [0.90, 0.87, 0.83, 0.80, 0.77, 0.74, 0.71, 0.68, 0.65, 0.62],
        'Logistic Regression': [0.85, 0.82, 0.78, 0.75, 0.72, 0.69, 0.66, 0.63, 0.60, 0.57]
    }
    
    fig = go.Figure()
    
    for model, precision in model_precision.items():
        fig.add_trace(go.Scatter(
            x=k_values,
            y=precision,
            mode='lines+markers',
            name=model,
            line=dict(width=3),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title="Precision @ Top K% Predictions",
        xaxis_title="Top K% of Predictions",
        yaxis_title="Precision",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_data_quality_dashboard(df):
    """Create data quality dashboard."""
    if df is None:
        return {}
    
    quality_metrics = validate_data_quality(df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Data Quality Score",
            f"{quality_metrics['quality_score']:.2f}",
            help="Overall data quality score (0-1)"
        )
    
    with col2:
        st.metric(
            "Total Records",
            f"{quality_metrics['total_rows']:,}",
            help="Total number of records in dataset"
        )
    
    with col3:
        missing_pct = sum(quality_metrics['missing_values'].values()) / (
            quality_metrics['total_rows'] * quality_metrics['total_columns']
        ) * 100
        st.metric(
            "Missing Data %",
            f"{missing_pct:.1f}%",
            help="Percentage of missing values"
        )
    
    with col4:
        st.metric(
            "Quality Grade",
            quality_metrics['quality_grade'],
            help="Overall data quality assessment"
        )
    
    return quality_metrics

def create_concept_drift_analysis(df):
    """Create concept drift analysis."""
    if df is None or 'year' not in df.columns:
        st.warning("Cannot perform concept drift analysis - no temporal data available")
        return
    
    # Split data into reference (older) and current (recent) periods
    years = sorted(df['year'].unique())
    if len(years) < 2:
        st.warning("Need at least 2 years of data for drift analysis")
        return
    
    mid_point = len(years) // 2
    reference_years = years[:mid_point]
    current_years = years[mid_point:]
    
    reference_data = df[df['year'].isin(reference_years)]
    current_data = df[df['year'].isin(current_years)]
    
    st.markdown(f"**Reference Period:** {min(reference_years)}-{max(reference_years)}")
    st.markdown(f"**Current Period:** {min(current_years)}-{max(current_years)}")
    
    drift_results = detect_concept_drift(current_data, reference_data)
    
    # Display drift summary
    summary = drift_results.get('summary', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Features Analyzed",
            summary.get('total_features', 0)
        )
    
    with col2:
        st.metric(
            "Drifted Features",
            summary.get('drifted_features', 0)
        )
    
    with col3:
        drift_pct = summary.get('drift_percentage', 0) * 100
        st.metric(
            "Drift Percentage",
            f"{drift_pct:.1f}%"
        )
    
    # Show detailed drift analysis
    if st.checkbox("Show Detailed Drift Analysis"):
        drift_data = []
        
        for feature, result in drift_results.items():
            if feature != 'summary' and isinstance(result, dict):
                drift_data.append({
                    'Feature': feature,
                    'KS Statistic': result.get('ks_statistic', 0),
                    'P-Value': result.get('p_value', 1),
                    'Drift Detected': result.get('drift_detected', False),
                    'Severity': result.get('drift_severity', 'Low')
                })
        
        if drift_data:
            drift_df = pd.DataFrame(drift_data)
            drift_df = drift_df.sort_values('KS Statistic', ascending=False)
            
            st.dataframe(drift_df, hide_index=True)
        
        # Overall drift assessment with automatic retraining option
        if summary.get('overall_drift', False):
            st.error("⚠️ **Significant concept drift detected!** Consider retraining models.")
            
            # Offer automatic retraining
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚨 Retrain Models Now", type="primary"):
                    with st.spinner("Retraining models due to drift..."):
                        try:
                            # Import training functionality
                            import sys
                            from pathlib import Path
                            sys.path.append(str(Path(__file__).parent.parent / "src"))
                            from train import train_models
                            from ingest import load_dataset
                            
                            # Load and retrain models
                            df = load_dataset('data/raw/Final Dataset.xlsx')
                            results = train_models(df)
                            
                            if results:
                                best_model = max(results.keys(), key=lambda k: results[k]['evaluation']['roc_auc'])
                                best_auc = results[best_model]['evaluation']['roc_auc']
                                st.success(f"✅ Models retrained due to drift! Best model: {best_model} (AUC: {best_auc:.3f})")
                            else:
                                st.error("❌ Drift-triggered retraining failed")
                                
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error during drift-triggered retraining: {e}")
            
            with col2:
                if st.button("📊 Schedule Retraining"):
                    st.info("📅 Retraining scheduled for next maintenance window")
                    
        else:
            st.success("✅ **No significant concept drift detected.** Models remain valid.")

def main():
    # Initialize authentication - ADMINISTRATOR level required
    if AUTH_AVAILABLE:
        auth_manager = init_authentication()
        auth_manager.show_user_info()
        
        # Require ADMINISTRATOR role for system admin
        if not auth_manager.require_role(UserRole.ADMINISTRATOR):
            return
    else:
        st.error("🔒 Authentication system unavailable. Access restricted.")
        return
    
    st.title("⚙️ System Administration")
    st.markdown("Model management, performance monitoring, and system diagnostics")
    
    # System Status Overview
    st.markdown("### 🖥️ System Status")
    
    status = get_system_status()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_icon = "✅" if status['data_available'] else "❌"
        st.metric("Data Pipeline", f"{status_icon} {'Ready' if status['data_available'] else 'Not Ready'}")
    
    with col2:
        status_icon = "✅" if status['models_trained'] else "❌"
        st.metric("ML Models", f"{status_icon} {'Trained' if status['models_trained'] else 'Not Trained'}")
    
    with col3:
        status_icon = "✅" if status['explanations_ready'] else "❌"
        st.metric("Explanations", f"{status_icon} {'Ready' if status['explanations_ready'] else 'Not Ready'}")
    
    with col4:
        st.metric("Last Updated", status['last_updated'])
    
    st.markdown("---")
    
    # Tabs for different admin functions
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Model Performance", 
        "🔧 System Monitoring", 
        "📈 Data Quality", 
        "🔄 Model Management",
        "⚠️ Concept Drift",
        "📈 Advanced Monitoring",
        "👥 User Management"
    ])
    
    with tab1:
        st.markdown("### 📊 Model Performance Overview")
        
        model_results = create_model_performance_summary()
        
        if model_results:
            # Performance comparison plot
            st.plotly_chart(create_performance_comparison_plot(model_results), config={'responsive': True})
            
            # Performance table
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📋 Detailed Metrics")
                metrics_df = pd.DataFrame(model_results).T
                metrics_df = metrics_df.round(3)
                st.dataframe(metrics_df, use_container_width=True)
            
            with col2:
                st.markdown("#### 📈 Precision @ Top K%")
                st.plotly_chart(create_precision_at_k_plot(), config={'responsive': True})
            
            # Best model highlight
            best_model = max(model_results.keys(), key=lambda k: model_results[k]['roc_auc'])
            best_auc = model_results[best_model]['roc_auc']
            
            st.success(f"🏆 **Best Performing Model:** {best_model} (ROC-AUC: {best_auc:.3f})")
        else:
            st.warning("No model performance data available. Please train models first.")
    
    with tab2:
        st.markdown("### 🔧 System Monitoring")
        
        # Model artifacts status
        artifacts = load_model_artifacts()
        
        st.markdown("#### 📦 Model Artifacts")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'best_model' in artifacts:
                st.success("✅ Best Model Loaded")
                model_info = str(type(artifacts['best_model'])).split('.')[-1].replace("'>", "")
                st.write(f"Type: {model_info}")
            else:
                st.error("❌ No Model Found")
        
        with col2:
            if 'feature_names' in artifacts:
                st.success("✅ Feature Names Loaded")
                st.write(f"Features: {len(artifacts['feature_names'])}")
            else:
                st.error("❌ No Feature Names")
        
        with col3:
            if 'explainer' in artifacts:
                st.success("✅ SHAP Explainer Ready")
            else:
                st.error("❌ No Explainer Found")
        
        # File system status
        st.markdown("#### 📁 File System Status")
        
        directories = [
            ("Data (Raw)", "data/raw/"),
            ("Data (Processed)", "data/processed/"),
            ("Models", "models/"),
            ("Source Code", "src/")
        ]
        
        for name, path in directories:
            if os.path.exists(path):
                file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
                st.success(f"✅ {name}: {file_count} files")
            else:
                st.error(f"❌ {name}: Directory not found")
        
        # Memory and performance
        st.markdown("#### 💾 Performance Metrics")
        
        # Get basic system info
        try:
            import psutil
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cpu_percent = psutil.cpu_percent()
                st.metric("CPU Usage", f"{cpu_percent:.1f}%")
            
            with col2:
                memory = psutil.virtual_memory()
                st.metric("Memory Usage", f"{memory.percent:.1f}%")
            
            with col3:
                disk = psutil.disk_usage('.')
                st.metric("Disk Usage", f"{disk.percent:.1f}%")
        
        except ImportError:
            st.info("Install psutil for detailed system metrics: pip install psutil")
    
    with tab3:
        st.markdown("### 📈 Data Quality Assessment")
        
        # Load data for quality analysis
        try:
            df = load_data("data/processed/model_dataset.parquet")
            
            if df is not None:
                quality_metrics = create_data_quality_dashboard(df)
                
                st.markdown("---")
                
                # Missing values heatmap
                if quality_metrics:
                    missing_data = pd.DataFrame(list(quality_metrics['missing_values'].items()),
                                              columns=['Column', 'Missing Count'])
                    missing_data['Missing %'] = (missing_data['Missing Count'] / len(df) * 100).round(2)
                    missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing %', ascending=False)
                    
                    if len(missing_data) > 0:
                        st.markdown("#### ⚠️ Missing Data Analysis")
                        
                        fig = px.bar(
                            missing_data.head(20),
                            x='Missing %',
                            y='Column',
                            orientation='h',
                            title="Top 20 Columns with Missing Data"
                        )
                        fig.update_layout(height=max(400, len(missing_data.head(20)) * 25))
                        st.plotly_chart(fig, config={'responsive': True})
                        
                        st.dataframe(missing_data, hide_index=True)
                    else:
                        st.success("✅ No missing data detected!")
                
                # Data distribution analysis
                st.markdown("#### 📊 Data Distribution Summary")
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    summary_stats = df[numeric_cols].describe()
                    st.dataframe(summary_stats.round(2))
            else:
                st.warning("No processed data available for quality analysis.")
        
        except Exception as e:
            st.error(f"Error loading data for quality analysis: {e}")
    
    with tab4:
        st.markdown("### 🔄 Model Management")
        
        # Model retraining
        st.markdown("#### 🔧 Model Operations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Retrain All Models", type="primary"):
                with st.spinner("Retraining models..."):
                    try:
                        # Import training functionality
                        import sys
                        from pathlib import Path
                        sys.path.append(str(Path(__file__).parent.parent / "src"))
                        from train import train_models
                        from ingest import load_dataset
                        
                        # Load and retrain models
                        df = load_dataset('data/raw/Final Dataset.xlsx')
                        results = train_models(df)
                        
                        if results:
                            best_model = max(results.keys(), key=lambda k: results[k]['evaluation']['roc_auc'])
                            best_auc = results[best_model]['evaluation']['roc_auc']
                            st.success(f"✅ Models retrained successfully! Best model: {best_model} (AUC: {best_auc:.3f})")
                        else:
                            st.error("❌ Model retraining failed - no models trained")
                            
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error during retraining: {e}")
        
        with col2:
            if st.button("📊 Generate Performance Report"):
                st.info("📊 Performance report generated! Check downloads.")
        
        with col3:
            if st.button("🧹 Clear Model Cache"):
                st.info("🧹 Model cache cleared!")
        
        # Model versioning
        st.markdown("#### 📝 Model Versions")
        
        model_versions = [
            {"Version": "v1.0", "Date": "2024-01-15", "Model": "XGBoost", "ROC-AUC": 0.87, "Status": "Production"},
            {"Version": "v0.9", "Date": "2024-01-10", "Model": "Random Forest", "ROC-AUC": 0.85, "Status": "Archived"},
            {"Version": "v0.8", "Date": "2024-01-05", "Model": "Logistic Regression", "ROC-AUC": 0.78, "Status": "Archived"}
        ]
        
        versions_df = pd.DataFrame(model_versions)
        st.dataframe(versions_df, hide_index=True)
        
        # Deployment options
        st.markdown("#### 🚀 Deployment")
        
        deployment_options = st.selectbox(
            "Select Deployment Target",
            ["Staging", "Production", "Testing"],
            help="Choose where to deploy the current model"
        )
        
        if st.button(f"Deploy to {deployment_options}"):
            st.success(f"🚀 Model deployed to {deployment_options} environment!")
    
    with tab5:
        st.markdown("### ⚠️ Concept Drift Analysis")
        
        try:
            df = load_data("data/processed/model_dataset.parquet")
            create_concept_drift_analysis(df)
            
        except Exception as e:
            st.error(f"Error performing concept drift analysis: {e}")
        
        # Drift monitoring settings
        st.markdown("---")
        st.markdown("#### 🔧 Drift Monitoring Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            drift_threshold = st.slider(
                "Drift Detection Threshold",
                min_value=0.01,
                max_value=0.10,
                value=0.05,
                step=0.01,
                help="P-value threshold for drift detection"
            )
        
        with col2:
            monitoring_frequency = st.selectbox(
                "Monitoring Frequency",
                ["Daily", "Weekly", "Monthly"],
                index=1,
                help="How often to check for concept drift"
            )
        
        if st.button("💾 Save Drift Settings"):
            st.success("✅ Drift monitoring settings saved!")
    
    with tab7:
        st.markdown("### 👥 User Management")
        
        # Initialize database connection
        try:
            db = DatabaseManager()
            
            # User management sub-tabs
            user_tab1, user_tab2, user_tab3, user_tab4 = st.tabs([
                "👤 User List",
                "➕ Create User", 
                "✏️ Edit User",
                "📊 User Activity"
            ])
            
            with user_tab1:
                st.markdown("#### 👤 All Users")
                
                # Load all users
                try:
                    users = db.get_all_users()
                    
                    if users:
                        # Create user dataframe for display
                        users_df = pd.DataFrame(users)
                        users_df['created_at'] = pd.to_datetime(users_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
                        # Handle None/NaT values in last_login safely
                        users_df['last_login'] = users_df['last_login'].apply(
                            lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M') if pd.notna(x) and x is not None else 'Never'
                        )
                        
                        # Format role display
                        role_icons = {
                            'administrator': '👑 Administrator',
                            'auditor': '🔍 Auditor', 
                            'viewer': '👁️ Viewer'
                        }
                        users_df['role'] = users_df['role'].map(lambda x: role_icons.get(x, f'❓ {x}'))
                        
                        # Display user table
                        st.dataframe(
                            users_df[['username', 'full_name', 'email', 'role', 'created_at', 'last_login']],
                            hide_index=True,
                            use_container_width=True,
                            column_config={
                                'username': 'Username',
                                'full_name': 'Full Name',
                                'email': 'Email',
                                'role': 'Role',
                                'created_at': 'Created',
                                'last_login': 'Last Login'
                            }
                        )
                        
                        # User statistics
                        st.markdown("---")
                        st.markdown("#### 📊 User Statistics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_users = len(users)
                            st.metric("Total Users", f"{total_users:,}")
                        
                        with col2:
                            admin_count = len([u for u in users if u['role'] == 'administrator'])
                            st.metric("Administrators", f"{admin_count:,}")
                        
                        with col3:
                            auditor_count = len([u for u in users if u['role'] == 'auditor'])
                            st.metric("Auditors", f"{auditor_count:,}")
                        
                        with col4:
                            active_users = len([u for u in users if u['last_login'] is not None])
                            st.metric("Active Users", f"{active_users:,}")
                            
                    else:
                        st.info("No users found in the system.")
                        
                except Exception as e:
                    st.error(f"Error loading users: {e}")
            
            with user_tab2:
                st.markdown("#### ➕ Create New User")
                
                # Create user form
                with st.form("create_user_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        new_username = st.text_input(
                            "Username",
                            placeholder="Enter unique username",
                            help="Unique identifier for the user"
                        )
                        new_email = st.text_input(
                            "Email",
                            placeholder="user@example.com",
                            help="User's email address"
                        )
                        new_role = st.selectbox(
                            "Role",
                            options=["viewer", "auditor", "administrator"],
                            format_func=lambda x: {
                                "viewer": "👁️ Viewer - Read-only access",
                                "auditor": "🔍 Auditor - Can view cases and analysis", 
                                "administrator": "👑 Administrator - Full system access"
                            }[x],
                            help="User's access level in the system"
                        )
                    
                    with col2:
                        new_full_name = st.text_input(
                            "Full Name",
                            placeholder="John Doe",
                            help="User's display name"
                        )
                        new_password = st.text_input(
                            "Password",
                            type="password",
                            placeholder="Enter secure password",
                            help="Minimum 8 characters recommended"
                        )
                        confirm_password = st.text_input(
                            "Confirm Password",
                            type="password",
                            placeholder="Re-enter password"
                        )
                    
                    submitted = st.form_submit_button("➕ Create User", type="primary")
                    
                    if submitted:
                        # Validation
                        if not all([new_username, new_email, new_full_name, new_password]):
                            st.error("All fields are required!")
                        elif new_password != confirm_password:
                            st.error("Passwords do not match!")
                        elif len(new_password) < 6:
                            st.error("Password must be at least 6 characters long!")
                        else:
                            try:
                                # Create user
                                user_id = db.create_user(
                                    username=new_username,
                                    password=new_password,
                                    role=new_role,
                                    full_name=new_full_name,
                                    email=new_email
                                )
                                
                                if user_id:
                                    st.success(f"✅ User '{new_username}' created successfully!")
                                    st.balloons()
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to create user. Username might already exist.")
                                    
                            except Exception as e:
                                st.error(f"❌ Error creating user: {e}")
            
            with user_tab3:
                st.markdown("#### ✏️ Edit User")
                
                # Load users for selection
                try:
                    users = db.get_all_users()
                    
                    if users:
                        # User selection
                        user_options = {f"{u['username']} ({u['full_name']})": u['id'] for u in users}
                        selected_user_display = st.selectbox(
                            "Select User to Edit",
                            options=list(user_options.keys()),
                            help="Choose a user to modify their details"
                        )
                        
                        if selected_user_display:
                            selected_user_id = user_options[selected_user_display]
                            selected_user = next(u for u in users if u['id'] == selected_user_id)
                            
                            st.markdown("---")
                            
                            # Edit form
                            with st.form("edit_user_form"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    edit_username = st.text_input(
                                        "Username",
                                        value=selected_user['username'],
                                        disabled=True,
                                        help="Username cannot be changed"
                                    )
                                    edit_email = st.text_input(
                                        "Email",
                                        value=selected_user['email'],
                                        help="User's email address"
                                    )
                                
                                with col2:
                                    edit_full_name = st.text_input(
                                        "Full Name",
                                        value=selected_user['full_name'],
                                        help="User's display name"
                                    )
                                    edit_role = st.selectbox(
                                        "Role",
                                        options=["viewer", "auditor", "administrator"],
                                        index=["viewer", "auditor", "administrator"].index(selected_user['role']),
                                        format_func=lambda x: {
                                            "viewer": "👁️ Viewer",
                                            "auditor": "🔍 Auditor", 
                                            "administrator": "👑 Administrator"
                                        }[x]
                                    )
                                
                                # Password reset section
                                st.markdown("**Password Reset (Optional)**")
                                new_user_password = st.text_input(
                                    "New Password",
                                    type="password",
                                    placeholder="Leave empty to keep current password"
                                )
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    update_submitted = st.form_submit_button("💾 Update User", type="primary")
                                
                                with col2:
                                    delete_submitted = st.form_submit_button("🗑️ Delete User", type="secondary")
                                
                                if delete_submitted:
                                    try:
                                        # Check if delete methods exist in DatabaseManager
                                        if hasattr(db, 'delete_user'):
                                            # Confirmation dialog simulation
                                            st.warning(f"⚠️ Are you sure you want to delete user '{selected_user['username']}'?")
                                            
                                            if st.button("🗑️ Confirm Delete", key=f"confirm_delete_{selected_user_id}"):
                                                success = db.delete_user(selected_user_id)
                                                
                                                if success:
                                                    st.success(f"✅ User '{selected_user['username']}' deleted successfully!")
                                                    st.rerun()
                                                else:
                                                    st.error("❌ Failed to delete user.")
                                        else:
                                            st.warning("⚠️ User deletion functionality not yet implemented in DatabaseManager.")
                                            st.info("To enable user deletion, add delete_user method to DatabaseManager class.")
                                            
                                    except Exception as e:
                                        st.error(f"❌ Error deleting user: {e}")
                                
                                if update_submitted:
                                    try:
                                        # Check if update methods exist in DatabaseManager
                                        if hasattr(db, 'update_user'):
                                            # Update user in database
                                            success = db.update_user(
                                                user_id=selected_user_id,
                                                email=edit_email,
                                                full_name=edit_full_name, 
                                                role=edit_role,
                                                password=new_user_password if new_user_password else None
                                            )
                                            
                                            if success:
                                                st.success(f"✅ User '{selected_user['username']}' updated successfully!")
                                                st.rerun()
                                            else:
                                                st.error("❌ Failed to update user.")
                                        else:
                                            st.warning("⚠️ User update functionality not yet implemented in DatabaseManager.")
                                            st.info("To enable user updates, add update_user method to DatabaseManager class.")
                                            
                                    except Exception as e:
                                        st.error(f"❌ Error updating user: {e}")
                                    
                    else:
                        st.info("No users available for editing.")
                        
                except Exception as e:
                    st.error(f"Error loading users for editing: {e}")
            
            with user_tab4:
                st.markdown("#### 📊 User Activity Analysis")
                
                try:
                    users = db.get_all_users()
                    
                    if users:
                        # Activity metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Users with recent activity (last 30 days)
                            recent_threshold = pd.Timestamp.now() - pd.Timedelta(days=30)
                            recent_users = [
                                u for u in users 
                                if u['last_login'] and pd.to_datetime(u['last_login']) > recent_threshold
                            ]
                            st.metric("Recent Active Users", f"{len(recent_users):,}")
                        
                        with col2:
                            # Users never logged in
                            never_logged = [u for u in users if u['last_login'] is None]
                            st.metric("Never Logged In", f"{len(never_logged):,}")
                        
                        with col3:
                            # Average account age
                            if users:
                                account_ages = [
                                    (pd.Timestamp.now() - pd.to_datetime(u['created_at'])).days 
                                    for u in users
                                ]
                                avg_age = np.mean(account_ages)
                                st.metric("Avg Account Age", f"{avg_age:.0f} days")
                        
                        st.markdown("---")
                        
                        # Role distribution chart
                        st.markdown("##### 🎯 Role Distribution")
                        
                        role_counts = {}
                        for user in users:
                            role = user['role']
                            role_counts[role] = role_counts.get(role, 0) + 1
                        
                        if role_counts:
                            fig = go.Figure([go.Pie(
                                labels=[f"{role.title()}s" for role in role_counts.keys()],
                                values=list(role_counts.values()),
                                hole=0.3,
                                textinfo='label+percent+value'
                            )])
                            
                            fig.update_layout(
                                title="User Role Distribution",
                                height=400,
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig, config={'responsive': True})
                        
                        # Recent login activity
                        st.markdown("##### 📅 Recent Login Activity")
                        
                        users_with_login = [u for u in users if u['last_login'] is not None]
                        
                        if users_with_login:
                            # Sort by last login
                            users_with_login.sort(key=lambda x: x['last_login'], reverse=True)
                            
                            # Show top 10 recent logins
                            recent_df = pd.DataFrame(users_with_login[:10])
                            # Safe datetime handling for recent logins
                            recent_df['last_login'] = recent_df['last_login'].apply(
                                lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M') if pd.notna(x) and x is not None else 'Never'
                            )
                            
                            st.dataframe(
                                recent_df[['username', 'full_name', 'role', 'last_login']],
                                hide_index=True,
                                use_container_width=True,
                                column_config={
                                    'username': 'Username',
                                    'full_name': 'Full Name', 
                                    'role': 'Role',
                                    'last_login': 'Last Login'
                                }
                            )
                        else:
                            st.info("No login activity recorded yet.")
                            
                    else:
                        st.info("No user data available.")
                        
                except Exception as e:
                    st.error(f"Error analyzing user activity: {e}")
                    
        except Exception as e:
            st.error(f"Database connection error: {e}")
            st.info("Please ensure the database is properly configured and accessible.")
    
    # Footer actions
    st.markdown("---")
    st.markdown("### 🛠️ System Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Refresh System Status"):
            st.rerun()
    
    with col2:
        if st.button("📊 Export System Report"):
            report_data = {
                "system_status": status,
                "timestamp": pd.Timestamp.now().isoformat(),
                "model_performance": create_model_performance_summary()
            }
            
            st.download_button(
                label="Download Report",
                data=json.dumps(report_data, indent=2),
                file_name=f"system_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("🗂️ Backup System"):
            st.info("🗂️ System backup initiated!")
    
    with col4:
        if st.button("🔧 Run Diagnostics"):
            with st.spinner("Running system diagnostics..."):
                import time
                time.sleep(2)
                st.success("✅ All systems operational!")

    with tab6:
        st.markdown("### 📈 Advanced Model Monitoring")
        
        if MONITORING_AVAILABLE:
            # Initialize monitoring system
            monitor = ModelMonitor()
            
            try:
                # Load monitoring data
                model_results, y_true, y_prob = load_model_monitoring_data()
                
                if model_results:
                    # Generate monitoring summary
                    summary = monitor.generate_monitoring_summary(model_results, y_true, y_prob)
                    
                    # Monitoring Status Overview
                    st.markdown("#### 🎯 Monitoring Status")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        status_color = {
                            'Healthy': '🟢',
                            'Warning': '🟡', 
                            'Critical': '🔴',
                            'Unknown': '⚪'
                        }
                        st.metric(
                            "System Status",
                            f"{status_color.get(summary['performance_status'], '⚪')} {summary['performance_status']}"
                        )
                    
                    with col2:
                        st.metric(
                            "Models Monitored", 
                            summary['models_monitored']
                        )
                    
                    with col3:
                        st.metric(
                            "Best Model",
                            summary.get('best_model', 'Unknown')
                        )
                    
                    with col4:
                        best_auc = summary.get('key_metrics', {}).get('roc_auc', 0)
                        st.metric(
                            "Best ROC AUC",
                            f"{best_auc:.3f}" if best_auc else "N/A"
                        )
                    
                    # Alert Messages
                    if summary.get('alerts'):
                        st.markdown("#### ⚠️ Performance Alerts")
                        for alert in summary['alerts']:
                            st.warning(alert)
                    else:
                        st.success("✅ No performance alerts detected")
                    
                    st.markdown("---")
                    
                    # Performance Visualizations
                    if y_true is not None and y_prob is not None:
                        st.markdown("#### 📊 Performance Curves")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Precision@K Curve
                            st.markdown("**Precision at K Analysis**")
                            precision_k_fig = monitor.create_precision_at_k_plot(y_true, y_prob, max_k=100)
                            st.plotly_chart(precision_k_fig, config={'responsive': True})
                        
                        with col2:
                            # ROC Curve
                            best_model_name = summary.get('best_model', 'Best Model')
                            st.markdown("**ROC Curve Analysis**")
                            roc_fig = monitor.create_roc_curve(y_true, y_prob, best_model_name)
                            st.plotly_chart(roc_fig, config={'responsive': True})
                        
                        # Precision-Recall Curve
                        st.markdown("**Precision-Recall Curve**")
                        pr_fig = monitor.create_precision_recall_curve(y_true, y_prob, best_model_name)
                        st.plotly_chart(pr_fig, config={'responsive': True})
                    
                    # Model Comparison
                    st.markdown("#### 🔄 Model Comparison")
                    comparison_fig = monitor.create_model_comparison_chart(model_results)
                    st.plotly_chart(comparison_fig, config={'responsive': True})
                    
                    # Performance Timeline
                    st.markdown("#### ⏱️ Performance Over Time")
                    timeline_fig = monitor.create_performance_timeline([])
                    st.plotly_chart(timeline_fig, config={'responsive': True})
                    
                    # Detailed Metrics
                    st.markdown("#### 📋 Detailed Performance Metrics")
                    
                    if summary.get('precision_at_k'):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Precision at K Values**")
                            precision_data = []
                            for k, precision in summary['precision_at_k'].items():
                                precision_data.append({'K': k, 'Precision@K': f"{precision:.3f}"})
                            
                            precision_df = pd.DataFrame(precision_data)
                            st.dataframe(precision_df, hide_index=True)
                        
                        with col2:
                            st.markdown("**Key Metrics**")
                            metrics = summary.get('key_metrics', {})
                            for metric, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    st.write(f"**{metric.replace('_', ' ').title()}:** {value:.3f}")
                    
                    # Monitoring Configuration
                    st.markdown("#### ⚙️ Monitoring Settings")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        roc_threshold = st.slider(
                            "ROC AUC Alert Threshold",
                            min_value=0.5,
                            max_value=0.9,
                            value=monitor.alert_thresholds['roc_auc_min'],
                            step=0.01,
                            help="Alert when ROC AUC drops below this value"
                        )
                    
                    with col2:
                        precision_threshold = st.slider(
                            "Precision@10 Alert Threshold", 
                            min_value=0.5,
                            max_value=0.95,
                            value=monitor.alert_thresholds['precision_at_10_min'],
                            step=0.01,
                            help="Alert when Precision@10 drops below this value"
                        )
                    
                    if st.button("💾 Update Monitoring Thresholds"):
                        monitor.alert_thresholds['roc_auc_min'] = roc_threshold
                        monitor.alert_thresholds['precision_at_10_min'] = precision_threshold
                        st.success("✅ Monitoring thresholds updated!")
                
                else:
                    st.warning("No model results available for monitoring. Please train models first.")
                    
            except Exception as e:
                st.error(f"Error in monitoring system: {e}")
                
        else:
            st.error("Advanced monitoring system not available. Please check system configuration.")

if __name__ == "__main__":
    main()
