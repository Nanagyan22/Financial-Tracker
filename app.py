import warnings
warnings.filterwarnings("ignore")
# Suppress specific warnings that may show in Streamlit UI
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# Suppress Plotly-specific warnings
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Import authentication
try:
    from auth import init_authentication, create_navigation_menu, UserRole
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

# Configure page
st.set_page_config(
    page_title="Financial Irregularities Prediction System",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modules
try:
    from ingest import load_dataset, detect_largest_sheet
    from labelling import process_pdf_labels
    from preprocess import preprocess_data
    from features import engineer_features
    from train import train_models
    from evaluate import evaluate_models
    from explain import generate_explanations
    from utils import save_model, load_model, get_model_metrics
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.warning("Some advanced ML features may not be available due to missing system dependencies.")
    # We can still continue with basic functionality
    pass

def main():
    # Initialize database layer on app startup
    try:
        import sys
        sys.path.append('src')
        from database import init_database
        if 'database_initialized' not in st.session_state:
            if init_database():
                st.session_state.database_initialized = True
            else:
                st.warning("⚠️ Database initialization failed - using fallback storage")
    except Exception as e:
        st.warning(f"⚠️ Database setup error: {e}")
    
    # Initialize authentication for user context (optional login)
    if AUTH_AVAILABLE:
        auth_manager = init_authentication()
        
        # Show user info in sidebar if logged in
        if auth_manager.is_authenticated():
            auth_manager.show_user_info()
            # Create navigation menu based on user role
            create_navigation_menu(auth_manager)
        else:
            # Show login option for anonymous users
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🔐 Login")
            if st.sidebar.button("Login to System"):
                auth_manager.show_login_form()
        
    # Sidebar for navigation and file upload
    st.sidebar.title("🔧 System Controls")
    
    # File upload section
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload New Dataset", 
        type=["xlsx", "csv"],
        help="Upload an Excel or CSV file with financial data"
    )
    
    uploaded_pdf = st.sidebar.file_uploader(
        "Upload Audit Report (PDF)", 
        type=["pdf"],
        help="Upload an Auditor-General report PDF"
    )
    
    # Initialize session state for data
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    
    # Main content area
    st.title("🏛️ Financial Irregularities Prediction System")
    st.markdown("**University of Ghana Business School**")
    st.markdown("---")
    
    # Data loading section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Data Processing Pipeline")
        
        # Step 1: Data Ingestion
        if st.button("1. Load and Process Data", type="primary"):
            with st.spinner("Loading dataset..."):
                try:
                    if uploaded_file is not None:
                        df = load_dataset(uploaded_file)
                        st.success(f"✅ Uploaded dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                    else:
                        # Try to load default dataset
                        default_path = "data/raw/Final Dataset.xlsx"
                        if os.path.exists(default_path):
                            df = load_dataset(default_path)
                            st.success(f"✅ Default dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                        else:
                            st.error("No default dataset found. Please upload a dataset.")
                            return
                    
                    st.session_state.raw_data = df
                    st.session_state.data_loaded = True
                    
                    # Show data preview
                    st.subheader("Data Preview")
                    st.dataframe(df.head())
                    
                except Exception as e:
                    st.error(f"Error loading data: {e}")
        
        # Step 2: PDF Labelling
        if st.session_state.data_loaded:
            if st.button("2. Process Audit Labels"):
                with st.spinner("Processing audit report..."):
                    try:
                        if uploaded_pdf is not None:
                            df_labeled = process_pdf_labels(st.session_state.raw_data, uploaded_pdf)
                        else:
                            # Try default PDF
                            default_pdf = "data/raw/2023 (1).pdf"
                            if os.path.exists(default_pdf):
                                df_labeled = process_pdf_labels(st.session_state.raw_data, default_pdf)
                            else:
                                st.warning("No audit report found. Creating synthetic labels for demonstration.")
                                df_labeled = st.session_state.raw_data.copy()
                                # Create sample audit flags based on data patterns
                                np.random.seed(42)
                                df_labeled['audit_flag'] = np.random.binomial(1, 0.15, len(df_labeled))
                        
                        st.session_state.labeled_data = df_labeled
                        st.success(f"✅ Labels processed. {df_labeled['audit_flag'].sum()} entities flagged")
                        
                        # Show label distribution
                        flag_counts = df_labeled['audit_flag'].value_counts()
                        st.write(f"**Label Distribution:** Normal: {flag_counts.get(0, 0)}, Flagged: {flag_counts.get(1, 0)}")
                        
                    except Exception as e:
                        st.error(f"Error processing labels: {e}")
        
        # Step 3: Preprocessing and Feature Engineering
        if st.session_state.data_loaded and 'labeled_data' in st.session_state:
            if st.button("3. Preprocess & Engineer Features"):
                with st.spinner("Preprocessing data and engineering features..."):
                    try:
                        # Preprocess data
                        df_processed = preprocess_data(st.session_state.labeled_data)
                        
                        # Engineer features
                        df_features = engineer_features(df_processed)
                        
                        st.session_state.processed_data = df_features
                        st.success(f"✅ Features engineered: {df_features.shape[1]} total features")
                        
                        # Show feature info
                        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
                        st.write(f"**Numeric Features:** {len(numeric_cols)}")
                        
                    except Exception as e:
                        st.error(f"Error in preprocessing: {e}")
        
        # Step 4: Model Training
        if 'processed_data' in st.session_state:
            if st.button("4. Train Models"):
                with st.spinner("Training machine learning models..."):
                    try:
                        results = train_models(st.session_state.processed_data)
                        st.session_state.model_results = results
                        st.session_state.model_trained = True
                        st.success("✅ Models trained successfully")
                        
                        # Show best model
                        best_model_name = max(results.keys(), key=lambda k: results[k]['roc_auc'])
                        st.write(f"**Best Model:** {best_model_name} (ROC-AUC: {results[best_model_name]['roc_auc']:.3f})")
                        
                    except Exception as e:
                        st.error(f"Error training models: {e}")
        
        # Step 5: Generate Explanations
        if st.session_state.model_trained:
            if st.button("5. Generate SHAP Explanations"):
                with st.spinner("Generating model explanations..."):
                    try:
                        explanations = generate_explanations(
                            st.session_state.processed_data,
                            st.session_state.model_results
                        )
                        st.session_state.explanations = explanations
                        st.success("✅ SHAP explanations generated")
                        
                    except Exception as e:
                        st.error(f"Error generating explanations: {e}")
    
    with col2:
        st.header("System Status")
        
        # Status indicators
        status_items = [
            ("Data Loaded", st.session_state.data_loaded),
            ("Labels Processed", 'labeled_data' in st.session_state),
            ("Features Engineered", 'processed_data' in st.session_state),
            ("Models Trained", st.session_state.model_trained),
            ("Explanations Ready", 'explanations' in st.session_state)
        ]
        
        for item, status in status_items:
            if status:
                st.success(f"✅ {item}")
            else:
                st.info(f"⏳ {item}")
    
    # Quick stats section
    if 'processed_data' in st.session_state:
        st.header("Dataset Overview")
        df = st.session_state.processed_data
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Entities", len(df))
        with col2:
            if 'audit_flag' in df.columns:
                flagged = df['audit_flag'].sum()
                st.metric("Flagged Entities", flagged)
        with col3:
            if 'year' in df.columns:
                years = df['year'].nunique()
                st.metric("Years Covered", years)
        with col4:
            st.metric("Features", len(df.columns))
    
    # Navigation to other pages
    if st.session_state.model_trained:
        st.header("🚀 Ready to Explore")
        st.info("Your models are trained! Use the sidebar navigation to explore:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("📊 **Dashboard** - View analytics and trends")
        with col2:
            st.write("🏢 **Entity Profiles** - Analyze specific entities")
        with col3:
            st.write("🔍 **Case Viewer** - Review flagged cases")

if __name__ == "__main__":
    main()
