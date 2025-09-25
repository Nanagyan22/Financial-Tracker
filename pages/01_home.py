import streamlit as st
import base64
from pathlib import Path
import sys

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Configure page
st.set_page_config(
    page_title="Financial Irregularities Prediction - Home",
    page_icon="🏛️",
    layout="wide"
)

def get_image_base64(image_path):
    """Convert image to base64 string."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

def main():
    # Header with UGBS branding and images
    col1, col2, col3 = st.columns([1, 2, 1])
    
    # Get base64 encoded images
    ugbs_logo = get_image_base64("assets/ugbs_logo.png")
    author_photo = get_image_base64("assets/author_photo.jpg")
    
    with col1:
        if ugbs_logo:
            st.markdown(f"""
            <div style="text-align: center; margin-top: 20px;">
                <img src="data:image/png;base64,{ugbs_logo}" width="150">
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <h1 style="color: #1f4e79; margin-bottom: 0;">🏛️ University of Ghana Business School</h1>
            <h3 style="color: #666; margin-top: 0; margin-bottom: 30px;">Financial Irregularities Prediction System</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if author_photo:
            st.markdown(f"""
            <div style="text-align: center; margin-top: 20px;">
                <img src="data:image/jpeg;base64,{author_photo}" width="120" style="border-radius: 50%; border: 3px solid #1f4e79;">
            </div>
            """, unsafe_allow_html=True)
    
    # Main content
    st.markdown("---")
    
    # Project title and details
    st.markdown("""
    ## Predicting Financial Irregularities in Ghanaian State Entities using Machine Learning
    
    **Author:** Bernardine Akorfa Gawu (22253324)  
    **Supervisor:** Dr. Kwaku Ohene-Asare  
    **Institution:** University of Ghana Business School  
    **Program:** Master of Business Administration (MBA)  
    **Year:** 2025
    """)
    
    st.markdown("---")
    
    # Project Overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📋 Project Overview
        
        This research develops a **machine learning-powered risk classification system** to detect financial irregularities 
        in Ghanaian State Entities (SEs). The system integrates structured financial data with findings from 
        Auditor-General reports to produce risk scores, explanations, and comprehensive dashboards.
        
        #### 🎯 Objectives
        - **Automated Detection:** Identify entities at high risk of financial irregularities
        - **Early Warning System:** Provide proactive alerts for potential audit issues
        - **Explainable AI:** Offer clear explanations for risk assessments using SHAP
        - **Data-Driven Insights:** Support evidence-based decision making in public sector oversight
        
        #### 🔧 Key Features
        - **Multi-Model Approach:** Logistic Regression, Random Forest, XGBoost, and LightGBM
        - **Time-Aware Analysis:** Historical trend analysis and year-over-year changes
        - **PDF Integration:** Automated extraction of irregularities from Auditor-General reports
        - **Interactive Dashboards:** Real-time visualization of risk metrics and trends
        - **Entity Profiling:** Detailed risk assessment for individual state entities
        - **Explainable Predictions:** SHAP-based explanations for all risk scores
        """)
        
        st.markdown("""
        #### 📊 Data Sources
        - **Primary Dataset:** Financial data from Ghanaian State Entities (2020-2023)
        - **Audit Reports:** Ghana Audit Service reports with irregularity findings
        - **Features:** Revenue, expenditure, assets, liabilities, procurement data, and more
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Quick Start Guide
        
        **Step 1:** Navigate to the main application to load your data
        
        **Step 2:** Process audit labels from PDF reports
        
        **Step 3:** Train machine learning models
        
        **Step 4:** Explore results in the Dashboard
        
        **Step 5:** Analyze specific entities in Entity Profiles
        
        **Step 6:** Review flagged cases in Case Viewer
        """)
        
        st.info("""
        💡 **Tip:** Upload your own datasets using the file upload feature in the sidebar 
        to analyze different periods or entities.
        """)
        
        st.success("""
        ✅ **Ready to Start:** Click on 'Main Application' in the sidebar to begin your analysis.
        """)
    
    st.markdown("---")
    
    # Methodology
    st.markdown("""
    ### 🔬 Methodology
    
    Our approach combines traditional financial analysis with modern machine learning techniques:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📥 Data Integration**
        - Excel file processing with automatic sheet detection
        - PDF text extraction using pdfplumber
        - Fuzzy matching for entity name standardization
        - Financial ratio computation
        """)
    
    with col2:
        st.markdown("""
        **🤖 Machine Learning**
        - Multiple algorithm comparison
        - Time-aware train/test splitting
        - Hyperparameter optimization
        - Cross-validation for robustness
        """)
    
    with col3:
        st.markdown("""
        **📈 Explainability**
        - SHAP global feature importance
        - Local explanations for individual entities
        - Risk factor identification
        - Waterfall plots for prediction breakdown
        """)
    
    # Research Significance
    st.markdown("""
    ### 🌍 Research Significance
    
    This research addresses critical challenges in public sector financial oversight in Ghana and similar developing economies:
    
    - **Policy Impact:** Provides tools for more efficient allocation of audit resources
    - **Transparency:** Enhances public sector accountability through data-driven oversight
    - **Prevention:** Enables proactive identification of potential irregularities before they escalate
    - **Capacity Building:** Supports the Ghana Audit Service with modern analytical capabilities
    - **Academic Contribution:** Advances the application of ML in public sector financial management
    """)
    
    # Technical Specifications
    st.markdown("---")
    
    with st.expander("🔧 Technical Specifications"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Machine Learning Models:**
            - Logistic Regression (baseline)
            - Random Forest (ensemble)
            - XGBoost (gradient boosting)
            - LightGBM (efficient boosting)
            
            **Feature Engineering:**
            - Financial ratios (liquidity, profitability, efficiency)
            - Procurement indicators (single-source ratio, contract values)
            - Temporal features (year-over-year changes, volatility)
            - Risk indicators (composite scores, outlier flags)
            """)
        
        with col2:
            st.markdown("""
            **Evaluation Metrics:**
            - ROC-AUC (overall performance)
            - Precision-Recall AUC (imbalanced data)
            - Precision@K (top predictions)
            - F1-Score (balanced metric)
            
            **Technology Stack:**
            - Python 3.8+
            - Streamlit (web interface)
            - Scikit-learn, XGBoost, LightGBM (ML)
            - SHAP (explainability)
            - Plotly (visualizations)
            """)
    
    # Usage Instructions
    with st.expander("📖 Detailed Usage Instructions"):
        st.markdown("""
        #### Getting Started
        
        1. **Data Upload:** Use the sidebar to upload Excel files (.xlsx, .csv) with financial data
        2. **PDF Processing:** Upload Auditor-General PDF reports for automated label generation
        3. **Pipeline Execution:** Run the 5-step processing pipeline in the main application
        4. **Model Training:** Train multiple ML models and compare their performance
        5. **Exploration:** Use the different pages to explore results and insights
        
        #### Navigation Guide
        
        - **🏠 Home:** Project overview and instructions (this page)
        - **📊 Dashboard:** High-level KPIs, trends, and summary visualizations
        - **🏢 Entity Profile:** Detailed analysis of individual state entities
        - **🔍 Case Viewer:** Review and filter flagged cases with audit findings
        - **⚙️ Admin:** Model management, performance metrics, and system monitoring
        
        #### Data Format Requirements
        
        **Excel/CSV Files should contain:**
        - Entity names (organization, institution, company, etc.)
        - Financial data (revenue, expenditure, assets, liabilities)
        - Year information
        - Optional: procurement data, employee counts, regional information
        
        **PDF Reports should be:**
        - Auditor-General reports from Ghana Audit Service
        - Text-searchable (not scanned images)
        - Containing entity names and irregularity descriptions
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 50px;">
        <p><strong>University of Ghana Business School</strong> | <strong>MBA Program</strong> | <strong>2025</strong></p>
        <p>Advancing Public Sector Financial Oversight through Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
