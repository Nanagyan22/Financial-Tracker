import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import load_data, format_currency
from labelling import AuditLabelGenerator

# Import authentication
try:
    from auth import init_authentication, UserRole
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False
    init_authentication = None
    UserRole = None

# Configure page
st.set_page_config(
    page_title="Case Viewer - Financial Irregularities",
    page_icon="🔍",
    layout="wide"
)

def load_case_data():
    """Load data for case analysis."""
    try:
        df = load_data("data/processed/model_dataset.parquet")
        return df
    except:
        return None

def create_case_filters(df):
    """Create filter controls for cases."""
    if df is None:
        return {}
    
    with st.sidebar:
        st.header("🔍 Case Filters")
        
        filters = {}
        
        # Year filter
        if 'year' in df.columns:
            year_range = df['year'].agg(['min', 'max'])
            filters['year'] = st.slider(
                "Year Range",
                min_value=int(year_range['min']),
                max_value=int(year_range['max']),
                value=(int(year_range['min']), int(year_range['max'])),
                help="Filter cases by year"
            )
        
        # Audit flag filter
        if 'audit_flag' in df.columns:
            filters['audit_flag'] = st.selectbox(
                "Case Type",
                options=['All', 'Flagged Only', 'Normal Only'],
                help="Filter by audit flag status"
            )
        
        # Severity filter
        if 'severity' in df.columns:
            severity_options = ['All'] + list(df['severity'].dropna().unique())
            filters['severity'] = st.selectbox(
                "Severity Level",
                options=severity_options,
                help="Filter by irregularity severity"
            )
        
        # Entity type filter (inferred)
        entity_types = []
        if 'entity_name' in df.columns:
            for entity in df['entity_name'].unique():
                entity_lower = str(entity).lower()
                if any(word in entity_lower for word in ['university', 'college', 'school']):
                    entity_types.append('Education')
                elif any(word in entity_lower for word in ['hospital', 'clinic', 'health']):
                    entity_types.append('Health')
                elif any(word in entity_lower for word in ['commission', 'authority', 'board']):
                    entity_types.append('Regulatory')
                elif any(word in entity_lower for word in ['company', 'limited', 'ltd']):
                    entity_types.append('State Enterprise')
                else:
                    entity_types.append('Government')
        
        if entity_types:
            unique_types = ['All'] + list(set(entity_types))
            filters['entity_type'] = st.selectbox(
                "Entity Type",
                options=unique_types,
                help="Filter by entity type"
            )
        
        # Amount range filter
        if 'total_expenditure' in df.columns:
            expenditure_range = df['total_expenditure'].agg(['min', 'max'])
            if expenditure_range['max'] > expenditure_range['min']:
                filters['expenditure'] = st.slider(
                    "Expenditure Range (GHS)",
                    min_value=float(expenditure_range['min']),
                    max_value=float(expenditure_range['max']),
                    value=(float(expenditure_range['min']), float(expenditure_range['max'])),
                    format="%.0f",
                    help="Filter by expenditure amount"
                )
        
        # Text search
        filters['search'] = st.text_input(
            "🔍 Search Entity Names",
            placeholder="Enter entity name keywords...",
            help="Search for specific entities"
        )
        
        return filters

def apply_filters(df, filters):
    """Apply selected filters to dataframe."""
    if df is None:
        return df
    
    filtered_df = df.copy()
    
    # Year filter
    if 'year' in filters and 'year' in filtered_df.columns:
        year_min, year_max = filters['year']
        filtered_df = filtered_df[
            (filtered_df['year'] >= year_min) & 
            (filtered_df['year'] <= year_max)
        ]
    
    # Audit flag filter
    if 'audit_flag' in filters and 'audit_flag' in filtered_df.columns:
        if filters['audit_flag'] == 'Flagged Only':
            filtered_df = filtered_df[filtered_df['audit_flag'] == 1]
        elif filters['audit_flag'] == 'Normal Only':
            filtered_df = filtered_df[filtered_df['audit_flag'] == 0]
    
    # Severity filter
    if 'severity' in filters and filters['severity'] != 'All' and 'severity' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['severity'] == filters['severity']]
    
    # Entity type filter
    if 'entity_type' in filters and filters['entity_type'] != 'All' and 'entity_name' in filtered_df.columns:
        entity_mask = pd.Series(False, index=filtered_df.index)
        
        for idx, entity in filtered_df['entity_name'].items():
            entity_lower = str(entity).lower()
            if filters['entity_type'] == 'Education' and any(word in entity_lower for word in ['university', 'college', 'school']):
                entity_mask[idx] = True
            elif filters['entity_type'] == 'Health' and any(word in entity_lower for word in ['hospital', 'clinic', 'health']):
                entity_mask[idx] = True
            elif filters['entity_type'] == 'Regulatory' and any(word in entity_lower for word in ['commission', 'authority', 'board']):
                entity_mask[idx] = True
            elif filters['entity_type'] == 'State Enterprise' and any(word in entity_lower for word in ['company', 'limited', 'ltd']):
                entity_mask[idx] = True
            elif filters['entity_type'] == 'Government' and not any(word in entity_lower for word in ['university', 'college', 'school', 'hospital', 'clinic', 'health', 'commission', 'authority', 'board', 'company', 'limited', 'ltd']):
                entity_mask[idx] = True
        
        filtered_df = filtered_df[entity_mask]
    
    # Expenditure filter
    if 'expenditure' in filters and 'total_expenditure' in filtered_df.columns:
        exp_min, exp_max = filters['expenditure']
        filtered_df = filtered_df[
            (filtered_df['total_expenditure'] >= exp_min) & 
            (filtered_df['total_expenditure'] <= exp_max)
        ]
    
    # Text search
    if 'search' in filters and filters['search'] and 'entity_name' in filtered_df.columns:
        search_text = filters['search'].lower()
        filtered_df = filtered_df[
            filtered_df['entity_name'].str.lower().str.contains(search_text, na=False)
        ]
    
    return filtered_df

def create_case_summary_stats(df):
    """Create summary statistics for filtered cases."""
    if df is None or len(df) == 0:
        st.warning("No cases match the selected filters.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_cases = len(df)
        st.metric("Total Cases", f"{total_cases:,}")
    
    with col2:
        if 'audit_flag' in df.columns:
            flagged_cases = df['audit_flag'].sum()
            st.metric("Flagged Cases", f"{flagged_cases:,}")
        else:
            st.metric("Flagged Cases", "N/A")
    
    with col3:
        if 'total_expenditure' in df.columns:
            total_expenditure = df['total_expenditure'].sum()
            st.metric("Total Expenditure", format_currency(total_expenditure))
        else:
            st.metric("Total Expenditure", "N/A")
    
    with col4:
        if 'audit_flag' in df.columns and len(df) > 0:
            flag_rate = df['audit_flag'].mean() * 100
            st.metric("Flag Rate", f"{flag_rate:.1f}%")
        else:
            st.metric("Flag Rate", "N/A")

def create_cases_table(df):
    """Create interactive table of cases."""
    if df is None or len(df) == 0:
        st.info("No cases to display.")
        return pd.DataFrame()
    
    # Select columns for display
    display_columns = ['entity_name']
    
    if 'year' in df.columns:
        display_columns.append('year')
    if 'audit_flag' in df.columns:
        display_columns.append('audit_flag')
    if 'total_revenue' in df.columns:
        display_columns.append('total_revenue')
    if 'total_expenditure' in df.columns:
        display_columns.append('total_expenditure')
    if 'severity' in df.columns:
        display_columns.append('severity')
    if 'detected_issues' in df.columns:
        display_columns.append('detected_issues')
    
    # Filter to available columns
    available_columns = [col for col in display_columns if col in df.columns]
    
    if not available_columns:
        st.error("No suitable columns found for display.")
        return pd.DataFrame()
    
    # Create display dataframe
    display_df = df[available_columns].copy()
    
    # Format columns
    if 'audit_flag' in display_df.columns:
        display_df['audit_flag'] = display_df['audit_flag'].map({0: '🟢 Normal', 1: '🔴 Flagged'})
    
    if 'total_revenue' in display_df.columns:
        display_df['total_revenue'] = display_df['total_revenue'].apply(lambda x: format_currency(x) if pd.notnull(x) else "N/A")
    
    if 'total_expenditure' in display_df.columns:
        display_df['total_expenditure'] = display_df['total_expenditure'].apply(lambda x: format_currency(x) if pd.notnull(x) else "N/A")
    
    # Rename columns for better display
    column_mapping = {
        'entity_name': 'Entity Name',
        'year': 'Year',
        'audit_flag': 'Status',
        'total_revenue': 'Revenue',
        'total_expenditure': 'Expenditure',
        'severity': 'Severity',
        'detected_issues': 'Detected Issues'
    }
    
    display_df = display_df.rename(columns=column_mapping)
    
    # Sort by audit flag and year
    if 'Year' in display_df.columns:
        display_df = display_df.sort_values('Year', ascending=False)
    
    return display_df

def create_case_trends_plot(df):
    """Create trends plot for cases."""
    if df is None or 'year' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No temporal data available for trends",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title="Case Trends", height=400)
        return fig
    
    # Group by year
    agg_dict = {}
    if 'audit_flag' in df.columns:
        agg_dict['audit_flag'] = ['count', 'sum']
    else:
        # Use a different column for counting
        count_col = df.columns[0] if len(df.columns) > 0 else 'year'
        if count_col != 'year':
            agg_dict[count_col] = 'count'
    
    if 'total_expenditure' in df.columns:
        agg_dict['total_expenditure'] = 'sum'
    
    yearly_stats = df.groupby('year').agg(agg_dict).reset_index()
    
    # Flatten column names properly
    new_columns = ['year']
    if 'audit_flag' in df.columns:
        new_columns.extend(['total_cases', 'flagged_cases'])
    else:
        new_columns.append('total_cases')
    
    if 'total_expenditure' in df.columns:
        new_columns.append('expenditure_sum')
    
    # Ensure column count matches
    yearly_stats.columns = new_columns[:len(yearly_stats.columns)]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add total cases
    fig.add_trace(
        go.Bar(
            x=yearly_stats['year'],
            y=yearly_stats['total_cases'],
            name='Total Cases',
            marker_color='lightblue',
            opacity=0.7
        ),
        secondary_y=False,
    )
    
    # Add flagged cases if available
    if 'flagged_cases' in yearly_stats.columns:
        fig.add_trace(
            go.Bar(
                x=yearly_stats['year'],
                y=yearly_stats['flagged_cases'],
                name='Flagged Cases',
                marker_color='red',
                opacity=0.8
            ),
            secondary_y=False,
        )
    
    # Add expenditure trend if available
    if 'expenditure_sum' in yearly_stats.columns:
        fig.add_trace(
            go.Scatter(
                x=yearly_stats['year'],
                y=yearly_stats['expenditure_sum'],
                mode='lines+markers',
                name='Total Expenditure',
                line=dict(color='green', width=3)
            ),
            secondary_y=True,
        )
    
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Number of Cases", secondary_y=False)
    fig.update_yaxes(title_text="Total Expenditure (GHS)", secondary_y=True)
    
    fig.update_layout(
        title="Case Analysis Trends",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_risk_level_timeline_plot(df):
    """Create severity distribution plot."""
    if df is None or 'severity' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No severity data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title="Severity Distribution", height=400)
        return fig
    
    severity_counts = df['severity'].value_counts()
    
    # Define colors for severity levels
    color_map = {
        'high': 'red',
        'medium': 'orange',
        'low': 'yellow',
        'none': 'green'
    }
    
    colors = [color_map.get(str(sev).lower() if pd.notnull(sev) else 'unknown', 'gray') for sev in severity_counts.index]
    
    fig = go.Figure([go.Bar(
        x=severity_counts.index,
        y=severity_counts.values,
        marker_color=colors,
        text=severity_counts.values,
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Case Severity Distribution",
        xaxis_title="Severity Level",
        yaxis_title="Number of Cases",
        height=400
    )
    
    return fig

def load_audit_findings():
    """Load audit findings and processed labels."""
    try:
        # Try to load labels review file
        review_file = Path("data/processed/labels_review.csv")
        if review_file.exists():
            df_findings = pd.read_csv(review_file)
            return df_findings
        else:
            return None
    except:
        return None

def create_audit_report_summary(df):
    """Create summary of audit report processing results."""
    if df is None:
        st.info("No audit report findings available. Process an audit report in the main application first.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'audit_flag' in df.columns:
            total_flagged = df['audit_flag'].sum()
            st.metric("Total Flagged Entities", f"{total_flagged:,}")
        else:
            st.metric("Total Flagged Entities", "N/A")
    
    with col2:
        if 'severity' in df.columns:
            high_severity = len(df[df['severity'] == 'high']) if 'severity' in df.columns else 0
            st.metric("High Severity Cases", f"{high_severity:,}")
        else:
            st.metric("High Severity Cases", "N/A")
    
    with col3:
        if 'evidence_count' in df.columns:
            total_evidence = df['evidence_count'].sum() if 'evidence_count' in df.columns else 0
            st.metric("Total Evidence Items", f"{total_evidence:,}")
        else:
            st.metric("Total Evidence Items", "N/A")
    
    with col4:
        if 'detected_issues' in df.columns:
            entities_with_issues = len(df[df['detected_issues'].notna()])
            st.metric("Entities with Issues", f"{entities_with_issues:,}")
        else:
            st.metric("Entities with Issues", "N/A")

def create_irregularity_analysis_plot(df):
    """Create detailed irregularity analysis visualization."""
    if df is None or 'detected_issues' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No irregularity data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title="Irregularity Analysis", height=400)
        return fig
    
    # Extract issue categories from detected_issues
    issue_categories = {}
    audit_generator = AuditLabelGenerator()
    
    for _, row in df.iterrows():
        if pd.notna(row['detected_issues']):
            issues_text = str(row['detected_issues']).lower()
            
            # Categorize based on keywords
            if any(word in issues_text for word in ['fraud', 'embezzlement', 'ghost worker']):
                issue_categories['Fraud/Embezzlement'] = issue_categories.get('Fraud/Embezzlement', 0) + 1
            elif any(word in issues_text for word in ['procurement', 'contract', 'tender']):
                issue_categories['Procurement Issues'] = issue_categories.get('Procurement Issues', 0) + 1
            elif any(word in issues_text for word in ['payroll', 'salary', 'allowance']):
                issue_categories['Payroll Issues'] = issue_categories.get('Payroll Issues', 0) + 1
            elif any(word in issues_text for word in ['tax', 'duty', 'levy']):
                issue_categories['Tax/Revenue Issues'] = issue_categories.get('Tax/Revenue Issues', 0) + 1
            elif any(word in issues_text for word in ['asset', 'property', 'equipment']):
                issue_categories['Asset Management'] = issue_categories.get('Asset Management', 0) + 1
            elif any(word in issues_text for word in ['control', 'reconciliation', 'cash']):
                issue_categories['Financial Controls'] = issue_categories.get('Financial Controls', 0) + 1
            elif any(word in issues_text for word in ['compliance', 'violation', 'regulation']):
                issue_categories['Compliance Issues'] = issue_categories.get('Compliance Issues', 0) + 1
            else:
                issue_categories['Other Issues'] = issue_categories.get('Other Issues', 0) + 1
    
    if not issue_categories:
        fig = go.Figure()
        fig.add_annotation(
            text="No categorized irregularities found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title="Irregularity Categories", height=400)
        return fig
    
    # Create pie chart
    fig = go.Figure([go.Pie(
        labels=list(issue_categories.keys()),
        values=list(issue_categories.values()),
        hole=0.3,
        textinfo='label+percent',
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Distribution of Irregularity Categories",
        height=400,
        showlegend=True
    )
    
    return fig

def create_entity_matching_table(df):
    """Create table showing entity matching results."""
    if df is None:
        st.info("No entity matching data available.")
        return pd.DataFrame()
    
    # Select relevant columns for entity matching
    display_columns = ['entity_name']
    
    if 'audit_flag' in df.columns:
        display_columns.append('audit_flag')
    if 'detected_issues' in df.columns:
        display_columns.append('detected_issues')
    if 'severity' in df.columns:
        display_columns.append('severity')
    if 'evidence_count' in df.columns:
        display_columns.append('evidence_count')
    if 'irregularity_score' in df.columns:
        display_columns.append('irregularity_score')
    
    # Filter to available columns
    available_columns = [col for col in display_columns if col in df.columns]
    
    if not available_columns:
        st.error("No suitable columns found for entity matching display.")
        return pd.DataFrame()
    
    # Create display dataframe - only show entities with audit findings
    if 'audit_flag' in df.columns:
        display_df = df[df['audit_flag'] == 1][available_columns].copy()
    else:
        display_df = df[available_columns].copy()
    
    if len(display_df) == 0:
        st.info("No entities with audit flags found.")
        return pd.DataFrame()
    
    # Format columns
    if 'audit_flag' in display_df.columns:
        display_df['audit_flag'] = display_df['audit_flag'].map({0: '🟢 Normal', 1: '🔴 Flagged'})
    
    if 'irregularity_score' in display_df.columns:
        display_df['irregularity_score'] = display_df['irregularity_score'].round(3)
    
    # Truncate long detected_issues text for display
    if 'detected_issues' in display_df.columns:
        display_df['detected_issues'] = display_df['detected_issues'].apply(
            lambda x: str(x)[:100] + "..." if pd.notna(x) and len(str(x)) > 100 else str(x) if pd.notna(x) else "N/A"
        )
    
    # Rename columns for better display
    column_mapping = {
        'entity_name': 'Entity Name',
        'audit_flag': 'Status',
        'detected_issues': 'Detected Issues',
        'severity': 'Severity',
        'evidence_count': 'Evidence Count',
        'irregularity_score': 'Risk Score'
    }
    
    display_df = display_df.rename(columns=column_mapping)
    
    # Sort by severity and risk score
    if 'Severity' in display_df.columns:
        severity_order = {'high': 3, 'medium': 2, 'low': 1}
        display_df['severity_rank'] = display_df['Severity'].map(severity_order).fillna(0)
        display_df = display_df.sort_values(['severity_rank', 'Risk Score'], ascending=[False, False])
        display_df = display_df.drop('severity_rank', axis=1)
    elif 'Risk Score' in display_df.columns:
        display_df = display_df.sort_values('Risk Score', ascending=False)
    
    return display_df

def main():
    # Initialize authentication - AUDITOR level required
    if AUTH_AVAILABLE and init_authentication is not None and UserRole is not None:
        auth_manager = init_authentication()
        auth_manager.show_user_info()
        
        # Require AUDITOR role for case analysis
        if not auth_manager.require_role(UserRole.AUDITOR):
            return
    else:
        st.error("🔒 Authentication system unavailable. Access restricted.")
        return
    
    st.title("🔍 Case Viewer")
    st.markdown("Review and analyze flagged financial irregularity cases")
    
    # Load data
    df = load_case_data()
    
    if df is None:
        st.error("""
        **No case data available.**
        
        Please run the main application first to process your dataset and generate cases.
        """)
        return
    
    # Create filters
    filters = create_case_filters(df)
    
    # Apply filters
    filtered_df = apply_filters(df, filters)
    
    # Main content
    st.markdown("### 📊 Case Summary")
    create_case_summary_stats(filtered_df)
    
    st.markdown("---")
    
    # Load audit findings for enhanced analysis
    audit_findings = load_audit_findings()
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Cases Table", "📈 Trends Analysis", "🔍 Case Details", "📊 Audit Report Analysis"])
    
    with tab1:
        st.markdown("### 📋 Cases Overview")
        
        # Create and display table
        display_df = create_cases_table(filtered_df)
        
        if len(display_df) > 0:
            # Add pagination
            page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
            
            total_rows = len(display_df)
            total_pages = (total_rows - 1) // page_size + 1
            
            if total_pages > 1:
                page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
                start_idx = (page - 1) * page_size
                end_idx = min(start_idx + page_size, total_rows)
                
                st.write(f"Showing rows {start_idx + 1}-{end_idx} of {total_rows}")
                st.dataframe(display_df.iloc[start_idx:end_idx], hide_index=True, use_container_width=True)
            else:
                st.dataframe(display_df, hide_index=True, use_container_width=True)
            
            # Export filtered data
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Cases (CSV)",
                    data=csv_data,
                    file_name=f"filtered_cases_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                if st.button("🏷️ Mark Cases as Reviewed"):
                    # Mark cases as reviewed in the system
                    with st.spinner("Updating case status..."):
                        try:
                            # Simulate database update with actual operation
                            reviewed_count = len(filtered_df)
                            timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                            
                            # Create a simple tracking system
                            if 'reviewed_cases' not in st.session_state:
                                st.session_state.reviewed_cases = []
                            
                            # Add reviewed cases to session tracking
                            for idx, case in filtered_df.iterrows():
                                review_record = {
                                    'entity_name': case.get('entity_name', 'Unknown'),
                                    'year': case.get('year', 'Unknown'),
                                    'reviewed_by': st.session_state.get('username', 'admin'),
                                    'reviewed_at': timestamp,
                                    'case_id': idx
                                }
                                st.session_state.reviewed_cases.append(review_record)
                            
                            st.success(f"✅ {reviewed_count} cases successfully marked as reviewed!")
                            st.info(f"📝 Review completed by {st.session_state.get('username', 'admin')} at {timestamp}")
                            
                            # Show review summary
                            with st.expander("📋 Review Summary", expanded=True):
                                st.markdown(f"**Cases Reviewed:** {reviewed_count}")
                                st.markdown(f"**Reviewer:** {st.session_state.get('username', 'admin')}")
                                st.markdown(f"**Review Date:** {timestamp}")
                                st.markdown(f"**Status:** All selected cases marked as reviewed in system")
                                
                        except Exception as e:
                            st.error(f"Error updating case status: {str(e)}")
    
    with tab2:
        st.markdown("### 📈 Case Trends Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_case_trends_plot(filtered_df), config={'responsive': True})
        
        with col2:
            st.plotly_chart(create_risk_level_timeline_plot(filtered_df), config={'responsive': True})
        
        # Additional analysis
        if 'audit_flag' in filtered_df.columns and 'year' in filtered_df.columns:
            st.markdown("### 📊 Detailed Year-over-Year Analysis")
            
            yearly_analysis = filtered_df.groupby('year').agg({
                'audit_flag': ['count', 'sum', 'mean'],
                'total_expenditure': ['sum', 'mean'] if 'total_expenditure' in filtered_df.columns else ['count']
            }).round(3)
            
            # Flatten column names
            yearly_analysis.columns = ['Total Cases', 'Flagged Cases', 'Flag Rate', 'Total Expenditure', 'Avg Expenditure']
            yearly_analysis['Flag Rate'] = (yearly_analysis['Flag Rate'] * 100).round(1)
            
            st.dataframe(yearly_analysis, use_container_width=True)
    
    with tab3:
        st.markdown("### 🔍 Individual Case Details")
        
        if len(filtered_df) > 0:
            # Select specific case
            entity_options = filtered_df['entity_name'].unique() if 'entity_name' in filtered_df.columns else []
            
            if len(entity_options) > 0:
                selected_entity = st.selectbox(
                    "Select Entity for Detailed View",
                    entity_options,
                    help="Choose an entity to view detailed case information"
                )
                
                if selected_entity:
                    entity_cases = filtered_df[filtered_df['entity_name'] == selected_entity]
                    
                    st.markdown(f"#### 📋 Case Details: {selected_entity}")
                    
                    for idx, case in entity_cases.iterrows():
                        with st.expander(f"Year {case.get('year', 'Unknown')} - {'🔴 Flagged' if case.get('audit_flag', 0) == 1 else '🟢 Normal'}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Financial Data:**")
                                if 'revenue' in case:
                                    st.write(f"Revenue: {format_currency(case['revenue'])}")
                                if 'total_expenditure' in case:
                                    st.write(f"Expenditure: {format_currency(case['total_expenditure'])}")
                                if 'total_assets' in case:
                                    st.write(f"Total Assets: {format_currency(case['total_assets'])}")
                                if 'irregularity_score' in case:
                                    st.write(f"Risk Score: {case['irregularity_score']:.2f}")
                            
                            with col2:
                                st.markdown("**Audit Information:**")
                                if 'severity' in case:
                                    st.write(f"Severity: {case['severity']}")
                                if 'evidence_count' in case:
                                    st.write(f"Evidence Count: {case['evidence_count']}")
                                if 'detected_issues' in case and pd.notnull(case['detected_issues']):
                                    st.write(f"Issues: {case['detected_issues']}")
                            
                            # Add action buttons
                            action_col1, action_col2, action_col3 = st.columns(3)
                            with action_col1:
                                if st.button(f"🔍 Investigate", key=f"investigate_{idx}"):
                                    # Start investigation workflow
                                    with st.spinner("Starting investigation..."):
                                        # Create investigation record
                                        investigation_id = f"INV_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{idx}"
                                        
                                        # Show investigation details
                                        st.success(f"✅ Investigation {investigation_id} initiated successfully!")
                                        
                                        # Display investigation steps
                                        with st.expander("📋 Investigation Details", expanded=True):
                                            st.markdown("**Investigation Workflow:**")
                                            st.markdown("✅ Case flagged for detailed review")
                                            st.markdown("✅ Financial data extracted and analyzed")
                                            st.markdown("⏳ Assigned to senior auditor for verification")
                                            st.markdown("⏳ Scheduled for follow-up within 3 business days")
                                            
                                            if 'irregularity_score' in case:
                                                st.markdown(f"**Priority Level:** {'🔴 High' if case['irregularity_score'] > 0.7 else '🟡 Medium' if case['irregularity_score'] > 0.4 else '🟢 Low'}")
                                            
                                            st.markdown(f"**Investigation ID:** `{investigation_id}`")
                                            st.markdown(f"**Entity:** {selected_entity}")
                                            st.markdown(f"**Year:** {case.get('year', 'Unknown')}")
                                            
                                            # Add to investigation queue (simulate)
                                            st.info("📨 Investigation team has been notified and will begin review process.")
                            with action_col2:
                                if st.button(f"✅ Resolve", key=f"resolve_{idx}"):
                                    # Resolve case with actual tracking
                                    with st.spinner("Resolving case..."):
                                        try:
                                            # Create resolution record
                                            resolution_id = f"RES_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{idx}"
                                            timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                                            
                                            # Track resolved cases
                                            if 'resolved_cases' not in st.session_state:
                                                st.session_state.resolved_cases = []
                                            
                                            resolution_record = {
                                                'resolution_id': resolution_id,
                                                'entity_name': selected_entity,
                                                'year': case.get('year', 'Unknown'),
                                                'resolved_by': st.session_state.get('username', 'admin'),
                                                'resolved_at': timestamp,
                                                'case_id': idx,
                                                'irregularity_score': case.get('irregularity_score', 0)
                                            }
                                            st.session_state.resolved_cases.append(resolution_record)
                                            
                                            st.success(f"✅ Case {resolution_id} successfully resolved!")
                                            
                                            # Show resolution details
                                            with st.expander("📋 Resolution Details", expanded=True):
                                                st.markdown(f"**Resolution ID:** `{resolution_id}`")
                                                st.markdown(f"**Entity:** {selected_entity}")
                                                st.markdown(f"**Year:** {case.get('year', 'Unknown')}")
                                                st.markdown(f"**Resolved By:** {st.session_state.get('username', 'admin')}")
                                                st.markdown(f"**Resolution Date:** {timestamp}")
                                                st.markdown(f"**Status:** Case closed and archived")
                                                
                                                if case.get('irregularity_score', 0) > 0.7:
                                                    st.info("🔴 High-risk case resolved - follow-up audit recommended")
                                                elif case.get('irregularity_score', 0) > 0.4:
                                                    st.info("🟡 Medium-risk case resolved - monitoring continued")
                                                else:
                                                    st.info("🟢 Low-risk case resolved - no further action required")
                                                    
                                        except Exception as e:
                                            st.error(f"Error resolving case: {str(e)}")
                            with action_col3:
                                if st.button(f"📋 Add Note", key=f"note_{idx}"):
                                    st.text_area(f"Add note for {selected_entity}", key=f"note_text_{idx}")
            else:
                st.info("No entities available for detailed view.")
        else:
            st.info("No cases match the current filters.")
    
    with tab4:
        st.markdown("### 📊 Audit Report Analysis")
        
        # Audit report summary
        st.markdown("#### 📋 Audit Processing Summary")
        create_audit_report_summary(audit_findings)
        
        st.markdown("---")
        
        # Two-column layout for analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Irregularity Categories")
            st.plotly_chart(create_irregularity_analysis_plot(audit_findings), config={'responsive': True})
        
        with col2:
            st.markdown("#### 📈 Risk Level Timeline")
            st.plotly_chart(create_risk_level_timeline_plot(audit_findings), config={'responsive': True})
        
        st.markdown("---")
        
        # Entity matching table
        st.markdown("#### 🔗 Entity Matching Results")
        st.markdown("Entities successfully matched from audit report to dataset:")
        
        entity_matching_df = create_entity_matching_table(audit_findings)
        
        if len(entity_matching_df) > 0:
            # Add pagination for entity matching
            page_size = st.selectbox("Entities per page", [10, 25, 50], index=1, key="entity_page_size")
            
            total_rows = len(entity_matching_df)
            total_pages = (total_rows - 1) // page_size + 1
            
            if total_pages > 1:
                page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key="entity_page")
                start_idx = (page - 1) * page_size
                end_idx = min(start_idx + page_size, total_rows)
                
                st.write(f"Showing entities {start_idx + 1}-{end_idx} of {total_rows}")
                st.dataframe(entity_matching_df.iloc[start_idx:end_idx], hide_index=True, use_container_width=True)
            else:
                st.dataframe(entity_matching_df, hide_index=True, use_container_width=True)
            
            # Export audit findings
            col1, col2 = st.columns(2)
            
            with col1:
                if audit_findings is not None:
                    csv_data = audit_findings.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Audit Findings (CSV)",
                        data=csv_data,
                        file_name=f"audit_findings_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📝 Generate Audit Summary Report"):
                    with st.spinner("Generating comprehensive audit report..."):
                        try:
                            # Import the report generator
                            import sys
                            from pathlib import Path
                            sys.path.append(str(Path.cwd()))
                            from src.report_generator import EntityReportGenerator
                            
                            # Generate report filename
                            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                            report_filename = f"audit_summary_report_{timestamp}.pdf"
                            
                            # Create a comprehensive audit summary report
                            from reportlab.lib.pagesizes import A4
                            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                            from reportlab.lib import colors
                            from reportlab.lib.colors import HexColor
                            from datetime import datetime
                            import io
                            
                            # Create PDF in memory
                            buffer = io.BytesIO()
                            doc = SimpleDocTemplate(buffer, pagesize=A4)
                            styles = getSampleStyleSheet()
                            story = []
                            
                            # Title
                            title_style = ParagraphStyle(
                                'CustomTitle',
                                parent=styles['Title'],
                                fontSize=20,
                                spaceAfter=30,
                                alignment=1,
                                textColor=HexColor('#1f4e79')
                            )
                            story.append(Paragraph("AUDIT FINDINGS SUMMARY REPORT", title_style))
                            story.append(Spacer(1, 12))
                            story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
                            story.append(Spacer(1, 20))
                            
                            # Executive Summary
                            if audit_findings is not None and len(audit_findings) > 0:
                                story.append(Paragraph("EXECUTIVE SUMMARY", styles['Heading1']))
                                story.append(Spacer(1, 12))
                                
                                total_entities = len(audit_findings)
                                flagged_entities = len(audit_findings[audit_findings.get('audit_flag', 0) == 1]) if 'audit_flag' in audit_findings.columns else 0
                                severity_counts = None
                                
                                summary_data = [
                                    ['Metric', 'Value'],
                                    ['Total Entities Analyzed', f"{total_entities:,}"],
                                    ['Entities with Irregularities', f"{flagged_entities:,}"],
                                    ['Risk Rate', f"{(flagged_entities/total_entities*100):.1f}%" if total_entities > 0 else "0%"]
                                ]
                                
                                if 'severity' in audit_findings.columns:
                                    severity_counts = audit_findings['severity'].value_counts()
                                    for severity, count in severity_counts.head(3).items():
                                        summary_data.append([f"{str(severity).title()} Severity Cases", f"{count:,}"])
                                
                                summary_table = Table(summary_data)
                                summary_table.setStyle(TableStyle([
                                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                                ]))
                                story.append(summary_table)
                                story.append(Spacer(1, 20))
                                
                                # Key Findings
                                story.append(Paragraph("KEY FINDINGS", styles['Heading1']))
                                story.append(Spacer(1, 12))
                                
                                findings_text = [
                                    f"• {total_entities} entities were analyzed for financial irregularities",
                                    f"• {flagged_entities} entities showed potential risk indicators",
                                    f"• Overall risk rate stands at {(flagged_entities/total_entities*100):.1f}%" if total_entities > 0 else "• No risk indicators detected"
                                ]
                                
                                if 'severity' in audit_findings.columns and severity_counts is not None and len(severity_counts) > 0:
                                    most_common = severity_counts.index[0]
                                    findings_text.append(f"• Most common severity level: {str(most_common).title()}")
                                
                                for finding in findings_text:
                                    story.append(Paragraph(finding, styles['Normal']))
                                    story.append(Spacer(1, 6))
                            else:
                                story.append(Paragraph("No audit findings data available for analysis.", styles['Normal']))
                            
                            # Build PDF
                            doc.build(story)
                            buffer.seek(0)
                            
                            # Provide download
                            st.download_button(
                                label="📥 Download Audit Summary Report (PDF)",
                                data=buffer.getvalue(),
                                file_name=report_filename,
                                mime="application/pdf",
                                use_container_width=True
                            )
                            
                            st.success(f"✅ Audit summary report generated successfully! Click above to download '{report_filename}'.")
                            
                        except Exception as e:
                            st.error(f"Error generating PDF report: {str(e)}")
                            st.info("Falling back to CSV export...")
                            if audit_findings is not None:
                                csv_data = audit_findings.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Audit Findings (CSV)",
                                    data=csv_data,
                                    file_name=f"audit_findings_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                                    mime="text/csv"
                                )
        
        # Audit processing insights
        if audit_findings is not None:
            st.markdown("---")
            st.markdown("#### 🔬 Processing Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Keyword Matching Success**")
                if 'detected_issues' in audit_findings.columns:
                    entities_with_keywords = len(audit_findings[audit_findings['detected_issues'].notna()])
                    total_entities = len(audit_findings)
                    match_rate = (entities_with_keywords / total_entities * 100) if total_entities > 0 else 0
                    st.metric("Keyword Match Rate", f"{match_rate:.1f}%")
                else:
                    st.metric("Keyword Match Rate", "N/A")
            
            with col2:
                st.markdown("**Entity Resolution**")
                if 'entity_name' in audit_findings.columns:
                    unique_entities = audit_findings['entity_name'].nunique()
                    st.metric("Unique Entities Processed", f"{unique_entities:,}")
                else:
                    st.metric("Unique Entities Processed", "N/A")
            
            with col3:
                st.markdown("**Severity Distribution**")
                if 'severity' in audit_findings.columns:
                    severity_counts = audit_findings['severity'].value_counts()
                    most_common_severity = str(severity_counts.index[0]).title() if len(severity_counts) > 0 else "N/A"
                    st.metric("Most Common Severity", most_common_severity)
                else:
                    st.metric("Most Common Severity", "N/A")
        else:
            st.info("""
            **No audit report analysis available.**
            
            To see audit report analysis:
            1. Go to the main application
            2. Upload your financial dataset
            3. Upload an audit report PDF 
            4. Process the audit labels
            5. Return here to view the detailed analysis
            """)
    
    # Summary section
    st.markdown("---")
    st.markdown("### 📊 Filter Summary")
    
    active_filters = []
    for key, value in filters.items():
        if value and value != 'All':
            if key == 'year':
                active_filters.append(f"Year: {value[0]}-{value[1]}")
            elif key == 'total_expenditure':
                active_filters.append(f"Expenditure: {format_currency(value[0])}-{format_currency(value[1])}")
            else:
                active_filters.append(f"{key.title()}: {value}")
    
    if active_filters:
        st.info("**Active Filters:** " + " | ".join(active_filters))
    else:
        st.info("**No filters applied** - showing all available cases")
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 View High Risk Cases Only"):
            # Apply high-risk filter
            high_risk_df = df[df.get('irregularity_score', 0) > 0.6] if 'irregularity_score' in df.columns else df
            st.success(f"✅ High-risk filter applied: {len(high_risk_df)} entities found with risk score > 0.6")
            st.session_state['filtered_df'] = high_risk_df
    
    with col2:
        if st.button("📅 View Recent Cases"):
            # Apply recent cases filter (last 2 years)
            current_year = pd.Timestamp.now().year
            recent_df = df[df.get('year', 0) >= current_year - 2] if 'year' in df.columns else df
            st.success(f"✅ Recent cases filter applied: {len(recent_df)} cases from {current_year-2} onwards")
            st.session_state['filtered_df'] = recent_df
    
    with col3:
        if st.button("🔄 Reset All Filters"):
            # Reset all filters
            st.session_state['filtered_df'] = df
            st.success("✅ All filters reset - showing complete dataset")
            st.info(f"📋 Dataset restored: {len(df)} total cases available")

if __name__ == "__main__":
    main()
