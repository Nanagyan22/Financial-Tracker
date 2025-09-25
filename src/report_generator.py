"""
Entity Report Generator

This module creates comprehensive PDF reports for individual entities
including ML predictions, risk analysis, SHAP explanations, and recommendations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, red, green, blue
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.platypus.flowables import HRFlowable
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.express as px
from pathlib import Path
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import joblib
import io
import base64

class EntityReportGenerator:
    """
    Generate comprehensive PDF reports for individual entities.
    """
    
    def __init__(self, data_path="data/processed/processed_dataset.pkl"):
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.feature_names = []
        self.explanations = None
        
        # Report styling
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center
            textColor=HexColor('#1f4e79')
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            textColor=HexColor('#2e75b6')
        )
        
        self.subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            textColor=HexColor('#4472c4')
        )
        
        self.normal_style = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6
        )
        
        self.alert_style = ParagraphStyle(
            'AlertStyle',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            textColor=red,
            backColor=HexColor('#ffebee')
        )
        
        self.success_style = ParagraphStyle(
            'SuccessStyle',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            textColor=green,
            backColor=HexColor('#e8f5e8')
        )
        
        # Load data and models
        self.load_data()
        self.load_models()
    
    def load_data(self):
        """Load the processed dataset."""
        try:
            if os.path.exists(self.data_path):
                self.df = joblib.load(self.data_path)
                print(f"✅ Loaded dataset with {len(self.df)} records")
            else:
                # Try to load from Excel as fallback
                excel_path = "data/raw/Final Dataset.xlsx"
                if os.path.exists(excel_path):
                    self.df = pd.read_excel(excel_path)
                    print(f"✅ Loaded dataset from Excel with {len(self.df)} records")
                else:
                    print("❌ No dataset found")
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def load_models(self):
        """Load trained models and artifacts."""
        try:
            # Load best model
            if os.path.exists("models/best_model.pkl"):
                self.models['best'] = joblib.load("models/best_model.pkl")
                print("✅ Loaded best model")
            
            # Load feature names
            if os.path.exists("models/feature_names.pkl"):
                self.feature_names = joblib.load("models/feature_names.pkl")
                print(f"✅ Loaded {len(self.feature_names)} feature names")
            
            # Load explainer
            if os.path.exists("models/explainer.pkl"):
                self.explanations = joblib.load("models/explainer.pkl")
                print("✅ Loaded SHAP explainer")
                
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def get_entity_data(self, entity_name: str) -> Optional[pd.DataFrame]:
        """Get data for a specific entity."""
        if self.df is None:
            return None
        
        # Find the entity column name
        entity_col = None
        possible_cols = ['state_entity', 'entity_name', 'Entity', 'Entity Name', 'entity']
        
        for col in possible_cols:
            if col in self.df.columns:
                entity_col = col
                break
        
        if entity_col is None:
            print("❌ No entity column found in dataset")
            return None
        
        # Try exact match first
        entity_data = self.df[self.df[entity_col] == entity_name]
        
        if entity_data.empty:
            # Try partial match
            entity_data = self.df[self.df[entity_col].str.contains(entity_name, case=False, na=False)]
        
        return entity_data if not entity_data.empty else None
    
    def generate_risk_score(self, entity_data: pd.DataFrame) -> Dict:
        """Calculate comprehensive risk score for entity."""
        risk_metrics = {
            'overall_score': 0.0,
            'audit_risk': 0.0,
            'financial_risk': 0.0,
            'activity_risk': 0.0,
            'prediction_confidence': 0.0,
            'risk_level': 'Low',
            'recommendations': []
        }
        
        if entity_data.empty:
            return risk_metrics
        
        try:
            # Audit risk component (check multiple possible column names)
            audit_col = None
            for col in ['audit_flag', 'dependet_variable', 'dependent_variable', 'target', 'flag']:
                if col in entity_data.columns:
                    audit_col = col
                    break
            
            if audit_col:
                audit_flags = entity_data[audit_col].sum()
                total_records = len(entity_data)
                risk_metrics['audit_risk'] = audit_flags / total_records if total_records > 0 else 0.0
            
            # Financial risk component
            if 'total_expenditure' in entity_data.columns:
                expenditure_values = entity_data['total_expenditure'].fillna(0)
                if len(expenditure_values) > 0:
                    avg_expenditure = expenditure_values.mean()
                    max_expenditure = expenditure_values.max()
                    
                    # Risk increases with high and variable expenditure
                    expenditure_cv = expenditure_values.std() / avg_expenditure if avg_expenditure > 0 else 0
                    risk_metrics['financial_risk'] = min(1.0, expenditure_cv * 0.5 + (max_expenditure / 10000000) * 0.1)
            
            # Activity risk component (frequent activity can indicate higher risk)
            activity_years = len(entity_data['year'].unique()) if 'year' in entity_data.columns else 1
            risk_metrics['activity_risk'] = min(1.0, activity_years / 10.0)
            
            # ML prediction risk (if available)
            if self.models and 'best' in self.models and self.feature_names:
                try:
                    # Create feature dataframe with all expected features
                    feature_df = pd.DataFrame()
                    
                    for feature_name in self.feature_names:
                        if feature_name in entity_data.columns:
                            feature_df[feature_name] = entity_data[feature_name].fillna(0)
                        else:
                            # Fill missing features with zeros
                            feature_df[feature_name] = 0
                    
                    if not feature_df.empty:
                        if hasattr(self.models['best'], 'predict_proba'):
                            predictions = self.models['best'].predict_proba(feature_df)[:, 1]
                            risk_metrics['prediction_confidence'] = np.mean(predictions)
                except Exception as e:
                    print(f"Error in ML prediction: {e}")
                    risk_metrics['prediction_confidence'] = 0.0
            
            # Calculate overall score
            weights = {'audit_risk': 0.4, 'financial_risk': 0.2, 'activity_risk': 0.1, 'prediction_confidence': 0.3}
            risk_metrics['overall_score'] = sum(
                risk_metrics[component] * weight for component, weight in weights.items()
            )
            
            # Determine risk level
            if risk_metrics['overall_score'] >= 0.7:
                risk_metrics['risk_level'] = 'Critical'
                risk_metrics['recommendations'] = [
                    'Immediate audit review required',
                    'Enhanced monitoring recommended',
                    'Financial controls assessment needed'
                ]
            elif risk_metrics['overall_score'] >= 0.4:
                risk_metrics['risk_level'] = 'High'
                risk_metrics['recommendations'] = [
                    'Detailed audit recommended',
                    'Quarterly monitoring suggested',
                    'Financial patterns review needed'
                ]
            elif risk_metrics['overall_score'] >= 0.2:
                risk_metrics['risk_level'] = 'Medium'
                risk_metrics['recommendations'] = [
                    'Annual audit sufficient',
                    'Standard monitoring protocols',
                    'Periodic expenditure review'
                ]
            else:
                risk_metrics['risk_level'] = 'Low'
                risk_metrics['recommendations'] = [
                    'Routine monitoring adequate',
                    'Standard compliance checks',
                    'Normal audit schedule'
                ]
                
        except Exception as e:
            print(f"Error calculating risk score: {e}")
        
        return risk_metrics
    
    def create_summary_chart(self, entity_data: pd.DataFrame, entity_name: str) -> str:
        """Create summary visualization chart."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Financial Summary - {entity_name}', fontsize=16, fontweight='bold')
            
            # 1. Expenditure over time
            if 'year' in entity_data.columns and 'total_expenditure' in entity_data.columns:
                yearly_exp = entity_data.groupby('year')['total_expenditure'].sum().reset_index()
                ax1.plot(yearly_exp['year'], yearly_exp['total_expenditure'], marker='o', linewidth=2)
                ax1.set_title('Expenditure Over Time')
                ax1.set_xlabel('Year')
                ax1.set_ylabel('Total Expenditure (GHS)')
                ax1.grid(True, alpha=0.3)
                
                # Format y-axis for large numbers
                ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
            else:
                ax1.text(0.5, 0.5, 'Expenditure data\nnot available', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Expenditure Over Time')
            
            # 2. Audit flags over time
            audit_col = None
            for col in ['audit_flag', 'dependet_variable', 'dependent_variable', 'target', 'flag']:
                if col in entity_data.columns:
                    audit_col = col
                    break
            
            if 'year' in entity_data.columns and audit_col:
                yearly_flags = entity_data.groupby('year')[audit_col].agg(['sum', 'count']).reset_index()
                yearly_flags['flag_rate'] = yearly_flags['sum'] / yearly_flags['count'] * 100
                
                bars = ax2.bar(yearly_flags['year'], yearly_flags['flag_rate'], color=['red' if x > 50 else 'orange' if x > 20 else 'green' for x in yearly_flags['flag_rate']])
                ax2.set_title('Audit Flag Rate by Year')
                ax2.set_xlabel('Year')
                ax2.set_ylabel('Flag Rate (%)')
                ax2.set_ylim(0, 100)
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Audit data\nnot available', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Audit Flag Rate by Year')
            
            # 3. Risk distribution
            risk_metrics = self.generate_risk_score(entity_data)
            components = ['Audit Risk', 'Financial Risk', 'Activity Risk', 'ML Confidence']
            values = [risk_metrics['audit_risk'], risk_metrics['financial_risk'], 
                     risk_metrics['activity_risk'], risk_metrics['prediction_confidence']]
            colors = ['#ff4444', '#ff8800', '#4488ff', '#44aa44']
            
            bars = ax3.bar(components, values, color=colors)
            ax3.set_title('Risk Component Analysis')
            ax3.set_ylabel('Risk Score (0-1)')
            ax3.set_ylim(0, 1)
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # 4. Overall risk gauge
            overall_score = risk_metrics['overall_score']
            risk_level = risk_metrics['risk_level']
            
            # Create gauge-like visualization
            theta = np.linspace(0, np.pi, 100)
            radius = 1
            
            # Background semicircle
            ax4.fill_between(theta, 0, radius, alpha=0.1, color='gray')
            
            # Risk zones
            ax4.fill_between(theta[0:33], 0, radius, alpha=0.3, color='green', label='Low Risk')
            ax4.fill_between(theta[33:66], 0, radius, alpha=0.3, color='orange', label='Medium Risk')
            ax4.fill_between(theta[66:100], 0, radius, alpha=0.3, color='red', label='High Risk')
            
            # Risk needle
            angle = np.pi * (1 - overall_score)
            ax4.arrow(0, 0, 0.8 * np.cos(angle), 0.8 * np.sin(angle), 
                     head_width=0.05, head_length=0.1, fc='black', ec='black')
            
            ax4.set_xlim(-1.2, 1.2)
            ax4.set_ylim(0, 1.2)
            ax4.set_aspect('equal')
            ax4.set_title(f'Overall Risk Level: {risk_level}')
            ax4.axis('off')
            ax4.legend(loc='upper right')
            
            plt.tight_layout()
            
            # Save chart
            chart_path = f"temp_entity_chart_{entity_name.replace(' ', '_')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            print(f"Error creating summary chart: {e}")
            return None
    
    def create_shap_explanation_chart(self, entity_data: pd.DataFrame, entity_name: str) -> str:
        """Create SHAP explanation visualization."""
        try:
            if not self.explanations or not self.models or 'best' not in self.models:
                return None
            
            # Prepare features for explanation
            feature_cols = [col for col in self.feature_names if col in entity_data.columns]
            if not feature_cols:
                return None
            
            features = entity_data[feature_cols].fillna(0)
            
            # Get SHAP values for the most recent record
            latest_record = features.iloc[-1:].values
            
            # Simple feature importance visualization (fallback for SHAP)
            try:
                if hasattr(self.models['best'], 'feature_importances_'):
                    importances = self.models['best'].feature_importances_
                    indices = np.argsort(importances)[::-1][:15]  # Top 15 features
                    
                    plt.figure(figsize=(10, 8))
                    plt.title(f'Feature Importance - {entity_name}')
                    plt.barh(range(len(indices)), importances[indices])
                    plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
                    plt.xlabel('Feature Importance')
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    
                    chart_path = f"temp_shap_chart_{entity_name.replace(' ', '_')}.png"
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    return chart_path
                    
            except Exception as e:
                print(f"Error creating feature importance chart: {e}")
                return None
                
        except Exception as e:
            print(f"Error creating SHAP explanation: {e}")
            return None
    
    def generate_entity_report(self, entity_name: str, output_path: Optional[str] = None) -> str:
        """Generate comprehensive PDF report for an entity."""
        
        if output_path is None:
            safe_name = entity_name.replace(' ', '_').replace('/', '_')
            output_path = f"reports/entity_report_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Create reports directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get entity data
        entity_data = self.get_entity_data(entity_name)
        if entity_data is None or entity_data.empty:
            raise ValueError(f"No data found for entity: {entity_name}")
        
        # Calculate risk metrics
        risk_metrics = self.generate_risk_score(entity_data)
        
        # Create visualizations
        summary_chart = self.create_summary_chart(entity_data, entity_name)
        shap_chart = self.create_shap_explanation_chart(entity_data, entity_name)
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        
        # Title page
        story.append(Paragraph("FINANCIAL IRREGULARITY ANALYSIS REPORT", self.title_style))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Entity: {entity_name}", self.heading_style))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", self.normal_style))
        story.append(Spacer(1, 6))
        story.append(Paragraph("Ghana Financial Irregularities Detection System", self.normal_style))
        story.append(HRFlowable(width="100%", thickness=1, lineCap='round', color=HexColor('#1f4e79')))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("EXECUTIVE SUMMARY", self.heading_style))
        story.append(Spacer(1, 12))
        
        # Risk level summary
        risk_color = HexColor('#d32f2f') if risk_metrics['risk_level'] in ['Critical', 'High'] else HexColor('#ff9800') if risk_metrics['risk_level'] == 'Medium' else HexColor('#388e3c')
        risk_style = ParagraphStyle(
            'RiskStyle',
            parent=self.normal_style,
            textColor=risk_color,
            fontSize=14,
            fontName='Helvetica-Bold'
        )
        
        story.append(Paragraph(f"Overall Risk Level: {risk_metrics['risk_level']}", risk_style))
        story.append(Paragraph(f"Risk Score: {risk_metrics['overall_score']:.2f}/1.00", self.normal_style))
        story.append(Spacer(1, 12))
        
        # Entity overview
        total_records = len(entity_data)
        years_active = entity_data['year'].nunique() if 'year' in entity_data.columns else 1
        total_expenditure = entity_data['total_expenditure'].sum() if 'total_expenditure' in entity_data.columns else 0
        
        # Find audit column
        audit_col = None
        for col in ['audit_flag', 'dependet_variable', 'dependent_variable', 'target', 'flag']:
            if col in entity_data.columns:
                audit_col = col
                break
        
        audit_flags = entity_data[audit_col].sum() if audit_col else 0
        
        overview_data = [
            ['Metric', 'Value'],
            ['Total Records Analyzed', f"{total_records:,}"],
            ['Years of Activity', f"{years_active}"],
            ['Total Expenditure', f"GHS {total_expenditure:,.2f}"],
            ['Audit Flags Raised', f"{audit_flags}"],
            ['Flag Rate', f"{(audit_flags/total_records*100):.1f}%" if total_records > 0 else "N/A"]
        ]
        
        overview_table = Table(overview_data, colWidths=[2.5*inch, 2*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1f4e79')),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f5f5f5')),
            ('GRID', (0, 0), (-1, -1), 1, black),
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 20))
        
        # Risk Analysis Section
        story.append(Paragraph("RISK ANALYSIS", self.heading_style))
        story.append(Spacer(1, 12))
        
        risk_components = [
            ['Risk Component', 'Score', 'Level'],
            ['Audit Risk', f"{risk_metrics['audit_risk']:.2f}", self._get_risk_level(risk_metrics['audit_risk'])],
            ['Financial Risk', f"{risk_metrics['financial_risk']:.2f}", self._get_risk_level(risk_metrics['financial_risk'])],
            ['Activity Risk', f"{risk_metrics['activity_risk']:.2f}", self._get_risk_level(risk_metrics['activity_risk'])],
            ['ML Prediction', f"{risk_metrics['prediction_confidence']:.2f}", self._get_risk_level(risk_metrics['prediction_confidence'])]
        ]
        
        risk_table = Table(risk_components, colWidths=[2*inch, 1*inch, 1.5*inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2e75b6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f5f5f5')),
            ('GRID', (0, 0), (-1, -1), 1, black),
        ]))
        story.append(risk_table)
        story.append(Spacer(1, 20))
        
        # Add summary chart if available
        if summary_chart and os.path.exists(summary_chart):
            story.append(Paragraph("FINANCIAL ANALYSIS DASHBOARD", self.heading_style))
            story.append(Spacer(1, 12))
            story.append(Image(summary_chart, width=6*inch, height=5*inch))
            story.append(Spacer(1, 20))
        
        # Page break before detailed analysis
        story.append(PageBreak())
        
        # Detailed Analysis
        story.append(Paragraph("DETAILED ANALYSIS", self.heading_style))
        story.append(Spacer(1, 12))
        
        # Historical trends
        if 'year' in entity_data.columns:
            story.append(Paragraph("Historical Activity Trends", self.subheading_style))
            
            # Build aggregation dict based on available columns
            agg_dict = {}
            if audit_col:
                agg_dict[audit_col] = 'sum'
            if 'total_expenditure' in entity_data.columns:
                agg_dict['total_expenditure'] = 'sum'
            
            yearly_summary = entity_data.groupby('year').agg(agg_dict).reset_index()
            
            trend_text = "Analysis of year-over-year patterns:\n"
            for _, row in yearly_summary.iterrows():
                year = row['year']
                audit_count = row.get(audit_col, 0) if audit_col else 0
                expenditure = row.get('total_expenditure', 0)
                trend_text += f"• {year}: {audit_count} flags, GHS {expenditure:,.2f} expenditure\n"
            
            story.append(Paragraph(trend_text, self.normal_style))
            story.append(Spacer(1, 12))
        
        # Add SHAP chart if available
        if shap_chart and os.path.exists(shap_chart):
            story.append(Paragraph("MACHINE LEARNING FEATURE ANALYSIS", self.subheading_style))
            story.append(Paragraph("The following chart shows which factors most influence the risk prediction for this entity:", self.normal_style))
            story.append(Spacer(1, 6))
            story.append(Image(shap_chart, width=6*inch, height=4*inch))
            story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(Paragraph("RECOMMENDATIONS", self.heading_style))
        story.append(Spacer(1, 12))
        
        for i, recommendation in enumerate(risk_metrics['recommendations'], 1):
            story.append(Paragraph(f"{i}. {recommendation}", self.normal_style))
            story.append(Spacer(1, 6))
        
        # Additional recommendations based on data
        story.append(Spacer(1, 12))
        story.append(Paragraph("Additional Considerations:", self.subheading_style))
        
        additional_recs = []
        if audit_flags > 0:
            additional_recs.append("Focus on addressing identified audit issues in future assessments")
        if total_expenditure > 100000000:  # 100M GHS
            additional_recs.append("Implement enhanced financial controls due to high expenditure volume")
        if years_active > 5:
            additional_recs.append("Consider comprehensive multi-year trend analysis")
        
        for i, rec in enumerate(additional_recs, len(risk_metrics['recommendations']) + 1):
            story.append(Paragraph(f"{i}. {rec}", self.normal_style))
            story.append(Spacer(1, 6))
        
        # Footer
        story.append(Spacer(1, 30))
        story.append(HRFlowable(width="100%", thickness=1, lineCap='round', color=HexColor('#1f4e79')))
        story.append(Spacer(1, 12))
        story.append(Paragraph("This report was generated by the Ghana Financial Irregularities Detection System", 
                              ParagraphStyle('Footer', parent=self.normal_style, fontSize=9, textColor=HexColor('#666666'), alignment=1)))
        story.append(Paragraph(f"Report ID: ENT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hash(entity_name) % 10000}", 
                              ParagraphStyle('Footer', parent=self.normal_style, fontSize=9, textColor=HexColor('#666666'), alignment=1)))
        
        # Build PDF
        doc.build(story)
        
        # Clean up temporary files
        temp_files = [summary_chart, shap_chart]
        for temp_file in temp_files:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        
        print(f"✅ Generated report: {output_path}")
        return output_path
    
    def _get_risk_level(self, score: float) -> str:
        """Convert risk score to level."""
        if score >= 0.7:
            return "Critical"
        elif score >= 0.4:
            return "High" 
        elif score >= 0.2:
            return "Medium"
        else:
            return "Low"

def generate_batch_reports(entity_names: List[str], output_dir: str = "reports") -> List[str]:
    """Generate reports for multiple entities."""
    generator = EntityReportGenerator()
    generated_reports = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    for entity_name in entity_names:
        try:
            safe_name = entity_name.replace(' ', '_').replace('/', '_')
            output_path = os.path.join(output_dir, f"entity_report_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
            
            report_path = generator.generate_entity_report(entity_name, output_path)
            generated_reports.append(report_path)
            print(f"✅ Generated report for {entity_name}")
            
        except Exception as e:
            print(f"❌ Failed to generate report for {entity_name}: {e}")
    
    return generated_reports

if __name__ == "__main__":
    # Example usage
    generator = EntityReportGenerator()
    
    if generator.df is not None:
        # Get sample entity for testing
        sample_entities = generator.df['entity_name'].unique()[:3]
        
        print(f"Generating sample reports for: {sample_entities}")
        for entity in sample_entities:
            try:
                report_path = generator.generate_entity_report(entity)
                print(f"✅ Report generated: {report_path}")
            except Exception as e:
                print(f"❌ Error generating report for {entity}: {e}")
    else:
        print("❌ No data available for report generation")