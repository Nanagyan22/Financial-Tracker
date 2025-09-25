import pandas as pd
import numpy as np
import pdfplumber
import re
from fuzzywuzzy import fuzz, process
from typing import List, Dict, Tuple, Union
import streamlit as st
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from collections import defaultdict

# Download NLTK data if not available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class AuditLabelGenerator:
    """Generate audit labels from PDF reports."""
    
    def __init__(self):
        self.irregularity_keywords = [
            # Financial irregularities
            'irregularity', 'irregular', 'irregularities',
            'fraud', 'fraudulent', 'embezzlement',
            'misappropriation', 'unauthorized', 'unapproved',
            'overpayment', 'underpayment', 'overcharge',
            'unearned', 'unaccounted', 'missing', 'shortfall',
            'discrepancy', 'variance', 'error', 'mistake',
            
            # Procurement issues
            'procurement', 'contract', 'tender', 'bid',
            'single source', 'sole source', 'no competition',
            'inflated', 'overpriced', 'excessive',
            'bid rigging', 'collusion',
            
            # Payroll and HR
            'payroll', 'salary', 'allowance', 'overtime',
            'ghost worker', 'phantom', 'duplicate',
            'unauthorised deduction', 'wrong computation',
            
            # Tax and revenue
            'tax', 'duty', 'levy', 'fee', 'charge',
            'undercharged', 'underassessed', 'exemption',
            'waiver', 'concession', 'rebate',
            
            # Asset management
            'asset', 'property', 'equipment', 'vehicle',
            'disposal', 'write-off', 'depreciation',
            'maintenance', 'repair', 'upgrade',
            
            # Financial controls
            'internal control', 'weak control', 'control failure',
            'reconciliation', 'bank reconciliation',
            'cash management', 'petty cash',
            
            # Compliance
            'non-compliance', 'violation', 'breach',
            'regulation', 'policy', 'procedure',
            'documentation', 'record keeping',
            
            # Recovery actions
            'recover', 'recovery', 'refund', 'repay',
            'surcharge', 'penalty', 'fine'
        ]
        
        self.severity_keywords = {
            'high': ['fraud', 'embezzlement', 'ghost worker', 'missing', 'unauthorized'],
            'medium': ['irregularity', 'discrepancy', 'non-compliance', 'overcharge'],
            'low': ['error', 'mistake', 'variance', 'documentation']
        }
        
        # Common stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def extract_text_from_pdf(self, pdf_path: Union[str, Path, object]) -> str:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file or uploaded file object
            
        Returns:
            Extracted text as string
        """
        try:
            text = ""
            
            if hasattr(pdf_path, 'read'):
                # Streamlit uploaded file
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            else:
                # File path
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for analysis.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and common headers/footers
        text = re.sub(r'page \d+', '', text)
        text = re.sub(r'ghana audit service', '', text)
        text = re.sub(r'auditor[- ]general', '', text)
        
        return text.strip()
    
    def extract_entity_mentions(self, text: str, entity_names: List[str]) -> Dict[str, List[Dict]]:
        """
        Extract mentions of entities from text along with context.
        
        Args:
            text: Preprocessed text
            entity_names: List of entity names to search for
            
        Returns:
            Dictionary mapping entity names to mention contexts
        """
        mentions = defaultdict(list)
        
        # Split text into sentences for context
        sentences = re.split(r'[.!?]+', text)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Check each entity name
            for entity in entity_names:
                entity_clean = self.clean_entity_name(entity)
                
                # Direct match
                if entity_clean.lower() in sentence:
                    mentions[entity].append({
                        'sentence': sentence,
                        'sentence_index': i,
                        'match_type': 'direct'
                    })
                    continue
                
                # Fuzzy match for partial names
                entity_words = entity_clean.lower().split()
                if len(entity_words) > 1:
                    # Check if most words of entity name appear in sentence
                    word_matches = sum(1 for word in entity_words if word in sentence)
                    if word_matches >= len(entity_words) * 0.6:  # At least 60% of words match
                        mentions[entity].append({
                            'sentence': sentence,
                            'sentence_index': i,
                            'match_type': 'fuzzy'
                        })
        
        return dict(mentions)
    
    def clean_entity_name(self, name: str) -> str:
        """Clean entity name for matching."""
        if pd.isna(name):
            return ""
        
        name = str(name).strip()
        
        # Remove common prefixes/suffixes
        name = re.sub(r'\b(limited|ltd|company|corp|corporation|inc)\b', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\b(the|of|and|&)\b', ' ', name, flags=re.IGNORECASE)
        
        # Clean extra spaces
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    def detect_irregularities(self, mentions: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """
        Detect irregularities for entities based on context.
        
        Args:
            mentions: Dictionary of entity mentions with contexts
            
        Returns:
            Dictionary with irregularity information for each entity
        """
        results = {}
        
        for entity, contexts in mentions.items():
            irregularity_score = 0
            detected_issues = []
            severity = 'low'
            evidence_sentences = []
            
            for context in contexts:
                sentence = context['sentence'].lower()
                
                # Check for irregularity keywords
                for keyword in self.irregularity_keywords:
                    if keyword in sentence:
                        irregularity_score += 1
                        detected_issues.append(keyword)
                        evidence_sentences.append(sentence)
                        
                        # Determine severity
                        for sev_level, sev_keywords in self.severity_keywords.items():
                            if keyword in sev_keywords:
                                if sev_level == 'high':
                                    severity = 'high'
                                elif sev_level == 'medium' and severity != 'high':
                                    severity = 'medium'
            
            # Calculate final label
            audit_flag = 1 if irregularity_score > 0 else 0
            
            results[entity] = {
                'audit_flag': audit_flag,
                'irregularity_score': irregularity_score,
                'detected_issues': list(set(detected_issues)),
                'severity': severity,
                'evidence_count': len(evidence_sentences),
                'evidence_sentences': evidence_sentences[:3]  # Keep top 3 for review
            }
        
        return results
    
    def match_entities_fuzzy(self, pdf_entities: List[str], dataset_entities: List[str], threshold: int = 80) -> Dict[str, str]:
        """
        Match entities from PDF to dataset using fuzzy matching.
        
        Args:
            pdf_entities: Entities found in PDF
            dataset_entities: Entities in dataset
            threshold: Minimum similarity score
            
        Returns:
            Dictionary mapping PDF entities to dataset entities
        """
        matches = {}
        
        for pdf_entity in pdf_entities:
            pdf_clean = self.clean_entity_name(pdf_entity)
            
            # Find best match
            best_match = process.extractOne(
                pdf_clean, 
                [self.clean_entity_name(e) for e in dataset_entities],
                scorer=fuzz.token_sort_ratio
            )
            
            if best_match and best_match[1] >= threshold:
                # Find original entity name
                for orig_entity in dataset_entities:
                    if self.clean_entity_name(orig_entity) == best_match[0]:
                        matches[pdf_entity] = orig_entity
                        break
        
        return matches

def process_pdf_labels(df: pd.DataFrame, pdf_path: Union[str, Path, object]) -> pd.DataFrame:
    """
    Main function to process PDF and generate labels for dataset.
    
    Args:
        df: Dataset DataFrame
        pdf_path: Path to PDF file or uploaded file object
        
    Returns:
        DataFrame with audit labels added
    """
    try:
        # Initialize label generator
        generator = AuditLabelGenerator()
        
        # Extract text from PDF
        print("Extracting text from PDF...")
        text = generator.extract_text_from_pdf(pdf_path)
        
        if not text:
            raise ValueError("No text could be extracted from PDF")
        
        # Preprocess text
        text = generator.preprocess_text(text)
        
        # Get entity names from dataset
        entity_names = df['entity_name'].dropna().unique().tolist()
        
        # Extract entity mentions
        print("Extracting entity mentions...")
        mentions = generator.extract_entity_mentions(text, entity_names)
        
        # Detect irregularities
        print("Detecting irregularities...")
        irregularities = generator.detect_irregularities(mentions)
        
        # Create labels DataFrame
        df_labeled = df.copy()
        df_labeled['audit_flag'] = 0
        df_labeled['irregularity_score'] = 0
        df_labeled['detected_issues'] = ''
        df_labeled['severity'] = 'none'
        df_labeled['evidence_count'] = 0
        
        # Apply labels
        for entity, info in irregularities.items():
            mask = df_labeled['entity_name'] == entity
            df_labeled.loc[mask, 'audit_flag'] = info['audit_flag']
            df_labeled.loc[mask, 'irregularity_score'] = info['irregularity_score']
            df_labeled.loc[mask, 'detected_issues'] = ', '.join(info['detected_issues'])
            df_labeled.loc[mask, 'severity'] = info['severity']
            df_labeled.loc[mask, 'evidence_count'] = info['evidence_count']
        
        # Save labels review file
        review_data = []
        for entity, info in irregularities.items():
            review_data.append({
                'entity_name': entity,
                'audit_flag': info['audit_flag'],
                'irregularity_score': info['irregularity_score'],
                'detected_issues': ', '.join(info['detected_issues']),
                'severity': info['severity'],
                'evidence_sentences': ' | '.join(info['evidence_sentences'])
            })
        
        if review_data:
            review_df = pd.DataFrame(review_data)
            output_dir = Path("data/processed")
            output_dir.mkdir(parents=True, exist_ok=True)
            review_df.to_csv(output_dir / "labels_review.csv", index=False)
        
        print(f"Generated labels for {len(irregularities)} entities")
        print(f"Total flagged entities: {df_labeled['audit_flag'].sum()}")
        
        return df_labeled
        
    except Exception as e:
        print(f"Error processing PDF labels: {e}")
        # Return original DataFrame with default labels if PDF processing fails
        df_with_labels = df.copy()
        df_with_labels['audit_flag'] = 0
        return df_with_labels
