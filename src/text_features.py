"""
Advanced Text Feature Engineering with TF-IDF for PDF Audit Reports

This module extends the existing audit text processing with TF-IDF vectorization
to create sophisticated text features for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
import re
from pathlib import Path
import joblib

class TFIDFFeatureExtractor:
    """
    Extract TF-IDF features from audit text data.
    """
    
    def __init__(self, 
                 max_features: int = 100,
                 max_df: float = 0.8,
                 min_df: int = 2,
                 ngram_range: Tuple[int, int] = (1, 3),
                 svd_components: int = 50):
        """
        Initialize TF-IDF feature extractor.
        
        Args:
            max_features: Maximum number of TF-IDF features
            max_df: Maximum document frequency threshold
            min_df: Minimum document frequency threshold
            ngram_range: N-gram range for feature extraction
            svd_components: Number of SVD components for dimensionality reduction
        """
        self.max_features = max_features
        self.max_df = max_df
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.svd_components = svd_components
        
        # Initialize models
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            max_df=max_df,
            min_df=min(min_df, 1),  # Handle small datasets
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        self.svd = TruncatedSVD(n_components=svd_components, random_state=42)
        self.scaler = MinMaxScaler()
        
        self.is_fitted = False
    
    def preprocess_text_advanced(self, text: str) -> str:
        """
        Advanced text preprocessing for TF-IDF.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned and preprocessed text
        """
        if pd.isna(text) or not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep financial terms
        text = re.sub(r'[^\w\s\-\.\,\%\$]', ' ', text)
        
        # Normalize financial amounts (keep the concept, remove specific values)
        text = re.sub(r'gh[sc]?\s*\d+[\d,\.\s]*', 'currency_amount', text)
        text = re.sub(r'\$\s*\d+[\d,\.\s]*', 'dollar_amount', text)
        text = re.sub(r'\d+\.\d{2}%', 'percentage_value', text)
        text = re.sub(r'\d+[,\d]*\.\d+', 'numeric_value', text)
        text = re.sub(r'\d{4}[-/]\d{2}[-/]\d{2}', 'date_value', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_domain_features(self, text: str) -> Dict[str, float]:
        """
        Extract domain-specific features from audit text.
        
        Args:
            text: Preprocessed text
            
        Returns:
            Dictionary of domain features
        """
        if not text:
            return {f'domain_{i}': 0.0 for i in range(20)}
        
        features = {}
        
        # Financial irregularity indicators
        financial_keywords = [
            'fraud', 'embezzlement', 'misappropriation', 'unauthorized',
            'irregularity', 'discrepancy', 'overpayment', 'shortfall',
            'unaccounted', 'missing', 'ghost worker', 'phantom'
        ]
        
        # Procedural violation indicators  
        procedural_keywords = [
            'non-compliance', 'violation', 'breach', 'procurement',
            'tender', 'contract', 'bid rigging', 'sole source',
            'internal control', 'weak control', 'documentation'
        ]
        
        # Recovery and penalty indicators
        recovery_keywords = [
            'recover', 'recovery', 'refund', 'repay', 'surcharge',
            'penalty', 'fine', 'corrective action', 'recommendation'
        ]
        
        # Asset and property indicators
        asset_keywords = [
            'asset', 'property', 'equipment', 'vehicle', 'disposal',
            'write-off', 'depreciation', 'maintenance', 'upgrade'
        ]
        
        # Count keyword occurrences
        features['domain_financial_score'] = sum(1 for kw in financial_keywords if kw in text)
        features['domain_procedural_score'] = sum(1 for kw in procedural_keywords if kw in text)
        features['domain_recovery_score'] = sum(1 for kw in recovery_keywords if kw in text)
        features['domain_asset_score'] = sum(1 for kw in asset_keywords if kw in text)
        
        # Text statistics
        features['domain_text_length'] = len(text)
        features['domain_word_count'] = len(text.split())
        features['domain_sentence_count'] = len([s for s in text.split('.') if s.strip()])
        features['domain_avg_word_length'] = np.mean([len(w) for w in text.split()]) if text.split() else 0
        
        # Financial amount mentions
        features['domain_currency_mentions'] = text.count('currency_amount') + text.count('dollar_amount')
        features['domain_numeric_mentions'] = text.count('numeric_value')
        features['domain_percentage_mentions'] = text.count('percentage_value')
        features['domain_date_mentions'] = text.count('date_value')
        
        # Severity indicators
        high_severity = ['fraud', 'embezzlement', 'ghost worker', 'missing']
        medium_severity = ['irregularity', 'discrepancy', 'non-compliance', 'overcharge']
        
        features['domain_high_severity_score'] = sum(1 for kw in high_severity if kw in text)
        features['domain_medium_severity_score'] = sum(1 for kw in medium_severity if kw in text)
        
        # Evidence strength
        features['domain_evidence_strength'] = (
            features['domain_financial_score'] * 3 +
            features['domain_procedural_score'] * 2 +
            features['domain_recovery_score'] * 1
        )
        
        # Compliance risk
        features['domain_compliance_risk'] = (
            features['domain_high_severity_score'] * 3 +
            features['domain_medium_severity_score'] * 2
        )
        
        # Text complexity
        features['domain_complexity_score'] = (
            features['domain_word_count'] / 100 +
            features['domain_sentence_count'] / 20 +
            features['domain_avg_word_length'] / 10
        )
        
        # Pad to ensure consistent feature count
        for i in range(len(features), 20):
            features[f'domain_{i}'] = 0.0
        
        return features
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit TF-IDF vectorizer and transform texts to features.
        
        Args:
            texts: List of text documents
            
        Returns:
            Feature matrix with TF-IDF and domain features
        """
        # Preprocess texts
        processed_texts = [self.preprocess_text_advanced(text) for text in texts]
        
        # Fit and transform TF-IDF
        tfidf_features = self.tfidf_vectorizer.fit_transform(processed_texts)
        
        # Apply SVD for dimensionality reduction (adaptive components)
        tfidf_dense = tfidf_features.toarray() if hasattr(tfidf_features, 'toarray') else tfidf_features
        n_features = tfidf_dense.shape[1]
        n_components = min(self.svd_components, n_features - 1, len(processed_texts) - 1)
        
        if n_components < 1:
            n_components = 1
        
        # Recreate SVD with adaptive components
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd_features = self.svd.fit_transform(tfidf_dense)
        
        # Scale SVD features
        svd_features = self.scaler.fit_transform(svd_features)
        
        # Extract domain features
        domain_features_list = []
        for text in processed_texts:
            domain_feat = self.extract_domain_features(text)
            domain_features_list.append(list(domain_feat.values()))
        
        domain_features = np.array(domain_features_list)
        
        # Combine all features
        combined_features = np.hstack([svd_features, domain_features])
        
        self.is_fitted = True
        # Use actual adaptive components, not fixed svd_components
        actual_svd_components = svd_features.shape[1]  
        self.feature_names_ = (
            [f'tfidf_svd_{i}' for i in range(actual_svd_components)] +
            [f'domain_{i}' for i in range(20)]
        )
        
        return combined_features
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform new texts using fitted vectorizer.
        
        Args:
            texts: List of text documents
            
        Returns:
            Feature matrix
        """
        if not self.is_fitted:
            raise ValueError("TF-IDF extractor must be fitted first")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text_advanced(text) for text in texts]
        
        # Transform TF-IDF
        tfidf_features = self.tfidf_vectorizer.transform(processed_texts)
        
        # Apply SVD
        tfidf_dense = tfidf_features.toarray() if hasattr(tfidf_features, 'toarray') else tfidf_features  
        svd_features = self.svd.transform(tfidf_dense)
        
        # Scale SVD features
        svd_features = self.scaler.transform(svd_features)
        
        # Extract domain features
        domain_features_list = []
        for text in processed_texts:
            domain_feat = self.extract_domain_features(text)
            domain_features_list.append(list(domain_feat.values()))
        
        domain_features = np.array(domain_features_list)
        
        # Combine all features
        combined_features = np.hstack([svd_features, domain_features])
        
        return combined_features
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        if not self.is_fitted:
            raise ValueError("TF-IDF extractor must be fitted first")
        return self.feature_names_
    
    def save(self, path: str):
        """Save fitted extractor to file."""
        if not self.is_fitted:
            raise ValueError("TF-IDF extractor must be fitted first")
        
        extractor_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'svd': self.svd,
            'scaler': self.scaler,
            'feature_names_': self.feature_names_,
            'config': {
                'max_features': self.max_features,
                'max_df': self.max_df,
                'min_df': self.min_df,
                'ngram_range': self.ngram_range,
                'svd_components': self.svd_components
            }
        }
        
        joblib.dump(extractor_data, path)
    
    @classmethod
    def load(cls, path: str):
        """Load fitted extractor from file."""
        data = joblib.load(path)
        
        # Create instance with saved config
        extractor = cls(**data['config'])
        
        # Restore fitted components
        extractor.tfidf_vectorizer = data['tfidf_vectorizer']
        extractor.svd = data['svd']
        extractor.scaler = data['scaler']
        extractor.feature_names_ = data['feature_names_']
        extractor.is_fitted = True
        
        return extractor

def create_text_features_pipeline(df: pd.DataFrame, text_column: str = 'audit_notes') -> pd.DataFrame:
    """
    Create enhanced dataset with TF-IDF text features.
    
    Args:
        df: Input DataFrame with text data
        text_column: Column name containing text data
        
    Returns:
        DataFrame with additional TF-IDF features
    """
    if text_column not in df.columns:
        print(f"Warning: {text_column} column not found. Creating dummy text features.")
        # Create dummy features if text column is missing
        for i in range(70):  # 50 SVD + 20 domain features
            if i < 50:
                df[f'tfidf_svd_{i}'] = 0.0
            else:
                df[f'domain_{i-50}'] = 0.0
        return df
    
    print(f"Creating TF-IDF features from {text_column} column...")
    
    # Initialize extractor
    extractor = TFIDFFeatureExtractor()
    
    # Get texts
    texts = df[text_column].fillna('').astype(str).tolist()
    
    # Create features
    text_features = extractor.fit_transform(texts)
    feature_names = extractor.get_feature_names()
    
    # Add features to dataframe with robust error checking
    if text_features.shape[1] != len(feature_names):
        raise ValueError(f"Feature array width ({text_features.shape[1]}) doesn't match feature names length ({len(feature_names)})")
    
    feature_df = pd.DataFrame(text_features, columns=feature_names, index=df.index)
    result_df = pd.concat([df, feature_df], axis=1)
    
    # Save extractor for later use
    Path('models').mkdir(exist_ok=True)
    extractor.save('models/text_feature_extractor.pkl')
    
    print(f"Created {len(feature_names)} TF-IDF features")
    print(f"Feature names: {feature_names[:10]}...")
    
    return result_df

def analyze_pdf_content(pdf_path: str, entity_names: List[str]) -> Dict:
    """
    Analyze PDF content and extract structured information.
    
    Args:
        pdf_path: Path to PDF file
        entity_names: List of entity names to search for
        
    Returns:
        Dictionary with analysis results
    """
    from labelling import AuditLabelGenerator
    
    # Initialize generator
    generator = AuditLabelGenerator()
    
    # Extract and preprocess text
    raw_text = generator.extract_text_from_pdf(pdf_path)
    clean_text = generator.preprocess_text(raw_text)
    
    # Extract entity mentions
    mentions = generator.extract_entity_mentions(clean_text, entity_names)
    
    # Detect irregularities
    irregularities = generator.detect_irregularities(mentions)
    
    # Create TF-IDF features
    extractor = TFIDFFeatureExtractor()
    
    if clean_text.strip():
        text_features = extractor.fit_transform([clean_text])
        tfidf_summary = {
            'feature_count': text_features.shape[1],
            'text_length': len(clean_text),
            'processed_successfully': True
        }
    else:
        tfidf_summary = {
            'feature_count': 0,
            'text_length': 0,
            'processed_successfully': False
        }
    
    return {
        'pdf_path': pdf_path,
        'raw_text_length': len(raw_text),
        'clean_text_length': len(clean_text),
        'entities_mentioned': len(mentions),
        'entities_with_irregularities': sum(1 for r in irregularities.values() if r['audit_flag'] == 1),
        'total_irregularity_score': sum(r['irregularity_score'] for r in irregularities.values()),
        'tfidf_features': tfidf_summary,
        'entity_analysis': irregularities,
        'sample_text': clean_text[:500] + "..." if len(clean_text) > 500 else clean_text
    }

if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "The audit revealed financial irregularities in procurement processes with unauthorized payments.",
        "Ghost workers were discovered in the payroll system leading to significant losses.",
        "Internal controls were weak and documentation was missing for several transactions.",
        "No irregularities were found and all procedures were followed correctly.",
        "Embezzlement was detected with fraudulent activities totaling GHS 2.5 million."
    ]
    
    extractor = TFIDFFeatureExtractor()
    features = extractor.fit_transform(sample_texts)
    
    print("TF-IDF Feature Extraction Test:")
    print(f"Feature matrix shape: {features.shape}")
    print(f"Feature names: {extractor.get_feature_names()[:10]}...")
    print(f"Sample features for first text: {features[0][:10]}")