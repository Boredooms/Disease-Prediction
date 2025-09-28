#!/usr/bin/env python3
"""
Advanced Disease Predictor System
Integrates OCR, NLP, and Symptom Data for Comprehensive Disease Prediction
"""

import os
import json
import pandas as pd
import numpy as np
import pickle
import warnings
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import logging

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import RFE, SelectKBest, chi2

# Deep Learning imports (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Dropout, Input, concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

warnings.filterwarnings('ignore')

class AdvancedDiseasePredictor:
    """
    Advanced Disease Prediction System integrating multiple data sources
    """
    
    def __init__(self, data_dir: str = None):
        """Initialize the disease predictor"""
        self.data_dir = data_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Disease Predictor')
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.disease_mappings = {}
        self.symptoms_mapping = {}
        self.model_performance = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load and prepare data
        self.load_datasets()
        
    def load_datasets(self):
        """Load all datasets and prepare for training"""
        print("🏥 LOADING DISEASE PREDICTION DATASETS")
        print("="*60)
        
        try:
            # Load training data (symptom-based)
            training_path = os.path.join(self.data_dir, 'Training.csv')
            self.training_df = pd.read_csv(training_path)
            print(f"   ✅ Training data: {len(self.training_df)} samples")
            
            # Load prediction data 
            prediction_path = os.path.join(self.data_dir, 'prediction.csv')
            self.prediction_df = pd.read_csv(prediction_path)
            print(f"   ✅ Prediction data: {len(self.prediction_df)} samples")
            
            # Load disease-drug mappings
            disease_path = os.path.join(self.data_dir, 'Disease.json')
            with open(disease_path, 'r', encoding='utf-8') as f:
                self.disease_drug_data = json.load(f)
            print(f"   ✅ Disease-drug mappings: {len(self.disease_drug_data)} entries")
            
            # Process the datasets
            self.prepare_data()
            
        except Exception as e:
            self.logger.error(f"Error loading datasets: {str(e)}")
            raise
    
    def prepare_data(self):
        """Prepare and preprocess the data"""
        print("\\n🔧 PREPARING DATA FOR TRAINING")
        print("-" * 40)
        
        # Clean the training data first
        self.training_df = self.training_df.dropna(subset=['prognosis'])
        self.training_df = self.training_df[self.training_df['prognosis'] != '']
        
        # Clean prediction data
        if 'prognosis' in self.prediction_df.columns:
            self.prediction_df = self.prediction_df.dropna(subset=['prognosis'])
            self.prediction_df = self.prediction_df[self.prediction_df['prognosis'] != '']
            # Combine both datasets
            combined_df = pd.concat([self.training_df, self.prediction_df], ignore_index=True)
        else:
            # Use only training data if prediction data doesn't have prognosis
            combined_df = self.training_df.copy()
        
        print(f"   📊 Combined dataset: {len(combined_df)} samples")
        
        # Extract features and target
        target_col = 'prognosis'
        
        # Remove unnamed/empty columns
        columns_to_drop = [col for col in combined_df.columns if col.startswith('Unnamed') or col == target_col]
        X = combined_df.drop(columns_to_drop, axis=1)
        y = combined_df[target_col]
        
        print(f"   📋 Valid samples after cleaning: {len(y)}")
        print(f"   🏥 Unique diseases: {len(y.unique())}")
        print(f"   📋 Sample diseases: {y.unique()[:5].tolist()}")
        
        # Store feature names
        self.feature_names = list(X.columns)
        print(f"   📊 Features: {len(self.feature_names)}")
        print(f"   🏥 Diseases: {len(y.unique())} unique conditions")
        
        # Encode diseases
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Create disease mappings
        self.disease_mappings = {
            'label_to_disease': dict(zip(range(len(self.label_encoder.classes_)), self.label_encoder.classes_)),
            'disease_to_label': dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))
        }
        
        # Create symptom mapping for easy lookup
        self.symptoms_mapping = {symptom: idx for idx, symptom in enumerate(self.feature_names)}
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"   ✅ Training samples: {len(self.X_train)}")
        print(f"   ✅ Testing samples: {len(self.X_test)}")
        
        # Process disease-drug mappings
        self.process_disease_drug_mappings()
        
    def process_disease_drug_mappings(self):
        """Process disease-drug mappings from JSON data"""
        print("\\n💊 PROCESSING DISEASE-DRUG MAPPINGS")
        print("-" * 40)
        
        self.disease_drug_mappings = {}
        drug_recommendations = {}
        
        for entry in self.disease_drug_data:
            if 'input' in entry and 'output' in entry:
                # Extract disease name from input
                input_text = entry['input'].lower()
                if 'disease:' in input_text:
                    disease_part = input_text.split('disease:')[1].split('.')[0].strip()
                    drug = entry['output'].strip()
                    
                    if disease_part not in drug_recommendations:
                        drug_recommendations[disease_part] = []
                    drug_recommendations[disease_part].append(drug)
        
        # Clean and standardize disease names
        for disease, drugs in drug_recommendations.items():
            # Try to match with our disease labels
            disease_clean = disease.title()
            best_match = None
            
            for known_disease in self.label_encoder.classes_:
                if disease_clean.lower() in known_disease.lower() or known_disease.lower() in disease_clean.lower():
                    best_match = known_disease
                    break
            
            if best_match:
                self.disease_drug_mappings[best_match] = list(set(drugs))
            else:
                self.disease_drug_mappings[disease_clean] = list(set(drugs))
        
        print(f"   ✅ Disease-drug mappings: {len(self.disease_drug_mappings)}")
        
    def train_models(self):
        """Train multiple machine learning models"""
        print("\\n🤖 TRAINING DISEASE PREDICTION MODELS")
        print("="*50)
        
        # Define models to train
        model_configs = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'Naive Bayes': GaussianNB(),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        # Train each model
        for name, model in model_configs.items():
            print(f"\\n🔧 Training {name}...")
            
            try:
                # Train model
                model.fit(self.X_train, self.y_train)
                
                # Make predictions
                y_pred = model.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
                
                # Store model and performance
                self.models[name] = model
                self.model_performance[name] = {
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                print(f"   ✅ {name}: {accuracy:.3f} accuracy (CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f})")
                
            except Exception as e:
                print(f"   ❌ {name} failed: {str(e)}")
        
        # Create ensemble model
        self.create_ensemble_model()
        
        # Train deep learning model if available
        if TENSORFLOW_AVAILABLE:
            self.train_deep_learning_model()
    
    def create_ensemble_model(self):
        """Create an ensemble of the best models"""
        print("\\n🎯 CREATING ENSEMBLE MODEL")
        print("-" * 30)
        
        # Select top 3 models based on accuracy
        top_models = sorted(self.model_performance.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:3]
        
        estimators = []
        for name, _ in top_models:
            estimators.append((name.lower().replace(' ', '_'), self.models[name]))
        
        # Create voting classifier
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        ensemble.fit(self.X_train, self.y_train)
        
        # Evaluate ensemble
        y_pred = ensemble.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        self.models['Ensemble'] = ensemble
        self.model_performance['Ensemble'] = {'accuracy': accuracy}
        
        print(f"   ✅ Ensemble Model: {accuracy:.3f} accuracy")
        print(f"   📊 Combined models: {[name for name, _ in top_models]}")
    
    def train_deep_learning_model(self):
        """Train a deep learning model using TensorFlow"""
        print("\\n🧠 TRAINING DEEP LEARNING MODEL")
        print("-" * 35)
        
        try:
            # Build neural network
            model = Sequential([
                Dense(256, activation='relu', input_shape=(len(self.feature_names),)),
                Dropout(0.3),
                Dense(128, activation='relu'),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(len(self.label_encoder.classes_), activation='softmax')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ]
            
            # Train model
            history = model.fit(
                self.X_train, self.y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate
            test_loss, test_accuracy = model.evaluate(self.X_test, self.y_test, verbose=0)
            
            self.models['Deep Learning'] = model
            self.model_performance['Deep Learning'] = {'accuracy': test_accuracy}
            
            print(f"   ✅ Deep Learning Model: {test_accuracy:.3f} accuracy")
            
        except Exception as e:
            print(f"   ❌ Deep Learning training failed: {str(e)}")
    
    def predict_disease_from_symptoms(self, symptoms: Dict[str, int], model_name: str = 'Ensemble') -> Dict[str, Any]:
        """
        Predict disease based on symptoms
        
        Args:
            symptoms: Dictionary of symptom_name -> presence (0 or 1)
            model_name: Name of model to use for prediction
            
        Returns:
            Dictionary with prediction results
        """
        # Prepare feature vector
        feature_vector = np.zeros(len(self.feature_names))
        
        for symptom, value in symptoms.items():
            if symptom in self.symptoms_mapping:
                feature_vector[self.symptoms_mapping[symptom]] = value
        
        # Scale features
        feature_vector_scaled = self.scaler.transform([feature_vector])
        
        # Get model
        model = self.models.get(model_name, self.models['Ensemble'])
        
        # Make prediction
        if model_name == 'Deep Learning' and TENSORFLOW_AVAILABLE:
            prediction_proba = model.predict(feature_vector_scaled, verbose=0)[0]
            prediction = np.argmax(prediction_proba)
        else:
            prediction = model.predict(feature_vector_scaled)[0]
            prediction_proba = model.predict_proba(feature_vector_scaled)[0]
        
        # Get disease name
        predicted_disease = self.disease_mappings['label_to_disease'][prediction]
        confidence = float(np.max(prediction_proba))
        
        # Get top 3 predictions
        top_indices = np.argsort(prediction_proba)[-3:][::-1]
        top_predictions = []
        
        for idx in top_indices:
            disease = self.disease_mappings['label_to_disease'][idx]
            prob = float(prediction_proba[idx])
            top_predictions.append({'disease': disease, 'probability': prob})
        
        return {
            'predicted_disease': predicted_disease,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'model_used': model_name
        }
    
    def predict_from_ocr_nlp(self, ocr_results: Dict, nlp_results: Dict) -> Dict[str, Any]:
        """
        Predict disease based on OCR and NLP results
        
        Args:
            ocr_results: Results from OCR processing
            nlp_results: Results from NLP processing
            
        Returns:
            Comprehensive prediction results
        """
        print("\\n🔍 PREDICTING DISEASE FROM OCR + NLP")
        print("-" * 45)
        
        # Extract information from OCR and NLP
        extracted_info = self.extract_medical_info(ocr_results, nlp_results)
        
        # Convert to symptom vector
        symptom_vector = self.medical_info_to_symptoms(extracted_info)
        
        # Make prediction using ensemble
        prediction_result = self.predict_disease_from_symptoms(symptom_vector, 'Ensemble')
        
        # Get drug recommendations
        drug_recommendations = self.get_drug_recommendations(prediction_result['predicted_disease'])
        
        # Combine results
        comprehensive_result = {
            **prediction_result,
            'extracted_medical_info': extracted_info,
            'symptom_analysis': symptom_vector,
            'drug_recommendations': drug_recommendations,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        print(f"   🏥 Predicted Disease: {prediction_result['predicted_disease']}")
        print(f"   🎯 Confidence: {prediction_result['confidence']:.2%}")
        print(f"   💊 Drug Recommendations: {len(drug_recommendations)} found")
        
        return comprehensive_result
    
    def extract_medical_info(self, ocr_results: Dict, nlp_results: Dict) -> Dict[str, Any]:
        """Extract medical information from OCR and NLP results"""
        medical_info = {
            'symptoms': [],
            'medicines': [],
            'conditions': [],
            'vital_signs': {},
            'patient_info': {}
        }
        
        # Extract from OCR results
        if 'medicines' in ocr_results:
            if isinstance(ocr_results['medicines'], list):
                medical_info['medicines'].extend(ocr_results['medicines'])
            elif isinstance(ocr_results['medicines'], dict):
                medical_info['medicines'].extend(list(ocr_results['medicines'].keys()))
        
        # Extract from NLP results
        if 'entities' in nlp_results:
            entities = nlp_results['entities']
            if 'medicines' in entities:
                medical_info['medicines'].extend(entities['medicines'])
            if 'diseases' in entities:
                medical_info['conditions'].extend(entities['diseases'])
        
        # Extract disease predictions from NLP
        if 'disease_predictions' in nlp_results:
            for pred in nlp_results['disease_predictions']:
                if 'predicted_disease' in pred:
                    medical_info['conditions'].append(pred['predicted_disease'])
        
        # Clean and deduplicate
        medical_info['medicines'] = list(set([m.upper().strip() for m in medical_info['medicines'] if m]))
        medical_info['conditions'] = list(set([c.lower().strip() for c in medical_info['conditions'] if c]))
        
        return medical_info
    
    def medical_info_to_symptoms(self, medical_info: Dict) -> Dict[str, int]:
        """Convert medical information to symptom vector"""
        symptom_vector = {}
        
        # Map conditions to symptoms
        condition_symptom_mapping = {
            'hypertension': ['high_fever', 'headache', 'dizziness'],
            'diabetes': ['excessive_hunger', 'fatigue', 'weight_loss', 'polyuria'],
            'angina': ['chest_pain', 'breathlessness'],
            'infection': ['high_fever', 'fatigue'],
            'pain': ['muscle_pain', 'joint_pain'],
            'gastrointestinal': ['stomach_pain', 'nausea', 'vomiting']
        }
        
        # Map medicines to likely conditions/symptoms
        medicine_symptom_mapping = {
            'AMLODIPINE': ['chest_pain', 'headache'],
            'ASPIRIN': ['headache', 'muscle_pain'],
            'METFORMIN': ['excessive_hunger', 'weight_loss']
        }
        
        # Process conditions
        for condition in medical_info['conditions']:
            for key, symptoms in condition_symptom_mapping.items():
                if key in condition:
                    for symptom in symptoms:
                        symptom_vector[symptom] = 1
        
        # Process medicines
        for medicine in medical_info['medicines']:
            if medicine in medicine_symptom_mapping:
                for symptom in medicine_symptom_mapping[medicine]:
                    symptom_vector[symptom] = 1
        
        return symptom_vector
    
    def get_drug_recommendations(self, disease: str) -> List[str]:
        """Get drug recommendations for a disease"""
        # Direct lookup
        if disease in self.disease_drug_mappings:
            return self.disease_drug_mappings[disease]
        
        # Fuzzy matching
        disease_lower = disease.lower()
        for mapped_disease, drugs in self.disease_drug_mappings.items():
            if disease_lower in mapped_disease.lower() or mapped_disease.lower() in disease_lower:
                return drugs
        
        return []
    
    def save_models(self, save_dir: str = None):
        """Save trained models"""
        if not save_dir:
            save_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\\n💾 SAVING MODELS TO: {save_dir}")
        print("-" * 40)
        
        # Save sklearn models
        for name, model in self.models.items():
            if name != 'Deep Learning':
                model_path = os.path.join(save_dir, f'{name.lower().replace(" ", "_")}_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"   ✅ Saved {name}")
        
        # Save TensorFlow model
        if 'Deep Learning' in self.models and TENSORFLOW_AVAILABLE:
            dl_path = os.path.join(save_dir, 'deep_learning_model.keras')
            self.models['Deep Learning'].save(dl_path)
            print(f"   ✅ Saved Deep Learning Model")
        
        # Save preprocessors and mappings
        metadata = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'disease_mappings': self.disease_mappings,
            'symptoms_mapping': self.symptoms_mapping,
            'disease_drug_mappings': self.disease_drug_mappings,
            'model_performance': self.model_performance
        }
        
        metadata_path = os.path.join(save_dir, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"   ✅ Saved metadata and preprocessors")
    
    def load_models(self, save_dir: str = None):
        """Load trained models"""
        if not save_dir:
            save_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
        
        if not os.path.exists(save_dir):
            print(f"❌ Model directory not found: {save_dir}")
            return False
        
        print(f"\n📂 LOADING MODELS FROM: {save_dir}")
        print("-" * 40)
        
        try:
            # Load metadata first
            metadata_path = os.path.join(save_dir, 'metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.scaler = metadata.get('scaler')
                self.label_encoder = metadata.get('label_encoder')
                self.feature_names = metadata.get('feature_names', [])
                self.disease_mappings = metadata.get('disease_mappings', {})
                self.symptoms_mapping = metadata.get('symptoms_mapping', {})
                self.disease_drug_mappings = metadata.get('disease_drug_mappings', {})
                self.model_performance = metadata.get('model_performance', {})
                print(f"   ✅ Loaded metadata and preprocessors")
            
            # Load sklearn models
            model_files = {
                'Random Forest': 'random_forest_model.pkl',
                'SVM': 'svm_model.pkl',
                'Gradient Boosting': 'gradient_boosting_model.pkl',
                'Neural Network': 'neural_network_model.pkl',
                'Ensemble': 'ensemble_model.pkl'
            }
            
            for name, filename in model_files.items():
                model_path = os.path.join(save_dir, filename)
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[name] = pickle.load(f)
                    print(f"   ✅ Loaded {name}")
            
            # Load TensorFlow model
            if TENSORFLOW_AVAILABLE:
                dl_path = os.path.join(save_dir, 'deep_learning_model.keras')
                if os.path.exists(dl_path):
                    import tensorflow as tf
                    self.models['Deep Learning'] = tf.keras.models.load_model(dl_path)
                    print(f"   ✅ Loaded Deep Learning Model")
            
            print(f"\n🎉 Successfully loaded {len(self.models)} models!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            return False
    
    def generate_comprehensive_report(self, prediction_result: Dict) -> str:
        """Generate a comprehensive medical report"""
        report_lines = [
            "ADVANCED DISEASE PREDICTION REPORT",
            "="*50,
            f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Analysis ID: DP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "",
            "PREDICTION SUMMARY:",
            "-"*20,
            f"Primary Diagnosis: {prediction_result['predicted_disease']}",
            f"Confidence Level: {prediction_result['confidence']:.1%}",
            f"Model Used: {prediction_result['model_used']}",
            "",
            "TOP DIFFERENTIAL DIAGNOSES:",
            "-"*30
        ]
        
        for i, pred in enumerate(prediction_result['top_predictions'], 1):
            report_lines.append(f"{i}. {pred['disease']} ({pred['probability']:.1%})")
        
        if 'drug_recommendations' in prediction_result and prediction_result['drug_recommendations']:
            report_lines.extend([
                "",
                "RECOMMENDED MEDICATIONS:",
                "-"*25
            ])
            for drug in prediction_result['drug_recommendations'][:5]:
                report_lines.append(f"• {drug.title()}")
        
        if 'extracted_medical_info' in prediction_result:
            info = prediction_result['extracted_medical_info']
            if info['medicines']:
                report_lines.extend([
                    "",
                    "CURRENT MEDICATIONS:",
                    "-"*20
                ])
                for med in info['medicines'][:5]:
                    report_lines.append(f"• {med}")
        
        report_lines.extend([
            "",
            "CLINICAL RECOMMENDATIONS:",
            "-"*25,
            "• Consult with healthcare provider for proper diagnosis",
            "• Follow prescribed medication regimen",
            "• Monitor symptoms and report changes",
            "• Schedule follow-up appointments as recommended",
            "",
            "DISCLAIMER:",
            "-"*10,
            "This analysis is for informational purposes only.",
            "Always consult healthcare professionals for medical advice.",
            "",
            "="*50
        ])
        
        return "\\n".join(report_lines)

def main():
    """Test the Advanced Disease Predictor"""
    print("🏥 ADVANCED DISEASE PREDICTOR SYSTEM")
    print("="*50)
    
    # Initialize predictor
    predictor = AdvancedDiseasePredictor()
    
    # Train models
    predictor.train_models()
    
    # Save models
    predictor.save_models()
    
    # Test with sample symptoms
    test_symptoms = {
        'itching': 1,
        'skin_rash': 1,
        'nodal_skin_eruptions': 1
    }
    
    print("\\n🧪 TESTING WITH SAMPLE SYMPTOMS:")
    print(f"   Symptoms: {list(test_symptoms.keys())}")
    
    result = predictor.predict_disease_from_symptoms(test_symptoms)
    print(f"\\n📊 PREDICTION RESULT:")
    print(f"   Disease: {result['predicted_disease']}")
    print(f"   Confidence: {result['confidence']:.2%}")
    
    # Test with OCR/NLP integration
    sample_ocr = {'medicines': ['AMLODIPINE', 'ASPIRIN']}
    sample_nlp = {'entities': {'diseases': ['hypertension']}, 'disease_predictions': []}
    
    comprehensive_result = predictor.predict_from_ocr_nlp(sample_ocr, sample_nlp)
    report = predictor.generate_comprehensive_report(comprehensive_result)
    
    print("\\n📋 COMPREHENSIVE ANALYSIS REPORT:")
    print("-" * 40)
    print(report[:500] + "..." if len(report) > 500 else report)

if __name__ == '__main__':
    main()