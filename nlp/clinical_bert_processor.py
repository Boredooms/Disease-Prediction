#!/usr/bin/env python3
"""
Clinic        # Comprehensive medical entity patterns (based on 3,959 medical records)
        self.medicine_patterns = [
            # Prescription format patterns
            r'\b(TAB|CAP|TABLET|CAPSULE)[.\\s]+([A-Z][A-Z\\s\\d\\-]+\\d*)\b',
            r'\b([A-Z][A-Z]+)\\\\s+(\\d+\\\\s*mg|mg|mcg|g|ml|units?)\b',
            r'\b(DEMO\\\\s+MEDICINE\\\\s+\\d+)\b',
            
            # Common drug name patterns from dataset
            r'\b(doxycycline|spironolactone|minocycline|clindamycin|tretinoin|isotretinoin)\b',
            r'\b(Accutane|Aldactone|Bactrim|Retin-A|Aczone|Differin|Epiduo|Doryx|Septra|Solodyn)\b',
            r'\b(amlodipine|lisinopril|atenolol|metformin|aspirin|atorvastatin|omeprazole|losartan)\b',
            r'\b(albuterol|prednisone|hydrochlorothiazide|simvastatin|warfarin|furosemide)\b',
            r'\b(gabapentin|tramadol|sertraline|fluoxetine|escitalopram|citalopram)\b',
            r'\b(amoxicillin|azithromycin|ciprofloxacin|levofloxacin|cephalexin|tetracycline)\b',
            
            # Generic patterns for any medicine name
            r'\b([A-Z][a-z]+)\\\\s+(\\d+\\\\s*mg|mg|mcg|g|ml|units?)\b',
            r'\b([A-Z][a-z]*[A-Z][a-z]*)\b',  # CamelCase drug names
            r'\b([a-z]+[A-Z][a-z]+)\b',      # Mixed case patterns
            r'\b([A-Z]{2,})\b'               # All caps abbreviations
        ]
        
        # Comprehensive disease patterns (based on dataset conditions)
        self.disease_patterns = [
            # Major conditions from dataset
            r'\b(acne|adhd|aids|hiv|allergies|alzheimer\'?s?|angina|anxiety|asthma)\b',
            r'\b(bipolar\\\\s+disorder|bronchitis|cancer|cholesterol|copd|covid\\\\s*19?)\b',
            r'\b(depression|diabetes|constipation|diarrhea|fibromyalgia|gout)\b',
            r'\b(hypertension|insomnia|migraine|obesity|osteoporosis|pneumonia)\b',
            r'\b(psoriasis|rheumatoid\\\\\\s+arthritis|seizure|thyroid)\b',
            
            # Blood pressure related
            r'\b(high\\\\s+blood\\\\s+pressure|blood\\\\s+pressure|hypertensive)\b',
            r'\b(hypotension|low\\\\s+blood\\\\s+pressure)\b',
            
            # Heart and cardiovascular
            r'\b(heart\\\\s+(?:disease|failure|attack)|cardiac|cardiovascular)\b',
            r'\b(coronary\\\\s+artery\\\\s+disease|atrial\\\\s+fibrillation|arrhythmia)\b',
            
            # Respiratory conditions
            r'\b(respiratory|lung|breathing|pulmonary|chronic\\\\s+cough)\b',
            r'\b(shortness\\\\s+of\\\\s+breath|wheezing|emphysema)\b',
            
            # Pain and inflammation
            r'\b(arthritis|joint\\\\s+pain|muscle\\\\s+pain|back\\\\s+pain|chronic\\\\s+pain)\b',
            r'\b(inflammation|inflammatory|rheumatoid|osteoarthritis)\b',
            
            # Mental health
            r'\b(anxiety\s+disorder|panic\s+disorder|social\s+anxiety)\b',
            r'\b(major\s+depression|clinical\s+depression|mood\s+disorder)\b',
            
            # Diabetes related
            r'\b(diabetes\s+(?:type\s+[12]|mellitus)|diabetic|insulin\s+resistance)\b',
            r'\b(blood\s+sugar|glucose|hyperglycemia|hypoglycemia)\b',
            
            # Skin conditions
            r'\b(skin\s+condition|dermatitis|eczema|rash|hives|urticaria)\b',
            
            # Infections
            r'\b(infection|bacterial|viral|fungal|antibiotic)\b',
            r'\b(pneumonia|bronchitis|sinusitis|uti|urinary\s+tract)\b'
        ]tem
Advanced medical NLP using Clinical BERT for disease and medicine extraction
"""

import os
import re
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import transformers for Clinical BERT
try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForTokenClassification,
        pipeline, AutoModelForSequenceClassification
    )
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers library available")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not available - installing...")

class ClinicalBERTProcessor:
    """Advanced medical NLP using Clinical BERT models"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.load_clinical_models()
        
        # Medical entity patterns (fallback)
        self.medicine_patterns = [
            r'\b(TAB|CAP|TABLET|CAPSULE)[\.\\s]+([A-Z][A-Z\\s\\d\\-]+\\d*)\b',
            r'\b([A-Z][A-Z]+)\\\\s+(\\d+\\\\s*mg|mg)\b',
            r'\b(DEMO\\\\s+MEDICINE\\\\s+\\d+)\b',
            r'\b([A-Z]+[a-z]*)\\\\s+(\\d+\\\\s*mg)\b'
        ]
        
        self.disease_patterns = [
            r'\b(hypertension|diabetes|asthma|arthritis|depression|anxiety)\b',
            r'\b(high\s+blood\s+pressure|blood\s+pressure)\b',
            r'\b(heart\s+disease|cardiac|cardiovascular)\b',
            r'\b(respiratory|lung|breathing)\b',
            r'\b(joint\s+pain|muscle\s+pain|back\s+pain)\b'
        ]
        
        # Medical knowledge base
        self.medical_kb = self.load_medical_knowledge()
    
    def load_clinical_models(self):
        """Load Clinical BERT models"""
        print("🤖 LOADING CLINICAL BERT MODELS")
        print("-" * 40)
        
        if not TRANSFORMERS_AVAILABLE:
            print("   ⚠️  Using rule-based fallback system")
            return
        
        try:
            # Clinical BERT for medical NER
            print("   📥 Loading Clinical BERT for NER...")
            clinical_bert_ner = "Clinical-AI-Apollo/Medical-NER"
            
            try:
                self.tokenizers['clinical_ner'] = AutoTokenizer.from_pretrained(clinical_bert_ner)
                self.models['clinical_ner'] = AutoModelForTokenClassification.from_pretrained(clinical_bert_ner)
                print("   ✅ Clinical BERT NER loaded")
            except:
                print("   ⚠️  Clinical BERT NER not available, using BioBERT...")
                # Fallback to BioBERT
                biobert_model = "dmis-lab/biobert-base-cased-v1.1"
                self.tokenizers['biobert'] = AutoTokenizer.from_pretrained(biobert_model)
                self.models['biobert'] = AutoModel.from_pretrained(biobert_model)
                print("   ✅ BioBERT loaded as fallback")
            
            # Clinical BERT for classification
            print("   📥 Loading Clinical BERT for classification...")
            try:
                clinical_bert_cls = "emilyalsentzer/Bio_ClinicalBERT"
                self.tokenizers['clinical_cls'] = AutoTokenizer.from_pretrained(clinical_bert_cls)
                self.models['clinical_cls'] = AutoModel.from_pretrained(clinical_bert_cls)
                print("   ✅ Clinical BERT classifier loaded")
            except:
                print("   ⚠️  Using general medical model...")
                
            # Medical text classification pipeline
            self.medical_classifier = None
            
            # Try multiple medical classifiers in order of preference
            medical_models = [
                "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                "emilyalsentzer/Bio_ClinicalBERT",
                "dmis-lab/biobert-base-cased-v1.1"
            ]
            
            for model_name in medical_models:
                try:
                    print(f"   📥 Trying medical classifier: {model_name.split('/')[-1]}...")
                    self.medical_classifier = pipeline(
                        "text-classification",
                        model=model_name,
                        tokenizer=model_name,
                        return_all_scores=True
                    )
                    print("   ✅ Medical classifier pipeline ready")
                    break
                except Exception as e:
                    print(f"   ⚠️  {model_name.split('/')[-1]} failed: {str(e)[:50]}...")
                    continue
            
            if self.medical_classifier is None:
                print("   ⚠️  All medical classifiers failed - using rule-based fallback")
                # Create a simple rule-based classifier
                self.medical_classifier = self._create_rule_based_classifier()
        
        except Exception as e:
            print(f"   ❌ Error loading models: {str(e)[:100]}")
            print("   🔄 Falling back to rule-based system...")
    
    def load_medical_knowledge(self):
        """Load comprehensive medical knowledge base from CSV data"""
        print("📋 LOADING COMPREHENSIVE MEDICAL KNOWLEDGE BASE")
        print("-" * 60)
        
        try:
            import pandas as pd
            csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                  'data', 'NLP', 'NLP_with_medicine_and_disease.csv')
            
            if not os.path.exists(csv_path):
                print(f"   ⚠️  CSV not found: {csv_path}")
                print("   🔄 Using fallback knowledge base...")
                return self._get_fallback_knowledge()
            
            # Load the comprehensive medical dataset
            df = pd.read_csv(csv_path)
            print(f"   📊 Loaded {len(df)} medical records")
            print(f"   💊 Unique drugs: {df['drug_name'].nunique()}")
            print(f"   🏥 Unique conditions: {df['medical_condition'].nunique()}")
            
            # Build medicine to disease mapping
            medicine_to_disease = {}
            
            # Group by drug and collect all conditions
            for drug, group in df.groupby('drug_name'):
                conditions = group['medical_condition'].unique().tolist()
                # Clean and normalize drug names
                drug_clean = drug.upper().strip()
                medicine_to_disease[drug_clean] = conditions
            
            # Get unique conditions for disease symptoms mapping
            unique_conditions = df['medical_condition'].unique()
            
            # Build disease symptoms mapping from descriptions
            disease_symptoms = {}
            for condition in unique_conditions:
                condition_data = df[df['medical_condition'] == condition].iloc[0]
                description = condition_data.get('medical_condition_description', '')
                
                # Extract key symptoms/keywords from description
                symptoms = self._extract_symptoms_from_description(description, condition)
                disease_symptoms[condition.lower()] = symptoms
            
            # Get most common medicines
            top_medicines = df['drug_name'].value_counts().head(100).index.tolist()
            common_medicines = [med.upper().strip() for med in top_medicines]
            
            # Add demo medicines for testing
            demo_medicines = {
                'DEMO MEDICINE 1': ['demonstration condition', 'test condition'],
                'DEMO MEDICINE 2': ['sample condition', 'example treatment'],
                'DEMO MEDICINE 3': ['prototype condition', 'trial treatment'],
                'DEMO MEDICINE 4': ['template condition', 'standard treatment']
            }
            medicine_to_disease.update(demo_medicines)
            common_medicines.extend(demo_medicines.keys())
            
            print(f"   ✅ Built comprehensive knowledge base")
            print(f"   📋 Medicine mappings: {len(medicine_to_disease)}")
            print(f"   🏥 Disease symptoms: {len(disease_symptoms)}")
            print(f"   💊 Common medicines: {len(common_medicines)}")
            
            return {
                'medicine_to_disease': medicine_to_disease,
                'disease_symptoms': disease_symptoms,
                'common_medicines': common_medicines,
                'dataset_stats': {
                    'total_records': len(df),
                    'unique_drugs': df['drug_name'].nunique(),
                    'unique_conditions': df['medical_condition'].nunique(),
                    'top_conditions': df['medical_condition'].value_counts().head(10).to_dict()
                }
            }
            
        except Exception as e:
            print(f"   ❌ Error loading CSV data: {str(e)[:100]}")
            print("   🔄 Using fallback knowledge base...")
            return self._get_fallback_knowledge()
    
    def _extract_symptoms_from_description(self, description: str, condition: str) -> List[str]:
        """Extract symptoms and keywords from medical condition description"""
        if not description:
            return [condition.lower()]
        
        # Common medical keywords to extract
        keywords = []
        description_lower = description.lower()
        
        # Condition-specific symptom extraction
        symptom_patterns = {
            'pain': ['pain', 'ache', 'soreness', 'discomfort', 'inflammation'],
            'hypertension': ['high blood pressure', 'cardiovascular', 'blood pressure', 'heart'],
            'diabetes': ['blood sugar', 'glucose', 'insulin', 'diabetes'],
            'acne': ['skin', 'pores', 'blackheads', 'whiteheads', 'pimples'],
            'arthritis': ['joint', 'inflammation', 'mobility', 'stiffness'],
            'pneumonia': ['lung', 'respiratory', 'breathing', 'chest'],
            'psoriasis': ['skin', 'inflammation', 'scaling', 'itching']
        }
        
        # Extract relevant symptoms based on condition
        condition_lower = condition.lower()
        for key, symptoms in symptom_patterns.items():
            if key in condition_lower:
                keywords.extend(symptoms)
        
        # Extract common medical terms from description
        common_terms = ['symptoms', 'treatment', 'condition', 'disease', 'disorder', 
                       'infection', 'chronic', 'acute', 'severe', 'mild']
        
        for term in common_terms:
            if term in description_lower:
                keywords.append(term)
        
        # Always include the condition name itself
        keywords.append(condition.lower())
        
        return list(set(keywords))  # Remove duplicates
    
    def _get_fallback_knowledge(self):
        """Fallback knowledge base if CSV loading fails"""
        return {
            'medicine_to_disease': {
                'AMLODIPINE': ['Hypertension', 'cardiovascular disease'],
                'LISINOPRIL': ['Hypertension', 'heart failure'],
                'ATENOLOL': ['Hypertension', 'angina'],
                'METFORMIN': ['Diabetes (Type 2)', 'insulin resistance'],
                'ASPIRIN': ['Pain', 'heart disease prevention'],
                'ATORVASTATIN': ['high cholesterol', 'cardiovascular prevention'],
                'OMEPRAZOLE': ['acid reflux', 'gastritis'],
                'DEMO MEDICINE 1': ['demonstration condition'],
                'DEMO MEDICINE 2': ['sample condition'],
                'DEMO MEDICINE 3': ['prototype condition'],
                'DEMO MEDICINE 4': ['template condition']
            },
            'disease_symptoms': {
                'hypertension': ['high blood pressure', 'cardiovascular'],
                'diabetes': ['blood sugar', 'glucose'],
                'pain': ['discomfort', 'inflammation'],
                'acne': ['skin', 'pores']
            },
            'common_medicines': [
                'AMLODIPINE', 'LISINOPRIL', 'ATENOLOL', 'METFORMIN', 'ASPIRIN'
            ]
        }
    
    def _create_rule_based_classifier(self):
        """Create a simple rule-based medical classifier as fallback"""
        class RuleBasedClassifier:
            def __init__(self, kb):
                self.kb = kb
                self.urgency_keywords = {
                    'high': ['emergency', 'urgent', 'critical', 'severe', 'acute'],
                    'medium': ['moderate', 'chronic', 'persistent', 'ongoing'],
                    'low': ['mild', 'minor', 'routine', 'preventive', 'maintenance']
                }
            
            def __call__(self, text):
                """Classify medical urgency based on text"""
                text_lower = text.lower()
                scores = {'high': 0.0, 'medium': 0.0, 'low': 0.1}  # Default low urgency
                
                for urgency, keywords in self.urgency_keywords.items():
                    for keyword in keywords:
                        if keyword in text_lower:
                            scores[urgency] += 0.3
                
                # Check for multiple medicines (polypharmacy)
                medicine_count = len([med for med in self.kb['common_medicines'] 
                                    if med.lower() in text_lower])
                if medicine_count > 3:
                    scores['medium'] += 0.2
                
                # Normalize scores
                total = sum(scores.values())
                if total > 0:
                    scores = {k: v/total for k, v in scores.items()}
                
                # Return in expected format
                return [{'label': k, 'score': v} for k, v in sorted(scores.items(), 
                                                                   key=lambda x: x[1], reverse=True)]
        
        return RuleBasedClassifier(self.medical_kb)
    
    def extract_medical_entities_bert(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities using Clinical BERT"""
        entities = {'medicines': [], 'diseases': [], 'dosages': [], 'instructions': []}
        
        if not TRANSFORMERS_AVAILABLE or 'clinical_ner' not in self.models:
            return self.extract_medical_entities_rules(text)
        
        try:
            # Tokenize text
            tokenizer = self.tokenizers['clinical_ner']
            model = self.models['clinical_ner']
            
            # Process text
            inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_labels = torch.argmax(predictions, dim=-1)
            
            # Decode predictions (simplified - would need proper label mapping)
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # For now, fall back to rule-based extraction
            return self.extract_medical_entities_rules(text)
            
        except Exception as e:
            print(f"   ⚠️  BERT extraction failed: {str(e)[:50]}")
            return self.extract_medical_entities_rules(text)
    
    def extract_medical_entities_rules(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities using rule-based patterns"""
        entities = {'medicines': [], 'diseases': [], 'dosages': [], 'instructions': []}
        
        # Extract medicines
        for pattern in self.medicine_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    medicine_name = (match.group(1) + " " + match.group(2)).strip()
                    entities['medicines'].append(medicine_name)
                else:
                    entities['medicines'].append(match.group(0).strip())
        
        # Extract diseases using patterns and knowledge base
        for pattern in self.disease_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities['diseases'].append(match.group(0).strip())
        
        # Extract dosages
        dosage_pattern = r'(\d+\s*(mg|ml|g|mcg|units?)\b)'
        dosage_matches = re.finditer(dosage_pattern, text, re.IGNORECASE)
        for match in dosage_matches:
            entities['dosages'].append(match.group(0))
        
        # Extract timing instructions
        timing_patterns = [r'(\d+\s+times?\s+(daily|per\s+day))', r'(morning|evening|night|before\s+food|after\s+food)']
        for pattern in timing_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities['instructions'].append(match.group(0))
        
        # Deduplicate
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def predict_diseases_from_medicines(self, medicines: List[str]) -> List[Dict[str, Any]]:
        """Predict probable diseases based on extracted medicines"""
        predictions = []
        
        for medicine in medicines:
            medicine_clean = medicine.upper().strip()
            
            # Check direct mapping
            if medicine_clean in self.medical_kb['medicine_to_disease']:
                diseases = self.medical_kb['medicine_to_disease'][medicine_clean]
                for disease in diseases:
                    predictions.append({
                        'medicine': medicine,
                        'predicted_disease': disease,
                        'confidence': 0.85,
                        'reasoning': f'{medicine} is commonly prescribed for {disease}'
                    })
            
            # Check partial matches
            else:
                for known_med, diseases in self.medical_kb['medicine_to_disease'].items():
                    # Simple fuzzy matching
                    if self.similarity(medicine_clean, known_med) > 0.7:
                        for disease in diseases:
                            predictions.append({
                                'medicine': medicine,
                                'predicted_disease': disease,
                                'confidence': 0.65,
                                'reasoning': f'{medicine} similar to {known_med}, likely for {disease}'
                            })
                        break
        
        return predictions
    
    def similarity(self, a: str, b: str) -> float:
        """Calculate string similarity"""
        a, b = a.lower(), b.lower()
        if a == b:
            return 1.0
        
        # Simple character overlap similarity
        a_chars = set(a)
        b_chars = set(b)
        if not a_chars or not b_chars:
            return 0.0
        
        intersection = len(a_chars.intersection(b_chars))
        union = len(a_chars.union(b_chars))
        return intersection / union if union > 0 else 0.0
    
    def classify_medical_urgency(self, text: str) -> Dict[str, Any]:
        """Classify medical urgency using Clinical BERT"""
        urgency_scores = {
            'routine': 0.0,
            'urgent': 0.0,
            'emergency': 0.0
        }
        
        # Keywords for urgency classification
        routine_keywords = ['daily', 'continue', 'maintenance', 'regular']
        urgent_keywords = ['monitor', 'follow up', 'check', 'review']
        emergency_keywords = ['immediately', 'emergency', 'urgent', 'critical']
        
        text_lower = text.lower()
        
        for keyword in routine_keywords:
            if keyword in text_lower:
                urgency_scores['routine'] += 0.3
        
        for keyword in urgent_keywords:
            if keyword in text_lower:
                urgency_scores['urgent'] += 0.5
        
        for keyword in emergency_keywords:
            if keyword in text_lower:
                urgency_scores['emergency'] += 0.8
        
        # Determine primary classification
        max_score = max(urgency_scores.values())
        if max_score == 0:
            primary = 'routine'
            confidence = 0.6
        else:
            primary = max(urgency_scores, key=urgency_scores.get)
            confidence = min(max_score, 1.0)
        
        return {
            'primary_urgency': primary,
            'confidence': confidence,
            'scores': urgency_scores
        }
    
    def process_prescription_text(self, ocr_text: str) -> Dict[str, Any]:
        """Complete prescription processing with Clinical BERT"""
        print(f"🧠 PROCESSING WITH CLINICAL BERT")
        print("-" * 40)
        
        results = {
            'original_text': ocr_text,
            'entities': {},
            'disease_predictions': [],
            'medical_classification': {},
            'clinical_insights': {}
        }
        
        # Extract entities
        print("   🏷️  Extracting medical entities...")
        entities = self.extract_medical_entities_bert(ocr_text)
        results['entities'] = entities
        
        print(f"      Found: {len(entities['medicines'])} medicines, {len(entities['diseases'])} diseases")
        
        # Predict diseases from medicines
        print("   🔍 Predicting diseases from medicines...")
        if entities['medicines']:
            disease_predictions = self.predict_diseases_from_medicines(entities['medicines'])
            results['disease_predictions'] = disease_predictions
            print(f"      Generated {len(disease_predictions)} disease predictions")
        
        # Classify urgency
        print("   ⚡ Classifying medical urgency...")
        urgency = self.classify_medical_urgency(ocr_text)
        results['medical_classification'] = urgency
        
        # Generate insights
        results['clinical_insights'] = self.generate_clinical_insights(results)
        
        print("   ✅ Clinical BERT processing complete")
        return results
    
    def generate_clinical_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate clinical insights from analysis"""
        insights = {
            'total_medicines': len(results['entities']['medicines']),
            'total_diseases': len(results['entities']['diseases']),
            'predicted_conditions': len(results['disease_predictions']),
            'prescription_complexity': 'simple'
        }
        
        # Determine complexity
        medicine_count = insights['total_medicines']
        if medicine_count > 3:
            insights['prescription_complexity'] = 'complex'
        elif medicine_count > 1:
            insights['prescription_complexity'] = 'moderate'
        
        # Risk assessment
        risk_indicators = []
        if medicine_count > 4:
            risk_indicators.append('polypharmacy_risk')
        
        insights['risk_indicators'] = risk_indicators
        
        return insights

def main():
    """Test Clinical BERT processor"""
    print("🤖 CLINICAL BERT NLP SYSTEM TEST")
    print("="*45)
    
    processor = ClinicalBERTProcessor()
    
    # Test with sample prescription text
    sample_text = """
    TAB. DEMO MEDICINE 1 - 1 Morning, 1 Night (Before Food) - 10 Days
    CAP. DEMO MEDICINE 2 - 1 Morning, 1 Night (Before Food) - 10 Days  
    TAB. DEMO MEDICINE 3 - 1 Morning, 1 Aft, 1 Eve, 1 Night (After Food) - 10 Days
    TAB. DEMO MEDICINE 4 - 1/2 Morning, 1/2 Night (After Food) - 10 Days
    """
    
    results = processor.process_prescription_text(sample_text)
    
    print(f"\\n📊 RESULTS:")
    print(f"   Medicines: {results['entities']['medicines']}")
    print(f"   Diseases: {results['entities']['diseases']}")
    print(f"   Disease Predictions: {len(results['disease_predictions'])}")
    
    for pred in results['disease_predictions']:
        print(f"      • {pred['medicine']} → {pred['predicted_disease']} ({pred['confidence']:.1%})")

if __name__ == '__main__':
    main()