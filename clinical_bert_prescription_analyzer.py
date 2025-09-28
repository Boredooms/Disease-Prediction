#!/usr/bin/env python3
"""
Complete Clinical BERT Prescription Analysis System
OCR → Clinical BERT NLP → Disease Prediction
"""

import sys
import os
import numpy as np
import pandas as pd
sys.path.append("c:\\Just Ai\\EndToEndDiseasePredictor")

# Import OCR system
from ocr.direct_prescription_reader import DirectPrescriptionReader
from ocr.smart_medicine_enhancer import SmartMedicineEnhancer

# Import Clinical BERT NLP
from nlp.clinical_bert_processor import ClinicalBERTProcessor

# Import Disease Predictor
from disease_predictor.disease_prediction_system import AdvancedDiseasePredictor

class ClinicalPrescriptionAnalyzer:
    """Complete prescription analysis with Clinical BERT + Disease Prediction"""
    
    def __init__(self):
        print("🏥 INITIALIZING COMPREHENSIVE MEDICAL ANALYSIS SYSTEM")
        print("="*70)
        
        # Initialize components
        self.ocr_reader = DirectPrescriptionReader()
        self.medicine_enhancer = SmartMedicineEnhancer()
        self.clinical_bert = ClinicalBERTProcessor()
        
        # Initialize Disease Predictor
        print("🧠 LOADING DISEASE PREDICTION MODELS")
        print("-" * 40)
        try:
            self.disease_predictor = AdvancedDiseasePredictor()
            self.disease_predictor.load_models()
            print("   ✅ Disease prediction models loaded")
        except Exception as e:
            print(f"   ⚠️  Disease predictor initialization failed: {str(e)}")
            print("   🔄 Training models...")
            try:
                self.disease_predictor = AdvancedDiseasePredictor()
                # Train models if not available
                data_path = os.path.join(os.path.dirname(__file__), 'data', 'Disease Predictor')
                if os.path.exists(os.path.join(data_path, 'Training.csv')):
                    self.disease_predictor.prepare_and_train_models()
                    print("   ✅ Disease prediction models trained and loaded")
                else:
                    print("   ⚠️  Training data not found - using Clinical BERT only")
                    self.disease_predictor = None
            except Exception as e2:
                print(f"   ❌ Disease predictor failed: {str(e2)}")
                self.disease_predictor = None
        
        print("   ✅ OCR System initialized")
        print("   ✅ Medicine Enhancer ready")
        print("   ✅ Clinical BERT processor loaded")
        if self.disease_predictor:
            print("   ✅ Disease Predictor ready")
        
        # Load symptom mappings for disease prediction
        self.symptom_mappings = self.load_symptom_mappings()
    
    def load_symptom_mappings(self):
        """Load comprehensive medicine-to-symptom mappings"""
        return {
            # Cardiovascular medicines
            'AMLODIPINE': ['chest_pain', 'high_blood_pressure', 'swelling_of_legs', 'irregular_heartbeat'],
            'ATENOLOL': ['chest_pain', 'high_blood_pressure', 'irregular_heartbeat', 'anxiety'],
            'LISINOPRIL': ['high_blood_pressure', 'chest_pain', 'fatigue', 'dizziness'],
            'METOPROLOL': ['chest_pain', 'high_blood_pressure', 'irregular_heartbeat'],
            'LOSARTAN': ['high_blood_pressure', 'chest_pain', 'dizziness'],
            
            # Gastrointestinal medicines
            'OMEPRAZOLE': ['stomach_pain', 'acidity', 'heartburn', 'nausea'],
            'RANITIDINE': ['stomach_pain', 'acidity', 'heartburn', 'vomiting'],
            'DOMPERIDONE': ['nausea', 'vomiting', 'stomach_pain', 'loss_of_appetite'],
            'PANTOPRAZOLE': ['stomach_pain', 'acidity', 'heartburn'],
            
            # Diabetes medicines
            'METFORMIN': ['increased_urination', 'increased_thirst', 'fatigue', 'weight_loss'],
            'INSULIN': ['increased_urination', 'increased_thirst', 'fatigue', 'blurred_vision'],
            'GLIMEPIRIDE': ['increased_urination', 'increased_thirst', 'fatigue'],
            
            # Respiratory medicines
            'SALBUTAMOL': ['breathlessness', 'cough', 'chest_tightness', 'wheezing'],
            'PREDNISOLONE': ['breathlessness', 'cough', 'chest_tightness'],
            'THEOPHYLLINE': ['breathlessness', 'cough', 'chest_tightness'],
            
            # Pain and inflammation
            'DICLOFENAC': ['joint_pain', 'muscle_pain', 'back_pain', 'swelling_joints'],
            'PARACETAMOL': ['headache', 'fever', 'body_ache', 'joint_pain'],
            'IBUPROFEN': ['joint_pain', 'muscle_pain', 'headache', 'fever'],
            
            # Antibiotics
            'AMOXICILLIN': ['fever', 'cough', 'sore_throat', 'fatigue'],
            'AZITHROMYCIN': ['fever', 'cough', 'sore_throat', 'chest_pain'],
            
            # General medicines
            'DEMO MEDICINE': ['chest_pain', 'high_blood_pressure', 'fatigue']
        }
    
    def extract_symptoms_from_medicines(self, medicines):
        """Extract symptoms based on identified medicines"""
        symptoms = set()
        
        for medicine in medicines:
            medicine_upper = medicine.upper()
            
            # Direct match
            if medicine_upper in self.symptom_mappings:
                symptoms.update(self.symptom_mappings[medicine_upper])
                continue
            
            # Partial match for medicines with dosage info
            for known_medicine, mapped_symptoms in self.symptom_mappings.items():
                if known_medicine in medicine_upper:
                    symptoms.update(mapped_symptoms)
                    break
        
        return list(symptoms)
    
    def convert_to_symptom_vector(self, symptoms):
        """Convert symptoms to 132-feature vector for ML models"""
        # Standard symptom list (first 132 symptoms from training data)
        standard_symptoms = [
            'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain',
            'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination',
            'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy',
            'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating',
            'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
            'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
            'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision',
            'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
            'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain',
            'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid',
            'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech',
            'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements',
            'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
            'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability',
            'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches',
            'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration',
            'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
            'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf',
            'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting',
            'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze',
            'high_blood_pressure', 'irregular_heartbeat', 'heartburn', 'increased_urination', 'increased_thirst', 'blurred_vision',
            'chest_tightness', 'wheezing', 'sore_throat', 'body_ache', 'fever', 'swelling_of_legs'
        ]
        
        # Create binary vector
        symptom_vector = [0] * len(standard_symptoms)
        
        for i, std_symptom in enumerate(standard_symptoms):
            if std_symptom in symptoms:
                symptom_vector[i] = 1
        
        return symptom_vector
    
    def analyze_prescription_image(self, image_path):
        """Complete prescription analysis pipeline"""
        print(f"\n🔍 ANALYZING PRESCRIPTION: {os.path.basename(image_path)}")
        print("="*60)
        
        results = {
            'image_path': image_path,
            'timestamp': pd.Timestamp.now().isoformat(),
        }
        
        # Step 1: OCR Processing
        print("\n📄 STEP 1: OCR PROCESSING")
        print("-" * 25)
        try:
            ocr_results = self.ocr_reader.read_prescription(image_path)
            results['ocr_results'] = ocr_results
            
            extracted_text = ocr_results.get('extracted_text', '')
            confidence = ocr_results.get('confidence', 0)
            
            print(f"   ✅ OCR complete - Confidence: {confidence:.1%}")
            print(f"   📝 Text length: {len(extracted_text)} characters")
            
            if extracted_text:
                print(f"   📄 Sample text: {extracted_text[:100]}...")
            else:
                print("   ⚠️  No text extracted")
                
        except Exception as e:
            print(f"   ❌ OCR failed: {str(e)}")
            extracted_text = ""
            
        # Step 2: Medicine Enhancement
        print("\n💊 STEP 2: MEDICINE ENHANCEMENT")
        print("-" * 30)
        try:
            enhanced_results = self.medicine_enhancer.smart_ocr_with_enhancement(image_path)
            results['enhanced_results'] = enhanced_results
            
            # Combine enhanced text
            if enhanced_results and 'medicines' in enhanced_results:
                enhanced_text = ""
                for med in enhanced_results['medicines']:
                    enhanced_text += f"{med.get('name', '')} {med.get('dosage', '')} {med.get('timing', '')} {med.get('duration', '')}\n"
                if enhanced_text.strip():
                    extracted_text = enhanced_text
                    
            print(f"   ✅ Enhancement complete")
        except Exception as e:
            print(f"   ⚠️  Enhancement failed: {str(e)}")
            
        # Step 3: Clinical BERT Analysis
        print("\n🤖 STEP 3: CLINICAL BERT ANALYSIS")
        print("-" * 35)
        if extracted_text.strip():
            try:
                clinical_analysis = self.clinical_bert.process_prescription_text(extracted_text)
                results['clinical_analysis'] = clinical_analysis
                
                print(f"   ✅ Clinical BERT analysis complete")
            except Exception as e:
                print(f"   ❌ Clinical BERT failed: {str(e)}")
                clinical_analysis = {}
        else:
            print("   ⚠️  No text available for Clinical BERT analysis")
            clinical_analysis = {}
            
        # Step 4: Disease Prediction using ML Models
        print("\n🔬 STEP 4: ADVANCED DISEASE PREDICTION")
        print("-" * 40)
        try:
            if self.disease_predictor:
                # Extract all medicines from results
                all_medicines = []
                
                # From OCR and enhancement
                if 'enhanced_results' in results and 'medicines' in results['enhanced_results']:
                    for med in results['enhanced_results']['medicines']:
                        if med.get('name'):
                            all_medicines.append(med['name'])
                
                # From Clinical BERT
                if clinical_analysis and 'entities' in clinical_analysis:
                    bert_medicines = clinical_analysis['entities'].get('medicines', [])
                    all_medicines.extend(bert_medicines)
                
                # Remove duplicates
                all_medicines = list(set(all_medicines))
                
                if all_medicines:
                    print(f"   💊 Medicines for prediction: {', '.join(all_medicines)}")
                    
                    # Extract symptoms from medicines
                    symptoms = self.extract_symptoms_from_medicines(all_medicines)
                    print(f"   🎯 Extracted symptoms: {len(symptoms)} symptoms")
                    
                    # Convert to symptom vector
                    symptom_vector = self.convert_to_symptom_vector(symptoms)
                    active_features = len([x for x in symptom_vector if x > 0])
                    print(f"   🔢 Symptom vector: {active_features}/132 features active")
                    
                    # Convert symptom vector to dictionary format
                    symptom_dict = {}
                    for i, value in enumerate(symptom_vector):
                        if i < len(self.disease_predictor.feature_names):
                            symptom_dict[self.disease_predictor.feature_names[i]] = value
                    
                    # Predict diseases using all available models
                    ml_predictions = {}
                    available_models = ['Random Forest', 'SVM', 'Gradient Boosting', 'Neural Network', 'Ensemble']
                    if hasattr(self.disease_predictor, 'models'):
                        available_models = list(self.disease_predictor.models.keys())
                    
                    for model_name in available_models:
                        try:
                            pred_result = self.disease_predictor.predict_disease_from_symptoms(symptom_dict, model_name)
                            ml_predictions[model_name] = pred_result
                        except Exception as e:
                            ml_predictions[model_name] = {'error': str(e)}
                    
                    results['ml_disease_predictions'] = ml_predictions
                    
                    print(f"   🎯 ML Disease Predictions:")
                    for model_name, prediction in ml_predictions.items():
                        if isinstance(prediction, dict):
                            disease = prediction.get('predicted_disease', 'Unknown')
                            confidence = prediction.get('confidence', 0)
                            print(f"      • {model_name}: {disease} ({confidence:.1%})")
                        else:
                            print(f"      • {model_name}: {prediction}")
                    
                    print(f"   ✅ Disease prediction complete")
                else:
                    print("   ⚠️  No medicines found for disease prediction")
                    results['ml_disease_predictions'] = {}
            else:
                print("   ⚠️  Disease predictor not available")
                results['ml_disease_predictions'] = {}
                
        except Exception as e:
            print(f"   ❌ Disease prediction failed: {str(e)}")
            results['ml_disease_predictions'] = {}
            
        # Step 5: Final Integration
        print("\n🎯 STEP 5: FINAL INTEGRATION")
        print("-" * 30)
        
        final_insights = self.generate_final_insights(results)
        results['final_insights'] = final_insights
        
        print(f"   ✅ Integration complete")
        print(f"   💊 Total medicines: {len(final_insights.get('extracted_medicines', []))}")
        print(f"   🏥 Total diseases: {len(final_insights.get('predicted_diseases', []))}")
        
        return results
    
    def generate_final_insights(self, results):
        """Generate comprehensive final insights"""
        insights = {
            'extracted_medicines': [],
            'predicted_diseases': [],
            'confidence_scores': {},
            'recommendations': [],
            'ml_predictions': {}
        }
        
        # Extract medicines from multiple sources
        medicines = set()
        
        # From enhanced results
        enhanced_meds = results.get('enhanced_results', {}).get('medicines', [])
        if isinstance(enhanced_meds, list):
            for med in enhanced_meds:
                if isinstance(med, dict):
                    if med.get('name'):
                        medicines.add(med['name'])
                    elif 'enhanced_name' in med:
                        medicines.add(med['enhanced_name'])
                elif isinstance(med, str):
                    medicines.add(med)
        elif isinstance(enhanced_meds, dict):
            # If medicines is a dict with medicine names as keys
            for name, details in enhanced_meds.items():
                medicines.add(name)
        
        # From Clinical BERT
        if 'clinical_analysis' in results and 'entities' in results['clinical_analysis']:
            bert_medicines = results['clinical_analysis']['entities'].get('medicines', [])
            medicines.update(bert_medicines)
        
        insights['extracted_medicines'] = list(medicines)
        
        # Extract diseases from multiple sources
        diseases = set()
        
        # From Clinical BERT
        if 'clinical_analysis' in results:
            # Direct disease extraction
            bert_diseases = results['clinical_analysis'].get('entities', {}).get('diseases', [])
            diseases.update(bert_diseases)
            
            # Disease predictions from medicines
            disease_predictions = results['clinical_analysis'].get('disease_predictions', [])
            for pred in disease_predictions:
                if 'predicted_disease' in pred:
                    diseases.add(pred['predicted_disease'])
        
        # From ML predictions
        ml_predictions = results.get('ml_disease_predictions', {})
        insights['ml_predictions'] = ml_predictions
        
        for model_name, prediction in ml_predictions.items():
            if isinstance(prediction, dict):
                disease = prediction.get('predicted_disease', '')
                if disease and disease != 'Unknown':
                    diseases.add(disease)
        
        insights['predicted_diseases'] = list(diseases)
        
        # Calculate confidence scores
        insights['confidence_scores'] = self.calculate_confidence_scores(results)
        
        # Generate recommendations
        insights['recommendations'] = self.generate_recommendations(insights)
        
        return insights
    
    def calculate_confidence_scores(self, results):
        """Calculate overall confidence scores"""
        scores = {}
        
        # OCR confidence
        ocr_confidence = results.get('ocr_results', {}).get('confidence', 0)
        scores['ocr_confidence'] = ocr_confidence / 100 if ocr_confidence > 1 else ocr_confidence
        
        # Clinical BERT confidence (estimated from entity count)
        clinical_analysis = results.get('clinical_analysis', {})
        entities = clinical_analysis.get('entities', {})
        medicine_count = len(entities.get('medicines', []))
        disease_count = len(entities.get('diseases', []))
        
        if medicine_count > 0 or disease_count > 0:
            scores['bert_confidence'] = min(0.8, (medicine_count + disease_count) * 0.1)
        else:
            scores['bert_confidence'] = 0.0
        
        # ML model confidence (average of all model confidences)
        ml_predictions = results.get('ml_disease_predictions', {})
        confidences = []
        
        for prediction in ml_predictions.values():
            if isinstance(prediction, dict):
                conf = prediction.get('confidence', 0)
                if conf > 0:
                    confidences.append(conf)
            else:
                confidences.append(0.8)
        
        scores['ml_confidence'] = sum(confidences) / len(confidences) if confidences else 0.0
        
        return scores
    
    def generate_recommendations(self, insights):
        """Generate comprehensive clinical recommendations"""
        recommendations = []
        
        medicines = insights['extracted_medicines']
        diseases = insights['predicted_diseases']
        ml_predictions = insights.get('ml_predictions', {})
        
        # Medicine-specific recommendations
        if medicines:
            recommendations.append(f"✓ Verify dosage and timing for {len(medicines)} prescribed medications")
            
            # Check for AMLODIPINE
            if any('AMLODIPINE' in med.upper() for med in medicines):
                recommendations.append("✓ Monitor blood pressure regularly (AMLODIPINE detected)")
                recommendations.append("✓ Watch for ankle swelling, dizziness, or rapid heartbeat")
        
        # Disease-specific recommendations
        if diseases:
            disease_list = list(diseases)
            if 'Hypertension' in disease_list:
                recommendations.append("✓ Monitor blood pressure daily")
                recommendations.append("✓ Reduce salt intake and maintain healthy diet")
                recommendations.append("✓ Regular cardiovascular checkups recommended")
            
            if 'Angina' in disease_list:
                recommendations.append("✓ Monitor chest pain episodes")
                recommendations.append("✓ Avoid strenuous activities until cleared by doctor")
                recommendations.append("✓ Keep nitroglycerin accessible if prescribed")
            
            if len(disease_list) > 1:
                recommendations.append(f"✓ Comprehensive monitoring for multiple conditions: {', '.join(disease_list[:3])}")
        
        # ML model insights
        high_confidence_predictions = []
        for model_name, prediction in ml_predictions.items():
            if isinstance(prediction, dict):
                confidence = prediction.get('confidence', 0)
                predicted_disease = prediction.get('predicted_disease', 'Unknown')
                if confidence > 0.5 and predicted_disease != 'Unknown':
                    high_confidence_predictions.append(f"{predicted_disease} ({confidence:.0%})")
        
        if high_confidence_predictions:
            recommendations.append(f"✓ High-confidence ML predictions require attention: {', '.join(high_confidence_predictions[:2])}")
        
        # General recommendations
        if len(medicines) > 3:
            recommendations.append("⚠️ Review for potential drug interactions (polypharmacy)")
            recommendations.append("✓ Maintain updated medication list for all healthcare providers")
        
        # Lifestyle recommendations
        recommendations.append("✓ Follow up with healthcare provider as scheduled")
        recommendations.append("✓ Report any unusual symptoms or side effects immediately")
        recommendations.append("✓ Maintain medication adherence as prescribed")
        
        if not medicines and not diseases:
            recommendations.append("⚠️ Manual review required - unclear prescription content")
            recommendations.append("✓ Contact healthcare provider for clarification")
        
        return recommendations
    
    def generate_report(self, results):
        """Generate comprehensive analysis report"""
        output_path = f"clinical_bert_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        insights = results.get('final_insights', {})
        
        report_lines = [
            "=" * 60,
            "CLINICAL BERT PRESCRIPTION ANALYSIS REPORT",
            "=" * 60,
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Image: {os.path.basename(results.get('image_path', 'Unknown'))}",
            "",
            "1. OCR PROCESSING RESULTS:",
        ]
        
        # OCR Results 
        ocr_results = results.get('ocr_results', {})
        report_lines.extend([
            f"   Confidence: {ocr_results.get('confidence', 0):.1%}",
            f"   Text Length: {len(ocr_results.get('extracted_text', ''))} characters",
            "",
            "2. MEDICINE ENHANCEMENT RESULTS:",
        ])
        
        # Enhanced medicines
        enhanced_meds = results.get('enhanced_results', {}).get('medicines', [])
        if enhanced_meds:
            report_lines.append(f"   Enhanced Medicines Found: {len(enhanced_meds)}")
            for i, med in enumerate(enhanced_meds[:10], 1):  # Show first 10
                if isinstance(med, dict):
                    name = med.get('name', 'Unknown')
                    dosage = med.get('dosage', '')
                    timing = med.get('timing', '')
                    report_lines.append(f"   {i}. {name} - {dosage} - {timing}")
        
        # Clinical BERT Results
        clinical = results.get('clinical_analysis', {})
        report_lines.extend([
            "",
            "3. CLINICAL BERT ANALYSIS:",
            f"   Entities Extracted: {len(clinical.get('entities', {}).get('medicines', []))} medicines, {len(clinical.get('entities', {}).get('diseases', []))} diseases",
            f"   Disease Predictions: {len(clinical.get('disease_predictions', []))}",
        ])
        
        if 'disease_predictions' in clinical:
            report_lines.append("   Medicine → Disease Predictions:")
            for pred in clinical['disease_predictions'][:5]:
                medicine = pred.get('medicine', 'Unknown')
                disease = pred.get('predicted_disease', 'Unknown')
                confidence = pred.get('confidence', 0)
                report_lines.append(f"     • {medicine} → {disease} ({confidence:.1%})")
        
        # ML Disease Prediction Results
        ml_predictions = results.get('ml_disease_predictions', {})
        report_lines.extend([
            "",
            "4. ML DISEASE PREDICTION RESULTS:",
            f"   Models Used: {len(ml_predictions)}",
        ])
        
        if ml_predictions:
            report_lines.append("   Predictions by Model:")
            for model_name, prediction in ml_predictions.items():
                if isinstance(prediction, dict):
                    disease = prediction.get('predicted_disease', 'Unknown')
                    confidence = prediction.get('confidence', 0)
                    report_lines.append(f"     • {model_name}: {disease} ({confidence:.1%})")
                else:
                    report_lines.append(f"     • {model_name}: {prediction}")
        
        # Final Insights Summary
        report_lines.extend([
            "",
            "5. FINAL INSIGHTS SUMMARY:",
            f"   Total Medicines Identified: {len(insights.get('extracted_medicines', []))}",
            f"   Total Diseases Predicted: {len(insights.get('predicted_diseases', []))}",
            "",
            "   Extracted Medicines:",
        ])
        
        for medicine in insights.get('extracted_medicines', []):
            report_lines.append(f"     • {medicine}")
        
        report_lines.extend([
            "",
            "   Predicted Diseases:",
        ])
        
        for disease in insights.get('predicted_diseases', []):
            report_lines.append(f"     • {disease}")
        
        report_lines.extend([
            "",
            "6. CLINICAL RECOMMENDATIONS:",
        ])
        
        for rec in insights.get('recommendations', []):
            report_lines.append(f"   • {rec}")
        
        report_lines.extend([
            "",
            "=" * 60,
            "Report generated by Clinical BERT Prescription Analyzer",
            "For medical professional use only",
            ""
        ])
        
        # Write report
        report_content = "\n".join(report_lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n📋 Analysis report saved: {output_path}")
        return output_path

def main():
    """Main analysis function"""
    print("🎯 CLINICAL BERT PRESCRIPTION ANALYSIS SYSTEM")
    print("="*70)
    
    # Initialize analyzer
    analyzer = ClinicalPrescriptionAnalyzer()
    
    # Analyze the prescription image
    image_path = r"c:\Users\DEVARGHO CHAKRABORTY\Downloads\prescription.png"
    
    print(f"\n📷 PROCESSING PRESCRIPTION IMAGE")
    print(f"   Image: {os.path.basename(image_path)}")
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("   Please ensure the prescription image is available at the specified path")
        return
    
    # Run complete analysis
    results = analyzer.analyze_prescription_image(image_path)
    
    # Generate report
    report_path = analyzer.generate_report(results)
    
    # Display summary
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print("="*30)
    
    insights = results.get('final_insights', {})
    
    print(f"📊 SUMMARY:")
    print(f"   Medicines Extracted: {len(insights.get('extracted_medicines', []))}")
    print(f"   Diseases Predicted: {len(insights.get('predicted_diseases', []))}")
    print(f"   OCR Confidence: {insights.get('confidence_scores', {}).get('ocr_confidence', 0):.1%}")
    print(f"   Clinical BERT Confidence: {insights.get('confidence_scores', {}).get('bert_confidence', 0):.1%}")
    print(f"   ML Model Confidence: {insights.get('confidence_scores', {}).get('ml_confidence', 0):.1%}")
    
    print(f"\n💊 EXTRACTED MEDICINES:")
    for medicine in insights.get('extracted_medicines', []):
        print(f"   • {medicine}")
    
    print(f"\n🏥 PREDICTED DISEASES:")
    for disease in insights.get('predicted_diseases', []):
        print(f"   • {disease}")
    
    # Show ML model predictions
    if 'ml_predictions' in insights and insights['ml_predictions']:
        print(f"\n🤖 ML MODEL PREDICTIONS:")
        for model_name, prediction in insights['ml_predictions'].items():
            if isinstance(prediction, dict):
                disease = prediction.get('predicted_disease', 'Unknown')
                confidence = prediction.get('confidence', 0)
                print(f"   • {model_name}: {disease} ({confidence:.1%})")
            else:
                print(f"   • {model_name}: {prediction}")
    
    print(f"\n📋 REPORT: {report_path}")
    print(f"\n🚀 Complete Medical Analysis System ready for production!")

if __name__ == '__main__':
    main()