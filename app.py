"""
Advanced Disease Prediction Web Application
Stunning dark-themed interface with OCR → Clinical BERT → ML Pipeline
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import tempfile
import base64
from PIL import Image
import io
import traceback
import logging
from datetime import datetime

# Import our disease prediction system components
from ocr.direct_prescription_reader import DirectPrescriptionReader
from ocr.smart_medicine_enhancer import SmartMedicineEnhancer
from nlp.clinical_bert_processor import ClinicalBERTProcessor
from disease_predictor.disease_prediction_system import AdvancedDiseasePredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'temp_uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Static file configuration
@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

# Initialize our prediction system
try:
    # Initialize all components
    logger.info("Initializing OCR system...")
    ocr_reader = DirectPrescriptionReader()
    medicine_enhancer = SmartMedicineEnhancer()
    
    logger.info("Initializing Clinical BERT processor...")
    bert_processor = ClinicalBERTProcessor()
    
    logger.info("Initializing Disease Prediction system...")
    disease_predictor = AdvancedDiseasePredictor()
    disease_predictor.load_models()
    
    logger.info("All systems initialized successfully!")
    
except Exception as e:
    logger.error(f"Failed to initialize systems: {str(e)}")
    traceback.print_exc()

@app.route('/')
def index():
    """Main page with stunning dark interface"""
    return render_template('index.html')

@app.route('/ping')
def ping():
    """Simple ping endpoint for quick health checks"""
    return jsonify({'status': 'ok', 'message': 'MediAI Disease Predictor is running'})

@app.route('/upload', methods=['POST'])
def upload_prescription():
    """Handle prescription image upload and process through complete pipeline"""
    try:
        # Check if file was uploaded
        if 'prescription' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['prescription']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(filepath)
        
        logger.info(f"Processing prescription: {temp_filename}")
        
        # Stage 1: OCR Processing
        logger.info("Stage 1: OCR Processing...")
        ocr_result = ocr_reader.read_prescription(filepath)
        
        # Stage 2: Medicine Enhancement
        logger.info("Stage 2: Medicine Enhancement...")
        enhanced_medicines = medicine_enhancer.smart_ocr_with_enhancement(filepath)
        
        # Stage 3: Clinical BERT Processing
        logger.info("Stage 3: Clinical BERT Processing...")
        detected_text = ocr_result.get('full_image', {}).get('text', '') if ocr_result else ''
        
        # Create enhanced text from both OCR and enhanced medicines for better analysis
        enhanced_text = detected_text
        if enhanced_medicines and 'medicines' in enhanced_medicines:
            enhanced_medicine_names = []
            for med in enhanced_medicines['medicines']:
                if isinstance(med, dict):
                    med_name = med.get('name', med.get('medicine', ''))
                    if med_name and len(med_name) < 50:  # Clean medicine names only
                        enhanced_medicine_names.append(med_name)
                else:
                    med_str = str(med)
                    if len(med_str) < 50:
                        enhanced_medicine_names.append(med_str)
            
            # Add enhanced medicines to the text for better Clinical BERT analysis
            if enhanced_medicine_names:
                enhanced_text += f"\n\nEnhanced Medicines: {', '.join(enhanced_medicine_names)}"
        
        clinical_analysis = bert_processor.process_prescription_text(enhanced_text)
        
        # Stage 4: Disease Prediction
        logger.info("Stage 4: Disease Prediction...")
        
        # Combine all extracted medicines
        all_medicines = []
        
        # From OCR results
        if ocr_result and 'medicines' in ocr_result:
            all_medicines.extend(ocr_result['medicines'])
        
        # From Smart Medicine Enhancer - get clean medicine names
        if enhanced_medicines and 'medicines' in enhanced_medicines:
            for med in enhanced_medicines['medicines']:
                if isinstance(med, dict):
                    # Extract clean medicine name (like AMLODIPINE)
                    med_name = med.get('name', med.get('medicine', ''))
                    if med_name and len(med_name) < 50:  # Filter out long gibberish text
                        all_medicines.append(med_name)
                else:
                    med_str = str(med)
                    if len(med_str) < 50:  # Filter out long gibberish text
                        all_medicines.append(med_str)
        
        # From Clinical BERT
        if clinical_analysis and 'medications' in clinical_analysis:
            all_medicines.extend(clinical_analysis['medications'])
        
        # Remove duplicates
        unique_medicines = list(set(all_medicines))
        
        # Get disease predictions using the comprehensive prediction system
        disease_predictions = []
        if unique_medicines and hasattr(disease_predictor, 'models'):
            try:
                # Prepare proper OCR and NLP data structure for prediction
                ocr_for_prediction = {
                    'medicines': unique_medicines,
                    'full_text': ocr_result.get('full_image', {}).get('text', '') if ocr_result else ''
                }
                
                nlp_for_prediction = {
                    'entities': {
                        'medicines': clinical_analysis.get('medications', []) if clinical_analysis else [],
                        'diseases': clinical_analysis.get('medical_conditions', []) if clinical_analysis else []
                    },
                    'disease_predictions': clinical_analysis.get('disease_predictions', []) if clinical_analysis else []
                }
                
                # Use the OCR and NLP results to get predictions
                pred_result = disease_predictor.predict_from_ocr_nlp(ocr_for_prediction, nlp_for_prediction)
                logger.info(f"Complete prediction result: {pred_result}")
                
                if pred_result and 'predicted_disease' in pred_result:
                    # Add the main prediction
                    disease_predictions.append({
                        'disease': pred_result['predicted_disease'],
                        'confidence': float(pred_result.get('confidence', 0)),
                        'model_name': 'Ensemble',
                        'medicine': ', '.join(unique_medicines[:3]),  # Show up to 3 medicines
                        'drug_recommendations': pred_result.get('drug_recommendations', [])
                    })
                    logger.info(f"Added main disease prediction: {pred_result['predicted_disease']} with confidence {pred_result.get('confidence', 0)}")
                            
            except Exception as e:
                logger.warning(f"Failed to get disease predictions: {str(e)}")
                
                # Fallback: try individual medicine predictions if available
                for medicine in unique_medicines:
                    try:
                        # Try to use medicine to symptoms mapping if available
                        if hasattr(disease_predictor, 'drug_to_symptoms') and medicine.upper() in disease_predictor.drug_to_symptoms:
                            symptoms = disease_predictor.drug_to_symptoms[medicine.upper()]
                            symptom_dict = {symptom: 1 for symptom in symptoms}
                            
                            # Get predictions from different models
                            available_models = list(disease_predictor.models.keys()) if hasattr(disease_predictor, 'models') else ['Random Forest']
                            for model_name in available_models[:3]:  # Limit to 3 models to avoid too many predictions
                                try:
                                    pred_result = disease_predictor.predict_disease_from_symptoms(symptom_dict, model_name)
                                    logger.info(f"Prediction result for {medicine} with {model_name}: {pred_result}")
                                    
                                    if pred_result and 'disease' in pred_result:
                                        disease_pred = {
                                            'disease': pred_result['disease'],
                                            'confidence': float(pred_result.get('confidence', 0)),
                                            'model_name': model_name,
                                            'medicine': medicine
                                        }
                                        disease_predictions.append(disease_pred)
                                        logger.info(f"Added prediction: {disease_pred}")
                                        
                                except Exception as model_e:
                                    logger.warning(f"Model {model_name} failed for {medicine}: {str(model_e)}")
                                    
                    except Exception as med_e:
                        logger.debug(f"Failed to predict for medicine {medicine}: {str(med_e)}")
        
        # Clean up temporary file
        try:
            os.remove(filepath)
        except:
            pass
        
        # Prepare response
        response_data = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'stages': {
                'ocr': {
                    'detected_text': ocr_result.get('full_image', {}).get('text', '') if ocr_result else '',
                    'confidence': ocr_result.get('full_image', {}).get('confidence', 0) / 100 if ocr_result else 0,
                    'processing_time': 1.0  # Placeholder
                },
                'medicine_enhancement': {
                    'enhanced_medicines': [
                        {
                            'medicine': med.get('name', str(med)) if isinstance(med, dict) else str(med),
                            'confidence': med.get('confidence', 1.0) if isinstance(med, dict) else 1.0,
                            'original_ocr': med.get('original_ocr', '') if isinstance(med, dict) else ''
                        } for med in enhanced_medicines.get('medicines', [])
                    ] if enhanced_medicines else [],
                    'count': enhanced_medicines.get('total_found', 0) if enhanced_medicines else 0
                },
                'clinical_bert': {
                    'medical_entities': len(unique_medicines),  # Use actual found medicines
                    'medications': unique_medicines[:10],  # Show first 10 actual medicines found
                    'medications_count': len(unique_medicines),  # Actual count of medicines
                    'medical_conditions': clinical_analysis.get('medical_conditions', [])[:10] if clinical_analysis else [],
                    'conditions_count': len(clinical_analysis.get('medical_conditions', [])) if clinical_analysis else 0,
                    'dosages': clinical_analysis.get('dosages', [])[:5] if clinical_analysis else [],
                    'disease_predictions_count': len(clinical_analysis.get('disease_predictions', [])) if clinical_analysis else 0
                },
                'disease_prediction': {
                    'predictions': disease_predictions,
                    'total_medicines_analyzed': len(unique_medicines),
                    'total_predictions': len(disease_predictions)
                }
            },
            'summary': {
                'total_medicines_found': len(unique_medicines),
                'unique_medicines': unique_medicines,
                'total_disease_predictions': len(disease_predictions),
                'top_predictions': sorted(disease_predictions, key=lambda x: x.get('confidence', 0), reverse=True)[:5] if disease_predictions else []
            }
        }
        
        logger.info(f"Processing complete! Found {len(unique_medicines)} medicines, {len(disease_predictions)} predictions")
        logger.info(f"Final disease predictions: {disease_predictions}")
        logger.info(f"Top predictions for frontend: {response_data['summary']['top_predictions']}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error processing prescription: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/demo', methods=['POST'])
def demo_prediction():
    """Demo endpoint for testing with medicine names"""
    try:
        data = request.get_json()
        medicine_name = data.get('medicine', '').strip()
        
        if not medicine_name:
            return jsonify({'error': 'No medicine name provided'}), 400
        
        logger.info(f"Demo prediction for: {medicine_name}")
        
        # Get predictions
        predictions = disease_predictor.predict_disease_from_medicine(medicine_name)
        
        response_data = {
            'success': True,
            'medicine': medicine_name,
            'predictions': predictions or [],
            'total_predictions': len(predictions) if predictions else 0
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Demo prediction error: {str(e)}")
        return jsonify({'error': f'Demo failed: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for deployment"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'systems': {
            'ocr': 'ready',
            'clinical_bert': 'ready',
            'disease_predictor': 'ready'
        }
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error. Please try again.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)