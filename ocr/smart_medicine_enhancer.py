#!/usr/bin/env python3
"""
Smart Medicine Enhancer
Use our medical dataset to improve medicine recognition with fuzzy matching
"""

import os
import cv2
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import re
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

class SmartMedicineEnhancer:
    def __init__(self, data_path="c:\\Just Ai\\EndToEndDiseasePredictor\\data\\OCR"):
        self.data_path = data_path
        self.medicine_names = self.load_medicine_database()
        self.setup_tesseract()
        
    def setup_tesseract(self):
        """Setup Tesseract"""
        if not TESSERACT_AVAILABLE:
            return False
        
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            'tesseract'
        ]
        
        for path in possible_paths:
            try:
                if os.path.exists(path) or path == 'tesseract':
                    pytesseract.pytesseract.tesseract_cmd = path
                    return True
            except:
                continue
        return False
    
    def load_medicine_database(self):
        """Load medicine names from our dataset"""
        print(f"💊 LOADING MEDICINE DATABASE")
        print("-" * 40)
        
        medicine_names = set()
        
        # Load from medicine CSV
        medicine_csv = os.path.join(self.data_path, "medicine_names_images.csv")
        if os.path.exists(medicine_csv):
            try:
                df = pd.read_csv(medicine_csv)
                
                # Try different column names
                for col in ['label', 'medicine_name', 'name', 'text']:
                    if col in df.columns:
                        names = df[col].dropna().astype(str).str.strip()
                        medicine_names.update(names)
                        print(f"   ✅ Loaded {len(names)} names from column '{col}'")
                        break
                
                # If no standard column, use first text column
                if not medicine_names and len(df.columns) > 1:
                    names = df.iloc[:, 1].dropna().astype(str).str.strip()
                    medicine_names.update(names)
                    print(f"   ✅ Loaded {len(names)} names from first text column")
                    
            except Exception as e:
                print(f"   ⚠️  Error loading medicine CSV: {e}")
        
        # Add common medicine patterns
        common_medicines = [
            'LISINOPRIL', 'ASPIRIN', 'METFORMIN', 'ATORVASTATIN', 'LEVOTHYROXINE',
            'AMLODIPINE', 'METOPROLOL', 'OMEPRAZOLE', 'SIMVASTATIN', 'LOSARTAN',
            'HYDROCHLOROTHIAZIDE', 'GABAPENTIN', 'SERTRALINE', 'PREDNISONE',
            'AMOXICILLIN', 'CIPROFLOXACIN', 'AZITHROMYCIN', 'DOXYCYCLINE',
            'IBUPROFEN', 'ACETAMINOPHEN', 'PARACETAMOL', 'DICLOFENAC'
        ]
        medicine_names.update(common_medicines)
        
        # Clean and process names
        cleaned_names = set()
        for name in medicine_names:
            cleaned = str(name).upper().strip()
            if len(cleaned) > 2 and cleaned.isalpha():  # Only alphabetic names
                cleaned_names.add(cleaned)
        
        medicine_list = sorted(list(cleaned_names))
        print(f"   📊 Total unique medicines: {len(medicine_list)}")
        print(f"   📋 Sample medicines: {medicine_list[:10]}")
        
        return medicine_list
    
    def similarity(self, a, b):
        """Calculate similarity between two strings"""
        return SequenceMatcher(None, a.upper(), b.upper()).ratio()
    
    def find_best_medicine_match(self, ocr_text, threshold=0.6):
        """Find best medicine match using fuzzy matching"""
        if not ocr_text or len(ocr_text) < 3:
            return None
        
        ocr_clean = re.sub(r'[^A-Z]', '', ocr_text.upper())
        if len(ocr_clean) < 3:
            return None
        
        best_match = None
        best_score = 0
        
        for medicine in self.medicine_names:
            score = self.similarity(ocr_clean, medicine)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = medicine
        
        return {
            'original': ocr_text,
            'corrected': best_match,
            'confidence': best_score
        } if best_match else None
    
    def extract_medicine_region(self, image_path):
        """Extract medicine table region with multiple approaches"""
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Medicine table region (25-70% of image height)
        medicine_start = int(height * 0.25)
        medicine_end = int(height * 0.70)
        medicine_region = gray[medicine_start:medicine_end, :]
        
        # Gentle enhancement
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        enhanced = clahe.apply(medicine_region)
        
        return enhanced
    
    def smart_ocr_with_enhancement(self, image_path):
        """Perform OCR with smart medicine enhancement"""
        print(f"🎯 SMART MEDICINE OCR")
        print("="*50)
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
        
        # Extract medicine region
        medicine_region = self.extract_medicine_region(image_path)
        cv2.imwrite('smart_medicine_region.png', medicine_region)
        print(f"💾 Medicine region saved: smart_medicine_region.png")
        
        # Multiple OCR attempts
        ocr_configs = [
            ('Table optimized', '--oem 3 --psm 6'),
            ('Line by line', '--oem 3 --psm 4'),
            ('Auto detection', '--oem 3 --psm 3'),
            ('Single column', '--oem 3 --psm 4'),
        ]
        
        all_results = {}
        best_medicines = []
        
        for config_name, config in ocr_configs:
            try:
                print(f"\\n🔧 {config_name.upper()}")
                print("-" * 25)
                
                # Get raw OCR text
                raw_text = pytesseract.image_to_string(medicine_region, config=config)
                lines = [line.strip() for line in raw_text.split('\\n') if line.strip()]
                
                print(f"   📝 Raw lines: {len(lines)}")
                
                # Process each line for medicines
                config_medicines = []
                for line in lines:
                    # Look for medicine patterns
                    if any(keyword in line.upper() for keyword in ['TAB', 'CAP', 'SYR', 'INJ']):
                        print(f"   🔍 Medicine line: {line}")
                        
                        # Extract medicine name using improved patterns
                        patterns = [
                            r'(TAB|CAP|SYR|INJ)[,\s]+([A-Z][A-Z\s]+?)(?:\s+\d|\s+[A-Z]+ing|\s*$)',  # TAB, MEDICINE NAME
                            r'(TAB|CAP|SYR|INJ)[,\s]+(\w+)',  # Simple TAB MEDICINE
                        ]
                        
                        found_medicine = False
                        for pattern in patterns:
                            matches = re.findall(pattern, line, re.IGNORECASE)
                            for match in matches:
                                if len(match) >= 2:
                                    med_type = match[0]
                                    potential_med = match[1].strip()
                                    
                                    if len(potential_med) > 2:  # Valid medicine name
                                        # Clean medicine name
                                        clean_med = re.sub(r'[^A-Z\s]', '', potential_med.upper()).strip()
                                        
                                        # Try to match with our database
                                        db_match = self.find_best_medicine_match(clean_med, threshold=0.5)
                                        if db_match:
                                            config_medicines.append({
                                                'line': line,
                                                'raw_name': potential_med,
                                                'corrected_name': db_match['corrected'],
                                                'confidence': db_match['confidence'],
                                                'config': config_name,
                                                'type': med_type
                                            })
                                            print(f"   💊 {potential_med} → {db_match['corrected']} ({db_match['confidence']:.2f})")
                                        else:
                                            # Keep original medicine name even if not in database
                                            config_medicines.append({
                                                'line': line,
                                                'raw_name': potential_med,
                                                'corrected_name': clean_med if clean_med else potential_med.upper(),
                                                'confidence': 0.7,  # Good OCR confidence
                                                'config': config_name,
                                                'type': med_type
                                            })
                                            print(f"   💊 {potential_med} → {clean_med or potential_med.upper()} (OCR original)")
                                        
                                        found_medicine = True
                                        break
                            
                            if found_medicine:
                                break
                
                all_results[config_name] = {
                    'raw_text': raw_text,
                    'medicines': config_medicines,
                    'medicine_count': len(config_medicines)
                }
                
                print(f"   ✅ Found {len(config_medicines)} medicines")
                
                # Keep best results
                if len(config_medicines) > len(best_medicines):
                    best_medicines = config_medicines
                    
            except Exception as e:
                print(f"   ❌ {config_name} failed: {str(e)[:50]}")
        
        # Combine and deduplicate results
        print(f"\\n🏆 SMART ENHANCEMENT RESULTS")
        print("="*40)
        
        unique_medicines = {}
        for med in best_medicines:
            key = med['corrected_name']
            if key not in unique_medicines or med['confidence'] > unique_medicines[key]['confidence']:
                unique_medicines[key] = med
        
        final_medicines = list(unique_medicines.values())
        
        print(f"📊 Total medicines found: {len(final_medicines)}")
        
        if final_medicines:
            print(f"\\n💊 ENHANCED MEDICINE LIST:")
            print("-" * 30)
            
            for i, med in enumerate(final_medicines, 1):
                print(f"{i:2d}. {med['corrected_name']}")
                print(f"    Original OCR: '{med['raw_name']}'")
                print(f"    Confidence: {med['confidence']:.1%}")
                print(f"    Full line: {med['line'][:60]}...")
                print()
        
        # Extract additional details (dosage, duration)
        enhanced_medicines = self.extract_medicine_details(final_medicines, all_results)
        
        # Save results
        self.save_enhanced_results(enhanced_medicines, all_results, image_path)
        
        return {
            'medicines': enhanced_medicines,
            'raw_results': all_results,
            'total_found': len(enhanced_medicines)
        }
    
    def extract_medicine_details(self, medicines, all_results):
        """Extract dosage and duration details"""
        print(f"\\n📋 EXTRACTING MEDICINE DETAILS")
        print("-" * 35)
        
        enhanced = []
        
        for med in medicines:
            details = {
                'name': med['corrected_name'],
                'raw_name': med['raw_name'],
                'confidence': med['confidence'],
                'line': med['line'],
                'dosage': self.extract_dosage_pattern(med['line']),
                'duration': self.extract_duration_pattern(med['line']),
                'timing': self.extract_timing_pattern(med['line'])
            }
            
            enhanced.append(details)
            
            print(f"💊 {details['name']}")
            print(f"   Dosage: {details['dosage'] or 'Not found'}")
            print(f"   Duration: {details['duration'] or 'Not found'}")
            print(f"   Timing: {details['timing'] or 'Not found'}")
        
        return enhanced
    
    def extract_dosage_pattern(self, line):
        """Extract dosage pattern from line"""
        # Look for patterns like "1 Morning", "1/2 Night", etc.
        patterns = [
            r'(\\d+(?:/\\d+)?)\\s*(?:Morning|Night|Eve|Aft)',
            r'(\\d+)\\s*(?:times?|tab|cap)',
            r'(\\d+(?:/\\d+)?)\\s*(?:mg|ml|g)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def extract_duration_pattern(self, line):
        """Extract duration from line"""
        patterns = [
            r'(\\d+)\\s*(?:days?|weeks?|months?)',
            r'for\\s+(\\d+)\\s*(?:days?|weeks?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return f"{match.group(1)} days"
        
        return None
    
    def extract_timing_pattern(self, line):
        """Extract timing information"""
        timings = []
        line_upper = line.upper()
        
        timing_words = ['MORNING', 'NIGHT', 'EVENING', 'AFTERNOON', 'EVE', 'AFT']
        for timing in timing_words:
            if timing in line_upper:
                timings.append(timing.capitalize())
        
        return ', '.join(timings) if timings else None
    
    def save_enhanced_results(self, medicines, all_results, image_path):
        """Save enhanced results"""
        with open('smart_medicine_results.txt', 'w', encoding='utf-8') as f:
            f.write("SMART MEDICINE ENHANCEMENT RESULTS\\n")
            f.write("="*50 + "\\n\\n")
            
            f.write(f"Source Image: {os.path.basename(image_path)}\\n")
            f.write(f"Total Medicines Found: {len(medicines)}\\n\\n")
            
            f.write("ENHANCED MEDICINES:\\n")
            f.write("-" * 25 + "\\n")
            
            for i, med in enumerate(medicines, 1):
                f.write(f"{i}. {med['name']}\\n")
                f.write(f"   Original OCR: {med['raw_name']}\\n")
                f.write(f"   Confidence: {med['confidence']:.1%}\\n")
                f.write(f"   Dosage: {med['dosage'] or 'Not specified'}\\n")
                f.write(f"   Duration: {med['duration'] or 'Not specified'}\\n")
                f.write(f"   Timing: {med['timing'] or 'Not specified'}\\n")
                f.write(f"   Full line: {med['line']}\\n\\n")
            
            f.write("\\nRAW OCR RESULTS:\\n")
            f.write("-" * 20 + "\\n")
            for config, result in all_results.items():
                f.write(f"\\n{config}:\\n")
                f.write(result['raw_text'] + "\\n\\n")
        
        print(f"\\n💾 Results saved: smart_medicine_results.txt")

def main():
    enhancer = SmartMedicineEnhancer()
    
    if len(enhancer.medicine_names) == 0:
        print(f"❌ No medicine database loaded!")
        return
    
    prescription_path = r'C:\\Users\\DEVARGHO CHAKRABORTY\\Downloads\\prescription.png'
    
    if os.path.exists(prescription_path):
        results = enhancer.smart_ocr_with_enhancement(prescription_path)
        
        if results and results['medicines']:
            print(f"\\n🎉 SUCCESS SUMMARY:")
            print(f"✅ Enhanced {results['total_found']} medicines with smart matching")
            print(f"📊 Using database of {len(enhancer.medicine_names)} medicines")
            
            print(f"\\n📋 FINAL ENHANCED MEDICINES:")
            for i, med in enumerate(results['medicines'], 1):
                dosage_info = f" - {med['dosage']}" if med['dosage'] else ""
                timing_info = f" - {med['timing']}" if med['timing'] else ""
                duration_info = f" - {med['duration']}" if med['duration'] else ""
                print(f"  {i}. {med['name']}{dosage_info}{timing_info}{duration_info}")
        else:
            print(f"❌ No medicines could be enhanced")
    else:
        print(f"❌ Prescription not found")

if __name__ == '__main__':
    main()