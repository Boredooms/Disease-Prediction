#!/usr/bin/env python3
"""
Direct Prescription Reader
Read the clear prescription image directly without heavy processing
"""

import os
import cv2
import numpy as np
from PIL import Image
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

class DirectPrescriptionReader:
    def __init__(self):
        self.setup_tesseract()
    
    def setup_tesseract(self):
        """Setup Tesseract paths"""
        if not TESSERACT_AVAILABLE:
            print("❌ Tesseract not available - install: pip install pytesseract")
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
                    version = pytesseract.get_tesseract_version()
                    print(f"✅ Tesseract {version} found: {path}")
                    return True
            except:
                continue
        
        print("❌ Tesseract not found")
        return False
    
    def minimal_enhance(self, image):
        """Very minimal enhancement - just ensure good contrast"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Very gentle contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        
        return enhanced
    
    def extract_text_regions(self, image):
        """Extract different regions of the prescription"""
        height, width = image.shape[:2]
        
        regions = {
            'header': image[0:int(height*0.15), :],           # Doctor info
            'patient': image[int(height*0.15):int(height*0.25), :],  # Patient info  
            'medicines': image[int(height*0.25):int(height*0.70), :], # Medicine table
            'advice': image[int(height*0.70):int(height*0.85), :],    # Advice
            'signature': image[int(height*0.85):, :]          # Signature area
        }
        
        return regions
    
    def read_prescription(self, image_path):
        """Read prescription with multiple approaches"""
        print(f"🔍 READING PRESCRIPTION: {os.path.basename(image_path)}")
        print("="*60)
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image")
            return None
        
        print(f"📐 Original size: {image.shape}")
        
        # Convert to grayscale and enhance minimally
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = self.minimal_enhance(gray)
        
        # Save enhanced version
        cv2.imwrite('direct_enhanced.png', enhanced)
        print(f"💾 Enhanced image saved: direct_enhanced.png")
        
        results = {}
        
        # Method 1: Full image OCR
        print(f"\n📋 METHOD 1: FULL IMAGE OCR")
        print("-" * 30)
        
        configs = [
            ('PSM 3 - Fully automatic', '--oem 3 --psm 3'),
            ('PSM 4 - Single column', '--oem 3 --psm 4'),
            ('PSM 6 - Uniform block', '--oem 3 --psm 6'),
            ('PSM 1 - Auto with orientation', '--oem 3 --psm 1'),
        ]
        
        best_full_text = ""
        best_confidence = 0
        
        for config_name, config in configs:
            try:
                text = pytesseract.image_to_string(enhanced, config=config)
                data = pytesseract.image_to_data(enhanced, config=config, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_conf = np.mean(confidences) if confidences else 0
                
                print(f"  🔧 {config_name}: {avg_conf:.1f}% confidence")
                
                if avg_conf > best_confidence:
                    best_full_text = text
                    best_confidence = avg_conf
                    
            except Exception as e:
                print(f"  ❌ {config_name}: {str(e)[:50]}")
        
        results['full_image'] = {
            'text': best_full_text,
            'confidence': best_confidence
        }
        
        # Method 2: Region-based OCR
        print(f"\n📋 METHOD 2: REGION-BASED OCR")
        print("-" * 30)
        
        regions = self.extract_text_regions(enhanced)
        region_results = {}
        
        for region_name, region_img in regions.items():
            try:
                text = pytesseract.image_to_string(region_img, config='--oem 3 --psm 6')
                data = pytesseract.image_to_data(region_img, config='--oem 3 --psm 6', output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_conf = np.mean(confidences) if confidences else 0
                
                clean_text = text.strip()
                if clean_text:
                    print(f"  📍 {region_name.capitalize():12}: {avg_conf:5.1f}% | {clean_text[:50]}")
                    region_results[region_name] = {
                        'text': clean_text,
                        'confidence': avg_conf
                    }
                
                # Save region for debugging
                cv2.imwrite(f'region_{region_name}.png', region_img)
                
            except Exception as e:
                print(f"  ❌ {region_name}: {str(e)[:30]}")
        
        results['regions'] = region_results
        
        # Method 3: Medicine extraction
        print(f"\n💊 METHOD 3: MEDICINE EXTRACTION")
        print("-" * 30)
        
        medicine_region = regions.get('medicines')
        if medicine_region is not None:
            medicines = self.extract_medicines(medicine_region)
            results['medicines'] = medicines
            
            if medicines:
                print(f"  ✅ Found {len(medicines)} medicines:")
                for i, med in enumerate(medicines, 1):
                    print(f"    {i}. {med}")
        
        # Display best results
        print(f"\n🏆 BEST RESULTS:")
        print("="*40)
        
        if best_full_text:
            lines = [line.strip() for line in best_full_text.split('\n') if line.strip()]
            print(f"📋 Full Text ({best_confidence:.1f}% confidence):")
            for i, line in enumerate(lines[:15], 1):  # First 15 lines
                print(f"  {i:2d}: {line}")
        
        # Save all results
        with open('direct_ocr_results.txt', 'w', encoding='utf-8') as f:
            f.write("DIRECT PRESCRIPTION OCR RESULTS\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"FULL IMAGE OCR ({best_confidence:.1f}% confidence):\n")
            f.write("-" * 30 + "\n")
            f.write(best_full_text + "\n\n")
            
            f.write("REGION-BASED OCR:\n")
            f.write("-" * 30 + "\n")
            for region, data in region_results.items():
                f.write(f"{region.upper()} ({data['confidence']:.1f}%):\n")
                f.write(data['text'] + "\n\n")
        
        print(f"\n💾 Results saved to: direct_ocr_results.txt")
        return results
    
    def extract_medicines(self, medicine_region):
        """Extract medicine names from the medicine table region"""
        try:
            # Try different OCR configs for medicine table
            configs = [
                '--oem 3 --psm 6',  # Uniform block
                '--oem 3 --psm 4',  # Single column
                '--oem 3 --psm 8',  # Single word
            ]
            
            medicines = []
            
            for config in configs:
                text = pytesseract.image_to_string(medicine_region, config=config)
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                for line in lines:
                    # Look for medicine patterns
                    if 'TAB' in line.upper() or 'CAP' in line.upper() or 'SYR' in line.upper():
                        medicines.append(line)
                    elif any(keyword in line.upper() for keyword in ['MORNING', 'NIGHT', 'DAYS']):
                        medicines.append(line)
                
                if medicines:
                    break
            
            return list(set(medicines))  # Remove duplicates
            
        except Exception as e:
            print(f"❌ Medicine extraction failed: {e}")
            return []

def main():
    reader = DirectPrescriptionReader()
    
    # Test on original prescription
    prescription_path = r'C:\Users\DEVARGHO CHAKRABORTY\Downloads\prescription.png'
    
    if os.path.exists(prescription_path):
        results = reader.read_prescription(prescription_path)
        
        print(f"\n🎯 SUMMARY:")
        print("="*30)
        
        if results and results.get('full_image', {}).get('text'):
            full_text = results['full_image']['text']
            confidence = results['full_image']['confidence']
            word_count = len(full_text.split())
            
            print(f"✅ Successfully extracted {word_count} words")
            print(f"📊 Overall confidence: {confidence:.1f}%")
            print(f"📄 Text length: {len(full_text)} characters")
            
            # Count medicine references
            medicine_count = full_text.upper().count('TAB') + full_text.upper().count('CAP')
            print(f"💊 Medicine references found: {medicine_count}")
            
        else:
            print(f"❌ No readable text extracted")
    
    else:
        print(f"❌ Prescription file not found: {prescription_path}")

if __name__ == '__main__':
    main()