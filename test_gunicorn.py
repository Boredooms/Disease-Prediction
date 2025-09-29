#!/usr/bin/env python3
"""
Test gunicorn configuration
"""
import subprocess
import os

def test_gunicorn_config():
    print("Testing gunicorn configuration...")
    
    # Set PORT environment variable
    os.environ['PORT'] = '5000'
    
    try:
        # Test gunicorn config syntax
        result = subprocess.run(['gunicorn', '--check-config', '-c', 'gunicorn.conf.py'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Gunicorn configuration is valid")
            return True
        else:
            print(f"❌ Gunicorn configuration error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Gunicorn config test timed out")
        return False
    except Exception as e:
        print(f"❌ Error testing gunicorn config: {e}")
        return False

if __name__ == "__main__":
    test_gunicorn_config()