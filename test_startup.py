#!/usr/bin/env python3
"""
Quick test to ensure the app can start properly
"""
import os
import sys

def test_app_startup():
    print("Testing app startup...")
    
    # Test imports
    try:
        print("Testing imports...")
        from app import app
        print("✅ App imported successfully")
        
        # Test Flask app creation
        print(f"✅ Flask app created: {app}")
        print(f"✅ App debug mode: {app.debug}")
        
        # Test port configuration
        port = int(os.environ.get('PORT', 5000))
        print(f"✅ Port configured: {port}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during startup test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_app_startup()
    if success:
        print("\n🎉 App startup test passed!")
        sys.exit(0)
    else:
        print("\n💥 App startup test failed!")
        sys.exit(1)