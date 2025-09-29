#!/usr/bin/env python3
"""
Test port binding for Render deployment
"""
import os
import socket
import time

def test_port_binding():
    port = int(os.environ.get('PORT', 5000))
    print(f"Testing port binding on port {port}...")
    
    # Test if we can bind to the port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('0.0.0.0', port))
        sock.listen(1)
        print(f"✅ Successfully bound to port {port}")
        
        # Test connection
        sock.settimeout(1.0)
        print(f"✅ Port {port} is available for binding")
        return True
        
    except socket.error as e:
        print(f"❌ Failed to bind to port {port}: {e}")
        return False
    finally:
        sock.close()

if __name__ == "__main__":
    success = test_port_binding()
    if success:
        print("🎉 Port binding test passed!")
    else:
        print("💥 Port binding test failed!")