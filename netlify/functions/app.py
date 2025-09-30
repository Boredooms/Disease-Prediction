import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app import app

def handler(event, context):
    """
    Netlify Functions handler for Flask app
    """
    try:
        # Convert Netlify event to WSGI environ
        environ = {
            'REQUEST_METHOD': event.get('httpMethod', 'GET'),
            'PATH_INFO': event.get('path', '/'),
            'QUERY_STRING': event.get('queryStringParameters', {}),
            'CONTENT_TYPE': event.get('headers', {}).get('content-type', ''),
            'CONTENT_LENGTH': str(len(event.get('body', ''))),
            'HTTP_HOST': event.get('headers', {}).get('host', 'localhost'),
            'SERVER_NAME': 'localhost',
            'SERVER_PORT': '80',
            'wsgi.input': event.get('body', ''),
            'wsgi.errors': sys.stderr,
            'wsgi.version': (1, 0),
            'wsgi.multithread': False,
            'wsgi.multiprocess': True,
            'wsgi.run_once': False,
            'wsgi.url_scheme': 'https',
        }
        
        # Add headers to environ
        headers = event.get('headers', {})
        for key, value in headers.items():
            key = 'HTTP_' + key.upper().replace('-', '_')
            environ[key] = value
        
        # Store response data
        response_data = {'status': '200', 'headers': [], 'body': ''}
        
        def start_response(status, headers):
            response_data['status'] = status
            response_data['headers'] = headers
        
        # Run the Flask app
        result = app(environ, start_response)
        
        # Collect response body
        body = b''.join(result).decode('utf-8')
        
        return {
            'statusCode': int(response_data['status'].split()[0]),
            'headers': dict(response_data['headers']),
            'body': body
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': f'Error: {str(e)}'
        }