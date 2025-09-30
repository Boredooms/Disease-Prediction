import sys
import os
from io import StringIO

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Set working directory to project root
os.chdir(project_root)

# Import Flask app
from app import app

def handler(event, context):
    """
    Netlify Functions handler for Flask app
    """
    try:
        # Handle different path formats
        path = event.get('path', '/')
        if path.startswith('/.netlify/functions/app'):
            path = path.replace('/.netlify/functions/app', '') or '/'
        
        # Parse query string
        query_string = ''
        if event.get('queryStringParameters'):
            import urllib.parse
            query_string = urllib.parse.urlencode(event['queryStringParameters'])
        
        # Get request body
        body = event.get('body', '') or ''
        if event.get('isBase64Encoded', False):
            import base64
            body = base64.b64decode(body).decode('utf-8')
        
        # Create WSGI environ
        environ = {
            'REQUEST_METHOD': event.get('httpMethod', 'GET'),
            'PATH_INFO': path,
            'QUERY_STRING': query_string,
            'CONTENT_TYPE': event.get('headers', {}).get('content-type', ''),
            'CONTENT_LENGTH': str(len(body)),
            'HTTP_HOST': event.get('headers', {}).get('host', 'localhost'),
            'SERVER_NAME': event.get('headers', {}).get('host', 'localhost').split(':')[0],
            'SERVER_PORT': '443',
            'wsgi.input': StringIO(body),
            'wsgi.errors': sys.stderr,
            'wsgi.version': (1, 0),
            'wsgi.multithread': False,
            'wsgi.multiprocess': True,
            'wsgi.run_once': False,
            'wsgi.url_scheme': 'https',
        }
        
        # Add all headers to environ
        headers = event.get('headers', {})
        for key, value in headers.items():
            key = 'HTTP_' + key.upper().replace('-', '_')
            environ[key] = value
        
        # Capture response
        response_data = {'status': '200 OK', 'headers': []}
        
        def start_response(status, headers, exc_info=None):
            response_data['status'] = status
            response_data['headers'] = headers
            return lambda s: None
        
        # Call Flask app
        app_iter = app(environ, start_response)
        
        # Collect response body
        try:
            body_parts = []
            for data in app_iter:
                if isinstance(data, bytes):
                    body_parts.append(data.decode('utf-8'))
                else:
                    body_parts.append(str(data))
            response_body = ''.join(body_parts)
        finally:
            if hasattr(app_iter, 'close'):
                app_iter.close()
        
        # Format headers for Netlify
        response_headers = {}
        for header_name, header_value in response_data['headers']:
            response_headers[header_name] = header_value
        
        # Extract status code
        status_code = int(response_data['status'].split()[0])
        
        return {
            'statusCode': status_code,
            'headers': response_headers,
            'body': response_body,
            'isBase64Encoded': False
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': f'{{"error": "Internal server error", "details": "{str(e)}", "traceback": "{error_details.replace('"', '\\\\\\\\\\\\\\\\\'')}"\'}}'
        }