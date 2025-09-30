def handler(event, context):
    """
    Simple test function to debug Netlify deployment
    """
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'text/html',
        },
        'body': '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Netlify Test</title>
        </head>
        <body>
            <h1>🎉 Netlify Functions Working!</h1>
            <p>This is a simple test function.</p>
            <p>Event path: ''' + str(event.get('path', 'unknown')) + '''</p>
            <p>HTTP method: ''' + str(event.get('httpMethod', 'unknown')) + '''</p>
        </body>
        </html>
        '''
    }