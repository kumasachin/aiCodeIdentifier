# Code Analysis Tool - Personal Project
# Author: Developer
# Started: Summer 2025
# Notes: This started as a side project to help me analyze different coding styles
# Still working on improving the accuracy...

from flask import Flask, render_template, request, jsonify, send_file
import os
import tempfile
import zipfile
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import ast
import json
from datetime import datetime

app = Flask(__name__)

# Global variable for the ML model - keeping it here for now
classifier_model = None

def extract_features_from_code(source_code, file_ext='.py'):
    """
    This function tries to pull out useful patterns from code.
    I've been tweaking this for weeks to get better results.
    Added support for TypeScript/React after working on some web projects.
    """
    if not source_code.strip():
        return [0] * 12  # just return zeros if there's nothing there
    
    # Basic stuff first
    total_chars = len(source_code)
    num_lines = source_code.count('\n') + 1
    
    # Comments - different languages use different styles
    python_comments = source_code.count('#')
    js_comments = source_code.count('//')
    multiline_comments = source_code.count('/*')
    
    # Check what kind of file we're dealing with
    if file_ext.lower() in ['.tsx', '.jsx']:
        # React stuff
        jsx_tags = source_code.count('<') + source_code.count('/>')
        hooks_usage = (source_code.count('useState') + source_code.count('useEffect') + 
                      source_code.count('useContext') + source_code.count('useCallback'))
        imports = source_code.count('import ')
        exports = source_code.count('export ')
    elif file_ext.lower() in ['.ts', '.js']:
        # Regular TypeScript/JavaScript
        jsx_tags = 0
        hooks_usage = 0
        imports = source_code.count('import ')
        exports = source_code.count('export ')
    else:
        # Python and other stuff
        jsx_tags = 0
        hooks_usage = 0
        imports = source_code.count('import ')
        exports = source_code.count('export ')  # Python has this too sometimes
    
    # Count functions and classes - works across languages mostly
    function_count = (source_code.count('def ') +  # Python
                     source_code.count('function ') +  # JS/TS
                     source_code.count(') => ') +  # Arrow functions
                     source_code.count('const ') + source_code.count('let '))  # Variable declarations
    
    class_count = (source_code.count('class ') +  # Most languages
                  source_code.count('interface ') +  # TypeScript
                  source_code.count('type '))  # TypeScript types
    
    # Average line length - trying to avoid divide by zero errors
    if num_lines > 0:
        avg_line_length = total_chars / num_lines
    else:
        avg_line_length = 0
    
    # Look for complex stuff - loops, conditions, etc.
    complexity_stuff = (source_code.count('if ') + source_code.count('if(') +
                       source_code.count('for ') + source_code.count('for(') +
                       source_code.count('while ') + source_code.count('while(') +
                       source_code.count('try') + source_code.count('catch'))
    
    # Return all the features we extracted
    return [
        total_chars,
        python_comments + js_comments + multiline_comments,  # All comments combined
        function_count,
        class_count,
        imports,
        avg_line_length,
        jsx_tags,  # React stuff
        complexity_stuff,
        hooks_usage,  # React hooks
        exports,
        num_lines,
        len(source_code.split())  # Total words
    ]

def setup_ml_model():
    """
    Set up the machine learning model with some training data.
    This is pretty basic for now - I collected these examples manually.
    TODO: Get more training data to improve accuracy
    """
    # Training examples I put together
    sample_data = {
        'code': [
            # Some human-written examples I found
            'def calculate_sum(a, b):\n    """Calculate sum of two numbers"""\n    return a + b',
            
            # Another human example
            'def process_data(data):\n    # Process the input data\n    result = []\n    for item in data:\n        if item > 0:\n            result.append(item * 2)\n    return result',
            
            # TypeScript example from a project
            'interface User {\n  name: string;\n  age: number;\n}\n\nfunction greetUser(user: User): string {\n  return `Hello, ${user.name}!`;\n}',
            
            # React component I wrote
            'import React, { useState } from "react";\n\nconst Counter: React.FC = () => {\n  const [count, setCount] = useState(0);\n  \n  return (\n    <div>\n      <p>Count: {count}</p>\n      <button onClick={() => setCount(count + 1)}>+</button>\n    </div>\n  );\n};',
            
            # Some generated-looking code
            '# Auto-generated function\ndef auto_function():\n    return "generated"',
            
            # More generated style
            'const processArray = <T>(arr: T[]): T[] => arr.filter(Boolean).map(item => item);',
            
            # Generated React
            'export const Button = ({ onClick, children }: { onClick: () => void; children: React.ReactNode }) => <button onClick={onClick}>{children}</button>;',
            
            # Classic algorithm implementation
            'def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)',
        ],
        'label': ['human', 'human', 'human', 'human', 'ai', 'ai', 'ai', 'ai']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Extract features from training examples
    features_list = []
    for code_sample in df['code']:
        # Try to guess the file type based on content
        if 'import React' in code_sample or 'useState' in code_sample or '<' in code_sample:
            extension = '.tsx'
        elif 'interface ' in code_sample or ': string' in code_sample or ': number' in code_sample:
            extension = '.ts'
        else:
            extension = '.py'
        features = extract_features_from_code(code_sample, extension)
        features_list.append(features)
    
    labels = df['label']
    
    # Train the model - RandomForest seems to work pretty well
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
    model.fit(features_list, labels)
    
    return model

def analyze_code_sample(code_content, filename):
    """
    Analyze a single piece of code and return what we think about it.
    This is where the actual analysis happens.
    """
    if not code_content.strip():
        return None
    
    # Figure out what kind of file this is
    file_ext = os.path.splitext(filename)[1].lower()
    
    # Extract features from the code
    features = extract_features_from_code(code_content, file_ext)
    
    # Use our model to make a prediction
    prediction = classifier_model.predict([features])[0]
    probabilities = classifier_model.predict_proba([features])[0]
    confidence = max(probabilities)
    
    # Figure out which probability corresponds to which class
    # (the model classes might be in different order)
    if classifier_model.classes_[0] == 'ai':
        ai_prob = probabilities[0]
        human_prob = probabilities[1]
    else:
        ai_prob = probabilities[1]
        human_prob = probabilities[0]
    
    # Get a nice display name for the file type
    file_type = get_file_type_name(file_ext)
    
    return {
        'file': filename,
        'file_type': file_type,
        'prediction': prediction,
        'confidence': round(confidence * 100, 2),
        'ai_probability': round(ai_prob * 100, 2),
        'human_probability': round(human_prob * 100, 2),
        'lines_of_code': len(code_content.split('\n')),
        'character_count': len(code_content)
    }

def get_file_type_name(extension):
    """Helper function to get nice names for file types"""
    types = {
        '.py': 'Python',
        '.js': 'JavaScript', 
        '.ts': 'TypeScript',
        '.tsx': 'React TypeScript',
        '.jsx': 'React JavaScript',
        '.java': 'Java',
        '.cpp': 'C++',
        '.c': 'C',
        '.cs': 'C#',
        '.php': 'PHP',
        '.rb': 'Ruby'
    }
    return types.get(extension, 'Unknown')

# Route handlers start here

@app.route('/')
def home():
    """Main page - just render the template"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier_model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze', methods=['POST'])
def analyze_files():
    """Handle file uploads and analyze them"""
    try:
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files uploaded'})
        
        results = []
        for file in files:
            if file.filename == '':
                continue
                
            # Read the file content
            try:
                content = file.read().decode('utf-8')
            except UnicodeDecodeError:
                results.append({
                    'file': file.filename,
                    'error': 'Could not read file - encoding issue'
                })
                continue
            
            # Analyze the code
            analysis = analyze_code_sample(content, file.filename)
            if analysis:
                results.append(analysis)
            else:
                results.append({
                    'file': file.filename,
                    'error': 'File is empty or could not be analyzed'
                })
        
        # Calculate summary stats
        total_files = len(results)
        ai_count = len([r for r in results if r.get('prediction') == 'ai'])
        human_count = len([r for r in results if r.get('prediction') == 'human'])
        
        # Average confidence (only for successful analyses)
        successful_results = [r for r in results if 'confidence' in r]
        if successful_results:
            avg_confidence = sum(r['confidence'] for r in successful_results) / len(successful_results)
        else:
            avg_confidence = 0
        
        summary = {
            'total_files': total_files,
            'ai_predicted': ai_count,
            'human_predicted': human_count,
            'average_confidence': round(avg_confidence, 2)
        }
        
        return jsonify({'results': results, 'summary': summary})
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/analyze-folder', methods=['POST'])
def analyze_folder():
    """Handle folder uploads - similar to file analysis but for multiple files"""
    try:
        files = request.files.getlist('folder')
        if not files:
            return jsonify({'error': 'No folder selected'})
        
        # Filter for supported file types
        supported_extensions = {'.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.cpp', '.c', '.cs', '.php', '.rb'}
        
        results = []
        for file in files:
            if file.filename == '':
                continue
                
            # Check if it's a supported file type
            _, ext = os.path.splitext(file.filename)
            if ext.lower() not in supported_extensions:
                continue
                
            # Read and analyze the file
            try:
                content = file.read().decode('utf-8')
                analysis = analyze_code_sample(content, file.filename)
                if analysis:
                    results.append(analysis)
            except UnicodeDecodeError:
                results.append({
                    'file': file.filename,
                    'error': 'Could not read file - encoding issue'
                })
                continue
            except Exception as e:
                results.append({
                    'file': file.filename, 
                    'error': f'Analysis failed: {str(e)}'
                })
                continue
        
        if not results:
            return jsonify({'error': 'No supported code files found in folder'})
        
        # Summary calculations
        total_files = len(results)
        ai_count = len([r for r in results if r.get('prediction') == 'ai'])
        human_count = len([r for r in results if r.get('prediction') == 'human'])
        
        successful_results = [r for r in results if 'confidence' in r]
        if successful_results:
            avg_confidence = sum(r['confidence'] for r in successful_results) / len(successful_results)
        else:
            avg_confidence = 0
        
        summary = {
            'total_files': total_files,
            'ai_predicted': ai_count,
            'human_predicted': human_count,
            'average_confidence': round(avg_confidence, 2)
        }
        
        return jsonify({'results': results, 'summary': summary})
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    """Analyze code that was pasted directly into the text area"""
    try:
        data = request.get_json()
        code = data.get('code', '')
        filename = data.get('filename', 'input.py')
        
        if not code.strip():
            return jsonify({'error': 'No code provided'})
        
        # Analyze the pasted code
        analysis = analyze_code_sample(code, filename)
        if analysis:
            return jsonify(analysis)
        else:
            return jsonify({'error': 'Could not analyze the provided code'})
            
    except Exception as e:
        return jsonify({'error': str(e)})

# Initialize the model when the app starts
try:
    classifier_model = setup_ml_model()
    print("Model initialized successfully!")
except Exception as e:
    print(f"Error initializing model: {e}")
    classifier_model = None

if __name__ == '__main__':
    if classifier_model is None:
        print("Warning: Model not initialized, trying again...")
        try:
            classifier_model = setup_ml_model()
            print("Model initialized on startup!")
        except Exception as e:
            print(f"Failed to initialize model: {e}")
    
    # Run the Flask app
    # Use environment PORT for production deployment
    port = int(os.environ.get('PORT', 8080))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
