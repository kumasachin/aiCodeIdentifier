# Code Analysis Tool - Command Line Script
# Developer: Personal Project  
# Started: Summer 2025
# Description: Batch analyze code files and generate HTML reports
# TODO: Clean up the report generation - it's a bit messy right now

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from jinja2 import Environment, FileSystemLoader
import ast

def load_training_data():
    """
    Load training examples for the model.
    This is pretty basic - just some examples I collected manually.
    Should probably expand this with more diverse samples.
    """
    # Some sample code I put together for training
    examples = pd.DataFrame({
        'code': [
            'def calculate_something(x, y):\n    # Basic calculation\n    return x * y + 1',
            '# Auto-generated helper\ndef process_input(data):\n    return data.strip().upper()'
        ],
        'label': ['human', 'ai']  # Simple labels for now
    })
    return examples

def analyze_code_features(code_text, file_ext='.py'):
    """
    Pull out features from code that might be useful.
    Added support for more file types based on what I've been working with.
    Still experimenting with what features work best.
    """
    if not code_text:
        return [0, 0, 0, 0, 0]  # Default values if no code
    
    # Basic metrics
    char_count = len(code_text)
    line_count = code_text.count('\n') + 1
    
    # Comments (different styles for different languages)
    comment_count = 0
    if file_ext in ['.py']:
        comment_count = code_text.count('#')
    elif file_ext in ['.js', '.ts', '.tsx', '.jsx', '.java', '.cpp', '.c', '.cs']:
        comment_count = code_text.count('//') + code_text.count('/*')
    elif file_ext in ['.php']:
        comment_count = code_text.count('#') + code_text.count('//')
    elif file_ext in ['.rb']:
        comment_count = code_text.count('#')
    
    # Function definitions (language-dependent)
    function_count = 0
    if file_ext == '.py':
        function_count = code_text.count('def ')
    elif file_ext in ['.js', '.ts', '.tsx', '.jsx']:
        function_count = code_text.count('function ') + code_text.count(') => ')
    
    # Control structures (mostly universal)
    control_structures = (code_text.count('if ') + code_text.count('for ') + 
                         code_text.count('while ') + code_text.count('try'))
    
    return [char_count, line_count, comment_count, function_count, control_structures]

def train_classifier_model(dataframe):
    """
    Train our classification model using the provided data.
    Using RandomForest because it seems to work reasonably well.
    """
    # Extract features from each code sample
    feature_matrix = dataframe['code'].apply(analyze_code_features).tolist()
    labels = dataframe['label']
    
    # Set up and train the classifier
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(feature_matrix, labels)
    
    return model

def scan_directory_for_code(directory_path):
    """
    Look through a directory and find code files to analyze.
    Updated to handle more file types that I've been working with.
    """
    # File extensions we can handle
    supported_files = ['.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.cpp', '.c', '.cs', '.php', '.rb']
    
    code_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_name, file_extension = os.path.splitext(file)
            
            if file_extension.lower() in supported_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    
                    # Get relative path for cleaner display
                    relative_path = os.path.relpath(file_path, directory_path)
                    
                    code_features = analyze_code_features(file_content, file_extension)
                    
                    code_files.append({
                        'filename': relative_path,
                        'content': file_content,
                        'features': code_features,
                        'file_type': file_extension,
                        'size': len(file_content)
                    })
                    
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
    
    return code_files

def generate_html_report(analysis_results, output_file='analysis_report.html'):
    """
    Create an HTML report with the analysis results.
    This template is pretty basic - could definitely be improved.
    """
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Code Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; margin-bottom: 20px; }
        .file-result { border: 1px solid #ddd; margin: 10px 0; padding: 15px; }
        .human { border-left: 5px solid green; }
        .ai { border-left: 5px solid red; }
        .stats { background: #f9f9f9; padding: 10px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Code Analysis Results</h1>
        <p>Generated on: {{ timestamp }}</p>
    </div>
    
    <div class="stats">
        <h2>Summary</h2>
        <p>Total files analyzed: {{ total_files }}</p>
        <p>Human-written: {{ human_count }}</p>
        <p>AI-generated: {{ ai_count }}</p>
    </div>
    
    <h2>Individual File Results</h2>
    {% for result in results %}
    <div class="file-result {{ result.prediction }}">
        <h3>{{ result.filename }}</h3>
        <p><strong>Prediction:</strong> {{ result.prediction|title }}</p>
        <p><strong>Confidence:</strong> {{ result.confidence }}%</p>
        <p><strong>File type:</strong> {{ result.file_type }}</p>
        <p><strong>Size:</strong> {{ result.size }} characters</p>
    </div>
    {% endfor %}
</body>
</html>
    """
    
    from jinja2 import Template
    template = Template(html_template)
    
    # Calculate summary stats
    total_files = len(analysis_results)
    human_count = len([r for r in analysis_results if r.get('prediction') == 'human'])
    ai_count = len([r for r in analysis_results if r.get('prediction') == 'ai'])
    
    from datetime import datetime
    
    html_content = template.render(
        results=analysis_results,
        total_files=total_files,
        human_count=human_count,
        ai_count=ai_count,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Report saved to {output_file}")

def main():
    """
    Main function - ties everything together.
    Could probably be organized better but it works for now.
    """
    print("Starting code analysis...")
    
    # Train the model with our sample data
    training_data = load_training_data()
    classifier = train_classifier_model(training_data)
    
    # Ask user for directory to analyze
    target_directory = input("Enter directory path to analyze (or press Enter for current directory): ").strip()
    if not target_directory:
        target_directory = '.'
    
    if not os.path.exists(target_directory):
        print(f"Directory {target_directory} not found!")
        return
    
    print(f"Scanning {target_directory} for code files...")
    
    # Find and analyze code files
    code_files = scan_directory_for_code(target_directory)
    
    if not code_files:
        print("No supported code files found!")
        return
    
    print(f"Found {len(code_files)} code files. Analyzing...")
    
    # Analyze each file
    results = []
    for file_info in code_files:
        try:
            prediction = classifier.predict([file_info['features']])[0]
            probabilities = classifier.predict_proba([file_info['features']])[0]
            confidence = max(probabilities) * 100
            
            results.append({
                'filename': file_info['filename'],
                'prediction': prediction,
                'confidence': round(confidence, 1),
                'file_type': file_info['file_type'],
                'size': file_info['size']
            })
            
            print(f"  {file_info['filename']}: {prediction} ({confidence:.1f}% confidence)")
            
        except Exception as e:
            print(f"Error analyzing {file_info['filename']}: {e}")
            continue
    
    # Generate HTML report
    print("\nGenerating HTML report...")
    generate_html_report(results)
    
    print("Analysis complete!")

if __name__ == '__main__':
    main()
