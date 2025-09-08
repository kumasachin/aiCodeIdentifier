#!/usr/bin/env python3
"""
Unit Tests for AI Code Identifier
Tests for app.py and main.py methods

Test coverage:
- Feature extraction functions
- ML model setup and training
- Code analysis methods
- File type detection
- Route handlers (basic functionality)
- Error handling

Run with: python -m pytest test_code_analyzer.py -v
"""

import unittest
import pytest
import tempfile
import os
import json
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Import the modules we're testing
import app
import main


class TestFeatureExtraction(unittest.TestCase):
    """Test the feature extraction functions"""
    
    def test_extract_features_from_code_python(self):
        """Test feature extraction for Python code"""
        python_code = """
# This is a test function
def calculate_sum(a, b):
    '''Calculate sum of two numbers'''
    if a > 0 and b > 0:
        return a + b
    return 0

class Calculator:
    def multiply(self, x, y):
        return x * y
"""
        features = app.extract_features_from_code(python_code, '.py')
        
        # Should return 12 features
        self.assertEqual(len(features), 12)
        
        # Check some specific features
        self.assertGreater(features[0], 0)  # total_chars
        self.assertGreater(features[1], 0)  # comments
        self.assertGreater(features[2], 0)  # function_count
        self.assertGreater(features[3], 0)  # class_count
        self.assertGreater(features[10], 0)  # num_lines
    
    def test_extract_features_from_code_typescript(self):
        """Test feature extraction for TypeScript code"""
        ts_code = """
// TypeScript interface
import { SomeType } from './types';

interface User {
    name: string;
    age: number;
}

function greetUser(user: User): string {
    return `Hello, ${user.name}!`;
}

export { greetUser };
"""
        features = app.extract_features_from_code(ts_code, '.ts')
        
        self.assertEqual(len(features), 12)
        self.assertGreater(features[0], 0)  # total_chars
        self.assertGreater(features[1], 0)  # comments
        self.assertGreater(features[4], 0)  # imports/exports
        self.assertGreater(features[9], 0)  # exports
    
    def test_extract_features_from_code_react(self):
        """Test feature extraction for React/TSX code"""
        react_code = """
import React, { useState } from 'react';

const Counter: React.FC = () => {
    const [count, setCount] = useState(0);
    
    return (
        <div>
            <p>Count: {count}</p>
            <button onClick={() => setCount(count + 1)}>
                Increment
            </button>
        </div>
    );
};

export default Counter;
"""
        features = app.extract_features_from_code(react_code, '.tsx')
        
        self.assertEqual(len(features), 12)
        self.assertGreater(features[6], 0)  # jsx_tags
        self.assertGreater(features[8], 0)  # hooks_usage
        self.assertGreater(features[4], 0)  # imports
    
    def test_extract_features_empty_code(self):
        """Test feature extraction with empty code"""
        features = app.extract_features_from_code("", '.py')
        
        # Should return 12 zeros
        self.assertEqual(features, [0] * 12)
    
    def test_extract_features_whitespace_only(self):
        """Test feature extraction with whitespace only"""
        features = app.extract_features_from_code("   \n  \t  \n", '.py')
        
        # Should return 12 zeros for effectively empty code
        self.assertEqual(features, [0] * 12)


class TestMainFeatureExtraction(unittest.TestCase):
    """Test feature extraction in main.py"""
    
    def test_analyze_code_features_python(self):
        """Test main.py feature extraction for Python"""
        python_code = """
def test_function():
    # This is a comment
    if True:
        for i in range(10):
            print(i)
"""
        features = main.analyze_code_features(python_code, '.py')
        
        # Should return 5 features
        self.assertEqual(len(features), 5)
        
        # Check that all features are reasonable
        self.assertGreater(features[0], 0)  # char_count
        self.assertGreater(features[1], 0)  # line_count
        self.assertGreater(features[2], 0)  # comment_count
        self.assertGreater(features[3], 0)  # function_count
        self.assertGreater(features[4], 0)  # control_structures
    
    def test_analyze_code_features_javascript(self):
        """Test main.py feature extraction for JavaScript"""
        js_code = """
// JavaScript function
function processData(data) {
    if (data.length > 0) {
        for (let item of data) {
            console.log(item);
        }
    }
}

const arrowFunc = () => {
    return "hello";
};
"""
        features = main.analyze_code_features(js_code, '.js')
        
        self.assertEqual(len(features), 5)
        self.assertGreater(features[2], 0)  # comment_count
        self.assertGreater(features[3], 0)  # function_count
        self.assertGreater(features[4], 0)  # control_structures
    
    def test_analyze_code_features_empty(self):
        """Test main.py feature extraction with empty code"""
        features = main.analyze_code_features("", '.py')
        
        # Should return default values
        self.assertEqual(features, [0, 0, 0, 0, 0])
    
    def test_analyze_code_features_none(self):
        """Test main.py feature extraction with None input"""
        features = main.analyze_code_features(None, '.py')
        
        # Should return default values
        self.assertEqual(features, [0, 0, 0, 0, 0])


class TestModelSetup(unittest.TestCase):
    """Test machine learning model setup"""
    
    def test_setup_ml_model(self):
        """Test ML model initialization"""
        model = app.setup_ml_model()
        
        # Should return a trained RandomForestClassifier
        self.assertIsInstance(model, RandomForestClassifier)
        
        # Should have the correct classes
        expected_classes = ['ai', 'human']
        self.assertTrue(all(cls in model.classes_ for cls in expected_classes))
    
    def test_train_classifier_model(self):
        """Test main.py model training"""
        # Create sample training data
        sample_data = pd.DataFrame({
            'code': [
                'def human_func():\n    return "natural"',
                'def ai_func():\n    return "generated"'
            ],
            'label': ['human', 'ai']
        })
        
        model = main.train_classifier_model(sample_data)
        
        # Should return a trained model
        self.assertIsInstance(model, RandomForestClassifier)
        
        # Should be able to make predictions
        test_features = [[100, 5, 1, 1, 2]]  # Sample features
        prediction = model.predict(test_features)
        self.assertIn(prediction[0], ['human', 'ai'])


class TestCodeAnalysis(unittest.TestCase):
    """Test code analysis methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock the classifier model
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = ['human']
        self.mock_model.predict_proba.return_value = [[0.3, 0.7]]
        self.mock_model.classes_ = ['ai', 'human']
        
        # Patch the global classifier_model
        app.classifier_model = self.mock_model
    
    def test_analyze_code_sample_valid(self):
        """Test code analysis with valid input"""
        code = "def test_function():\n    return 'hello'"
        filename = "test.py"
        
        result = app.analyze_code_sample(code, filename)
        
        # Should return analysis results
        self.assertIsNotNone(result)
        self.assertEqual(result['file'], filename)
        self.assertEqual(result['file_type'], 'Python')
        self.assertEqual(result['prediction'], 'human')
        self.assertIn('confidence', result)
        self.assertIn('ai_probability', result)
        self.assertIn('human_probability', result)
    
    def test_analyze_code_sample_empty(self):
        """Test code analysis with empty input"""
        result = app.analyze_code_sample("", "test.py")
        
        # Should return None for empty code
        self.assertIsNone(result)
    
    def test_analyze_code_sample_whitespace(self):
        """Test code analysis with whitespace only"""
        result = app.analyze_code_sample("   \n  \t  ", "test.py")
        
        # Should return None for effectively empty code
        self.assertIsNone(result)
    
    def test_get_file_type_name(self):
        """Test file type name mapping"""
        test_cases = [
            ('.py', 'Python'),
            ('.js', 'JavaScript'),
            ('.ts', 'TypeScript'),
            ('.tsx', 'React TypeScript'),
            ('.jsx', 'React JavaScript'),
            ('.java', 'Java'),
            ('.unknown', 'Unknown')
        ]
        
        for extension, expected_name in test_cases:
            with self.subTest(extension=extension):
                self.assertEqual(app.get_file_type_name(extension), expected_name)


class TestDirectoryScanning(unittest.TestCase):
    """Test directory scanning functionality"""
    
    def test_scan_directory_for_code(self):
        """Test directory scanning with temporary files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_files = {
                'test.py': 'def hello():\n    return "world"',
                'test.js': 'function hello() {\n    return "world";\n}',
                'test.txt': 'This is not a code file',
                'subdir/test.ts': 'const hello = (): string => "world";'
            }
            
            # Create subdirectory
            os.makedirs(os.path.join(temp_dir, 'subdir'), exist_ok=True)
            
            # Write test files
            for filename, content in test_files.items():
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # Scan the directory
            code_files = main.scan_directory_for_code(temp_dir)
            
            # Should find Python, JS, and TS files but not TXT
            self.assertGreater(len(code_files), 0)
            
            # Check that we found the expected files
            found_extensions = [os.path.splitext(f['filename'])[1] for f in code_files]
            self.assertIn('.py', found_extensions)
            self.assertIn('.js', found_extensions)
            self.assertIn('.ts', found_extensions)
            self.assertNotIn('.txt', found_extensions)


class TestFlaskRoutes(unittest.TestCase):
    """Test Flask route handlers"""
    
    def setUp(self):
        """Set up test client"""
        app.app.config['TESTING'] = True
        self.client = app.app.test_client()
        
        # Mock the classifier model
        mock_model = MagicMock()
        mock_model.predict.return_value = ['human']
        mock_model.predict_proba.return_value = [[0.2, 0.8]]
        mock_model.classes_ = ['ai', 'human']
        app.classifier_model = mock_model
    
    def test_home_route(self):
        """Test the home page route"""
        response = self.client.get('/')
        
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<!DOCTYPE html>', response.data)
    
    def test_analyze_text_route(self):
        """Test the text analysis route"""
        test_data = {
            'code': 'def test():\n    return "hello"',
            'filename': 'test.py'
        }
        
        response = self.client.post('/analyze-text',
                                  data=json.dumps(test_data),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('prediction', data)
        self.assertIn('confidence', data)
    
    def test_analyze_text_route_empty(self):
        """Test text analysis with empty code"""
        test_data = {
            'code': '',
            'filename': 'test.py'
        }
        
        response = self.client.post('/analyze-text',
                                  data=json.dumps(test_data),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_analyze_files_route_no_files(self):
        """Test file analysis with no files uploaded"""
        response = self.client.post('/analyze')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'No files uploaded')


class TestErrorHandling(unittest.TestCase):
    """Test error handling in various scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Set up Flask test client
        app.app.config['TESTING'] = True
        self.client = app.app.test_client()
    
    def test_analyze_with_model_not_initialized(self):
        """Test behavior when ML model is not initialized"""
        # Temporarily set model to None
        original_model = app.classifier_model
        app.classifier_model = None
        
        try:
            test_data = {
                'code': 'def test():\n    return "hello"',
                'filename': 'test.py'
            }
            
            response = self.client.post('/analyze-text',
                                      data=json.dumps(test_data),
                                      content_type='application/json')
            
            # Should handle the error gracefully
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('error', data)
            
        finally:
            # Restore the original model
            app.classifier_model = original_model
    
    def test_feature_extraction_with_malformed_code(self):
        """Test feature extraction with malformed code"""
        malformed_code = "def incomplete_function(\n    # Missing closing parenthesis"
        
        # Should not crash
        features = app.extract_features_from_code(malformed_code, '.py')
        
        # Should still return 12 features
        self.assertEqual(len(features), 12)
        self.assertIsInstance(features[0], (int, float))


class TestDataHandling(unittest.TestCase):
    """Test data loading and handling functions"""
    
    def test_load_training_data(self):
        """Test training data loading in main.py"""
        data = main.load_training_data()
        
        # Should return a DataFrame
        self.assertIsInstance(data, pd.DataFrame)
        
        # Should have required columns
        self.assertIn('code', data.columns)
        self.assertIn('label', data.columns)
        
        # Should have some data
        self.assertGreater(len(data), 0)
    
    def test_training_data_format(self):
        """Test that training data has correct format"""
        data = main.load_training_data()
        
        # All labels should be either 'human' or 'ai'
        valid_labels = {'human', 'ai'}
        self.assertTrue(all(label in valid_labels for label in data['label']))
        
        # All code samples should be strings
        self.assertTrue(all(isinstance(code, str) for code in data['code']))


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
