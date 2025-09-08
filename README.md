# Code Analysis Tool

A personal project for analyzing different coding patterns and styles in various programming languages.

## What it does

- Analyzes code files to identify different writing patterns
- Works with Python, JavaScript, TypeScript, React, Java, C++, C#, PHP, and Ruby
- Provides both web interface and command-line tools
- Generates detailed analysis reports

## Built with

- Python & Flask for the web app
- scikit-learn for machine learning
- Bootstrap for the frontend
- pandas for data handling

## How to use

### Web Interface

1. Install dependencies: `pip install -r requirements.txt`
2. Start the web app: `python app.py`
3. Open http://127.0.0.1:8080 in your browser
4. Upload files or paste code to analyze

### Command Line

1. Run the CLI tool: `python main.py`
2. Enter a directory path when prompted
3. Check the generated HTML report

## Testing

The project includes comprehensive unit tests covering all major functionality:

### Quick Test Run

```bash
python run_tests.py -f          # Fast mode (using unittest)
python run_tests.py             # Standard mode (using pytest)
python run_tests.py -v          # Verbose output
python run_tests.py -c          # With coverage report
```

### Test Coverage

- **24 test cases** covering:
  - Feature extraction (Python, TypeScript, React)
  - Machine learning model setup and training
  - Code analysis and prediction accuracy
  - Flask route handlers and API endpoints
  - Error handling and edge cases
  - Directory scanning and file processing

### Coverage Reports

Run `python run_tests.py -c` to generate detailed coverage reports in the `htmlcov/` directory.

Current test coverage: **55%** (app.py: 57%, main.py: 51%)

## Notes

This is a personal project I've been working on to explore different coding patterns. The analysis is based on various code features like comment styles, function definitions, complexity, etc. Still experimenting with improving accuracy!
