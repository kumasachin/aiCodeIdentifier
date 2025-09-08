#!/usr/bin/env python3
"""
Test Runner Script for AI Code Identifier

This script provides an easy way to run all tests with various options.

Usage:
    python run_tests.py           # Run all tests
    python run_tests.py -v        # Run with verbose output
    python run_tests.py -c        # Run with coverage report
    python run_tests.py -h        # Show help
"""

import subprocess
import sys
import os
import argparse


def run_command(command, description):
    """Run a command and handle the output"""
    print(f"\n🔍 {description}")
    print("=" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run tests for AI Code Identifier')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Run tests with verbose output')
    parser.add_argument('-c', '--coverage', action='store_true',
                       help='Run tests with coverage report')
    parser.add_argument('-f', '--fast', action='store_true',
                       help='Run tests without coverage (faster)')
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Activate virtual environment and run tests
    base_command = "source .venv/bin/activate && "
    
    if args.coverage:
        # Run with coverage
        test_command = (base_command + 
                       "python -m pytest test_code_analyzer.py "
                       "--cov=app --cov=main --cov-report=html --cov-report=term")
        success = run_command(test_command, "Running tests with coverage")
        
        if success:
            print(f"\n📊 Coverage report saved to: {script_dir}/htmlcov/index.html")
            
    elif args.fast:
        # Run without coverage (fastest)
        test_command = base_command + "python test_code_analyzer.py"
        success = run_command(test_command, "Running tests (fast mode)")
        
    else:
        # Default: run with pytest
        verbose_flag = "-v" if args.verbose else ""
        test_command = base_command + f"python -m pytest test_code_analyzer.py {verbose_flag}"
        success = run_command(test_command, "Running tests")
    
    # Summary
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Your code is working correctly.")
    else:
        print("❌ Some tests failed. Please check the output above.")
        sys.exit(1)
    
    # Additional info
    print("\n📝 Test Information:")
    print("  • Total test cases: 24")
    print("  • Test categories:")
    print("    - Feature extraction (Python, TypeScript, React)")
    print("    - Machine learning model setup and training")
    print("    - Code analysis and prediction")
    print("    - Flask route handlers")
    print("    - Error handling and edge cases")
    print("    - Directory scanning and file processing")
    
    if args.coverage:
        print("  • Coverage reports generated in 'htmlcov/' directory")


if __name__ == '__main__':
    main()
