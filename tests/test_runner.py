#!/usr/bin/env python3
"""
Simple test runner for nanoPLM tests.

This script runs the test suite without requiring pytest.
Usage:
    python tests/test_runner.py                    # Run all tests
    python tests/test_runner.py --pattern "*collator*"  # Run specific tests
    python tests/test_runner.py --verbose         # Verbose output
"""

import sys
import os
import glob
import importlib.util
import traceback
from pathlib import Path
import time
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestRunner:
    """Simple test runner for nanoPLM."""

    def __init__(self, verbose=False, pattern="*"):
        self.verbose = verbose
        self.pattern = pattern
        self.results = {
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'total': 0
        }

    def log(self, message, level='info'):
        """Log a message."""
        if self.verbose or level in ['error', 'warning']:
            print(f"[{level.upper()}] {message}")

    def discover_tests(self):
        """Discover test files matching the pattern."""
        test_dir = Path(__file__).parent
        pattern = f"test_{self.pattern}.py"
        test_files = list(test_dir.glob(pattern))

        self.log(f"Found {len(test_files)} test files matching '{pattern}'")
        for test_file in test_files:
            self.log(f"  - {test_file.name}", 'info')

        return test_files

    def load_test_module(self, test_file):
        """Load a test module from file."""
        module_name = test_file.stem
        spec = importlib.util.spec_from_file_location(module_name, test_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        return spec, module

    def run_test_method(self, module, class_name, method_name):
        """Run a single test method."""
        try:
            test_class = getattr(module, class_name)
            test_instance = test_class()

            # Setup method if exists
            if hasattr(test_instance, 'setUp'):
                test_instance.setUp()

            # Run the test method
            method = getattr(test_instance, method_name)
            start_time = time.time()
            method()
            end_time = time.time()

            self.results['passed'] += 1
            self.log(f"✓ {class_name}.{method_name} ({end_time - start_time:.3f}s)", 'success')
            return True

        except Exception as e:
            self.results['failed'] += 1
            self.log(f"✗ {class_name}.{method_name}: {str(e)}", 'error')
            if self.verbose:
                traceback.print_exc()
            return False

    def run_test_class(self, module, class_name):
        """Run all test methods in a class."""
        test_class = getattr(module, class_name)
        test_instance = test_class()

        # Get all test methods
        test_methods = [method for method in dir(test_instance)
                       if method.startswith('test_') and callable(getattr(test_instance, method))]

        self.log(f"Running {len(test_methods)} tests in {class_name}")

        for method_name in test_methods:
            self.run_test_method(module, class_name, method_name)

    def run_test_file(self, test_file):
        """Run all tests in a test file."""
        self.log(f"Running tests from {test_file.name}")

        try:
            spec, module = self.load_test_module(test_file)
            spec.loader.exec_module(module)

            # Find test classes
            test_classes = [name for name in dir(module)
                          if name.startswith('Test') and hasattr(getattr(module, name), '__init__')]

            for class_name in test_classes:
                self.run_test_class(module, class_name)

        except Exception as e:
            self.results['errors'] += 1
            self.log(f"Error loading {test_file.name}: {str(e)}", 'error')
            if self.verbose:
                traceback.print_exc()

    def run_all_tests(self):
        """Run all discovered tests."""
        test_files = self.discover_tests()

        start_time = time.time()

        for test_file in test_files:
            self.run_test_file(test_file)

        end_time = time.time()

        # Print summary
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        print(f"Total tests: {self.results['total']}")
        print(f"Passed: {self.results['passed']}")
        print(f"Failed: {self.results['failed']}")
        print(f"Errors: {self.results['errors']}")
        print(".2f")
        print(f"Success rate: {(self.results['passed'] / max(1, self.results['total'])) * 100:.1f}%")

        return self.results['failed'] == 0 and self.results['errors'] == 0


def main():
    parser = argparse.ArgumentParser(description="Run nanoPLM tests")
    parser.add_argument("--pattern", default="*", help="Pattern to match test files (default: *)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--list", action="store_true", help="List test files without running")

    args = parser.parse_args()

    runner = TestRunner(verbose=args.verbose, pattern=args.pattern)

    if args.list:
        test_files = runner.discover_tests()
        return 0

    success = runner.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
