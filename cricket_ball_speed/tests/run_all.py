"""
Master Test Runner.
Executes all unit and integration tests for the cricket ball speed measurement system.
"""

import unittest
import sys
import os

if __name__ == "__main__":
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(tests_dir, ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("=" * 70)
    print("RUNNING CRICKET BALL SPEED SYSTEM TEST SUITE")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=tests_dir, pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures:  {len(result.failures)}")
    print(f"Errors:    {len(result.errors)}")
    print(f"Result:    {'ALL PASSED' if result.wasSuccessful() else 'FAILED'}")
    print("=" * 70)

    sys.exit(0 if result.wasSuccessful() else 1)
