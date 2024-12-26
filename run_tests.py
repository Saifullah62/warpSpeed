#!/usr/bin/env python
import os
import sys
import pytest
import datetime
import json
from pathlib import Path
from typing import Dict, List, Any

class TestRunner:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_tests(self) -> Dict[str, Any]:
        """Run all tests and return the results."""
        # Create a results file name with timestamp
        results_file = self.output_dir / f"test_results_{self.timestamp}.json"
        
        # Configure pytest to collect results
        class ResultCollector:
            def __init__(self):
                self.results = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "summary": {},
                    "test_cases": []
                }

        collector = ResultCollector()

        class CustomPlugin:
            def __init__(self, collector):
                self.collector = collector
                self.start_time = None

            def pytest_sessionstart(self, session):
                self.start_time = datetime.datetime.now()

            def pytest_runtest_logreport(self, report):
                if report.when == "call":
                    result = {
                        "name": report.nodeid,
                        "outcome": report.outcome,
                        "duration": report.duration,
                    }
                    if hasattr(report, "longrepr"):
                        result["error"] = str(report.longrepr)
                    self.collector.results["test_cases"].append(result)

            def pytest_sessionfinish(self, session):
                end_time = datetime.datetime.now()
                duration = (end_time - self.start_time).total_seconds()
                self.collector.results["summary"] = {
                    "total": session.testscollected or 0,
                    "passed": (session.testscollected or 0) - (session.testsfailed or 0),
                    "failed": session.testsfailed or 0,
                    "duration": duration
                }

        # Run pytest with our custom plugin
        pytest.main([
            "tests",
            "-v",
            "--no-header",
            "--tb=short"
        ], plugins=[CustomPlugin(collector)])

        # Save results to file
        with open(results_file, 'w') as f:
            json.dump(collector.results, f, indent=2)

        return collector.results

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable report from the test results."""
        report = []
        report.append("Test Execution Report")
        report.append("===================")
        report.append(f"\nTimestamp: {results['timestamp']}")
        report.append("\nSummary:")
        report.append(f"- Total Tests: {results['summary']['total']}")
        report.append(f"- Passed: {results['summary']['passed']}")
        report.append(f"- Failed: {results['summary']['failed']}")
        report.append(f"- Duration: {results['summary']['duration']:.2f} seconds")

        report.append("\nDetailed Results:")
        report.append("----------------")
        for test in results['test_cases']:
            report.append(f"\n{test['name']}:")
            report.append(f"  Outcome: {test['outcome']}")
            report.append(f"  Duration: {test['duration']:.2f} seconds")
            if test.get('error'):
                report.append(f"  Error: {test['error']}")

        report_text = "\n".join(report)
        
        # Save the report
        report_file = self.output_dir / f"test_report_{self.timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
            
        return report_text

def main():
    runner = TestRunner()
    print("Starting test execution...")
    results = runner.run_tests()
    print("\nGenerating report...")
    report = runner.generate_report(results)
    print("\nTest execution completed!")
    print(f"\nResults saved in the 'reports' directory")
    
    # Print summary to console
    print(f"\nQuick Summary:")
    print(f"Total Tests: {results['summary']['total']}")
    print(f"Passed: {results['summary']['passed']}")
    print(f"Failed: {results['summary']['failed']}")

if __name__ == "__main__":
    main()
