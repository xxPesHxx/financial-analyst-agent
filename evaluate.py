import pytest
import os
import sys

def run_evaluation():
    print("Running evaluation...")
    
    # Run pytest and capture results using a simple plugin
    class ReportPlugin:
        def __init__(self):
            self.passed = 0
            self.failed = 0
            self.results = []

        def pytest_runtest_logreport(self, report):
            if report.when == 'call':
                status = "PASS" if report.passed else "FAIL"
                self.results.append((report.nodeid, status))
                if report.passed:
                    self.passed += 1
                else:
                    self.failed += 1

    plugin = ReportPlugin()
    # Run tests in tests/test_main.py
    # -q: quiet
    ret_code = pytest.main(["-q", "tests/test_main.py"], plugins=[plugin])
    
    total = plugin.passed + plugin.failed
    success_rate = (plugin.passed / total * 100) if total > 0 else 0
    
    report_content = f"""# Evaluation Report

**Total Tests:** {total}
**Passed:** {plugin.passed}
**Failed:** {plugin.failed}
**Success Rate:** {success_rate:.2f}%

## Detailed Results
| Test ID | Status |
| :--- | :--- |
"""
    for nodeid, status in plugin.results:
        # Clean up nodeid
        name = nodeid.split("::")[-1]
        report_content += f"| {name} | {status} |\n"

    # Observability check: Check if app.log exists and has entries
    if os.path.exists("app.log"):
        with open("app.log", "r") as f:
            log_lines = len(f.readlines())
        report_content += f"\n## Observability\nApp log contains {log_lines} entries.\n"
    else:
        report_content += "\n## Observability\nApp log not found (could be due to no traffic yet).\n"

    with open("evaluation_report.md", "w") as f:
        f.write(report_content)
    
    print(f"Evaluation complete. Report saved to evaluation_report.md. Success rate: {success_rate:.2f}%")

if __name__ == "__main__":
    run_evaluation()
