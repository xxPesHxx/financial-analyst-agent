import pytest
import os
import sys

def run_evaluation():
    print("Rakieta startuje...")
    
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
    ret_code = pytest.main(["-q", "tests/test_main.py"], plugins=[plugin])
    
    total = plugin.passed + plugin.failed
    success_rate = (plugin.passed / total * 100) if total > 0 else 0
    
    report_content = f"""# Raport z ewaluacji

**Testy:** {total}
**Zaliczone:** {plugin.passed}
**Niezaliczone:** {plugin.failed}
**Współczynnik sukcesu:** {success_rate:.2f}%

## Wyniki
| Test ID | Status |
| :--- | :--- |
"""
    for nodeid, status in plugin.results:
        name = nodeid.split("::")[-1]
        report_content += f"| {name} | {status} |\n"

    if os.path.exists("app.log"):
        with open("app.log", "r") as f:
            log_lines = len(f.readlines())
        report_content += f"\n## Obserwowalność\nLogi aplikacji zawierają {log_lines} wpisów.\n"
    else:
        report_content += "\n## Obserwowalność\nLogi aplikacji nie zostały znalezione (może być spowodowane brakiem ruchu).\n"

    with open("raport.md", "w") as f:
        f.write(report_content)
    
    print(f"Ewaluacja zakończona. Raport zapisany do raport.md. Współczynnik sukcesu: {success_rate:.2f}%")

if __name__ == "__main__":
    run_evaluation()
