#!/usr/bin/env python3
"""Comprehensive project validation script."""

import ast
import sys
from pathlib import Path


def validate_syntax(file_path: Path) -> tuple[bool, str]:
    """Validate Python syntax of a file."""
    try:
        with open(file_path, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Run validation checks."""
    print("=" * 70)
    print("PROJECT VALIDATION REPORT")
    print("=" * 70)
    
    # Check critical files exist
    print("\n1. CRITICAL FILES CHECK:")
    critical_files = [
        "README.md",
        "LICENSE",
        "pyproject.toml",
        "configs/default.yaml",
        "configs/ablation.yaml",
        "scripts/train.py",
        "scripts/evaluate.py",
        "scripts/predict.py",
    ]
    
    all_exist = True
    for file in critical_files:
        path = Path(file)
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
        if not exists:
            all_exist = False
    
    # Check Python files syntax
    print("\n2. PYTHON SYNTAX VALIDATION:")
    python_files = list(Path("src").rglob("*.py")) + list(Path("scripts").rglob("*.py"))
    
    all_valid = True
    for py_file in sorted(python_files):
        valid, msg = validate_syntax(py_file)
        status = "✓" if valid else "✗"
        print(f"  {status} {py_file}")
        if not valid:
            print(f"      Error: {msg}")
            all_valid = False
    
    # Check README length
    print("\n3. README LENGTH CHECK:")
    readme_lines = len(open("README.md").readlines())
    readme_ok = readme_lines < 200
    status = "✓" if readme_ok else "✗"
    print(f"  {status} README.md: {readme_lines} lines (target: <200)")
    
    # Check YAML configs don't use scientific notation
    print("\n4. YAML CONFIG VALIDATION:")
    import re
    sci_notation_pattern = re.compile(r'\b\d+\.?\d*[eE][+-]?\d+\b')
    
    yaml_ok = True
    for yaml_file in Path("configs").glob("*.yaml"):
        content = yaml_file.read_text()
        if sci_notation_pattern.search(content):
            print(f"  ✗ {yaml_file}: Contains scientific notation")
            yaml_ok = False
        else:
            print(f"  ✓ {yaml_file}: No scientific notation")
    
    # Check LICENSE file
    print("\n5. LICENSE FILE CHECK:")
    license_path = Path("LICENSE")
    if license_path.exists():
        license_content = license_path.read_text()
        has_mit = "MIT License" in license_content
        has_copyright = "Copyright (c) 2026 Alireza Shojaei" in license_content
        
        license_ok = has_mit and has_copyright
        status = "✓" if license_ok else "✗"
        print(f"  {status} MIT License: {has_mit}")
        print(f"  {status} Copyright 2026: {has_copyright}")
    else:
        print("  ✗ LICENSE file missing")
        license_ok = False
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY:")
    print("=" * 70)
    
    checks = [
        ("Critical files exist", all_exist),
        ("Python syntax valid", all_valid),
        ("README length OK", readme_ok),
        ("YAML configs OK", yaml_ok),
        ("LICENSE file OK", license_ok),
    ]
    
    all_passed = all(passed for _, passed in checks)
    
    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check_name}")
    
    print("=" * 70)
    
    if all_passed:
        print("✓ ALL VALIDATION CHECKS PASSED")
        return 0
    else:
        print("✗ SOME VALIDATION CHECKS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
