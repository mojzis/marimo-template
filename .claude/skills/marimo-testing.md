---
description: "Testing and debugging marimo notebooks by running them as Python scripts. Use when validating notebook functionality, catching errors, or automating notebook execution."
---

# Marimo Notebook Testing & Debugging

This skill provides comprehensive guidelines for testing and debugging marimo notebooks by running them as Python scripts, validating their functionality, and catching errors before deployment.

## When to Use This Skill

Use this skill when:
- Testing marimo notebooks to ensure they run without errors
- Debugging issues in marimo notebook execution
- Running marimo notebooks as part of CI/CD pipelines
- Validating notebook functionality before sharing
- Automating notebook execution for data processing
- The user wants to run `.py` files containing `@app.cell` decorators as scripts

## Core Marimo Execution Modes

Marimo notebooks can be executed in several ways:

### 1. Interactive Mode (Development)
```bash
marimo edit notebook.py
```
Opens the notebook in a web browser for interactive development.

### 2. Run Mode (View-Only)
```bash
marimo run notebook.py
```
Runs the notebook in view-only mode, useful for dashboards and presentations.

### 3. Script Mode (Testing/Automation)
```bash
python notebook.py
```
Executes the notebook as a Python script - **THIS IS THE KEY FOR TESTING**.

## Running Marimo Notebooks as Python Scripts

### Basic Execution Pattern

Every marimo notebook can be run directly as a Python script:

```bash
# Simple execution
python notebook.py

# With arguments (if notebook accepts them)
python notebook.py --arg1 value1 --arg2 value2

# Capture output
python notebook.py > output.log 2>&1

# Exit code checking
python notebook.py && echo "Success!" || echo "Failed!"
```

### How It Works

Marimo notebooks are valid Python files that contain:
```python
import marimo

__generated_with = "0.x.x"
app = marimo.App()

@app.cell
def __():
    import marimo as mo
    return mo,

@app.cell
def __(mo):
    mo.md("Hello, marimo!")
    return

if __name__ == "__main__":
    app.run()
```

When run as `python notebook.py`, the `app.run()` at the bottom executes all cells in dependency order.

## Testing Strategies

### Strategy 1: Simple Smoke Test

**Goal**: Verify the notebook runs without errors.

```bash
#!/bin/bash
# test_notebook.sh

NOTEBOOK="analysis.py"

echo "Testing $NOTEBOOK..."
if python "$NOTEBOOK" > /dev/null 2>&1; then
    echo "✅ $NOTEBOOK passed"
    exit 0
else
    echo "❌ $NOTEBOOK failed"
    exit 1
fi
```

### Strategy 2: Output Validation

**Goal**: Verify the notebook produces expected output.

```bash
#!/bin/bash
# test_with_output.sh

NOTEBOOK="data_pipeline.py"
OUTPUT=$(python "$NOTEBOOK" 2>&1)
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Execution succeeded"

    # Validate output contains expected strings
    if echo "$OUTPUT" | grep -q "Processing complete"; then
        echo "✅ Output validation passed"
    else
        echo "❌ Expected output not found"
        exit 1
    fi
else
    echo "❌ Execution failed with exit code $EXIT_CODE"
    echo "$OUTPUT"
    exit 1
fi
```

### Strategy 3: Timeout Protection

**Goal**: Prevent notebooks from hanging indefinitely.

```bash
#!/bin/bash
# test_with_timeout.sh

NOTEBOOK="expensive_computation.py"
TIMEOUT=300  # 5 minutes

echo "Running $NOTEBOOK with ${TIMEOUT}s timeout..."
timeout $TIMEOUT python "$NOTEBOOK"
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Completed successfully"
elif [ $EXIT_CODE -eq 124 ]; then
    echo "❌ Timeout after ${TIMEOUT}s"
    exit 1
else
    echo "❌ Failed with exit code $EXIT_CODE"
    exit 1
fi
```

### Strategy 4: Multiple Notebook Testing

**Goal**: Test multiple notebooks in batch.

```bash
#!/bin/bash
# test_all_notebooks.sh

NOTEBOOKS_DIR="notebooks"
FAILED=0

echo "Testing all notebooks in $NOTEBOOKS_DIR..."
for notebook in "$NOTEBOOKS_DIR"/*.py; do
    if [ -f "$notebook" ]; then
        echo -n "Testing $(basename "$notebook")... "
        if python "$notebook" > /dev/null 2>&1; then
            echo "✅"
        else
            echo "❌"
            ((FAILED++))
        fi
    fi
done

if [ $FAILED -eq 0 ]; then
    echo "✅ All notebooks passed"
    exit 0
else
    echo "❌ $FAILED notebook(s) failed"
    exit 1
fi
```

## Debugging Techniques

### 1. Verbose Error Output

Run with full error output to see detailed tracebacks:

```bash
# Show all errors
python notebook.py 2>&1 | tee debug.log

# Python verbose mode
python -v notebook.py 2>&1 | tee verbose.log
```

### 2. Cell-by-Cell Debugging

If a notebook fails, identify which cell is causing the issue:

**Pattern A: Add debug prints**
```python
@app.cell
def __():
    print("DEBUG: Starting cell 1")
    import marimo as mo
    import pandas as pd
    print("DEBUG: Imports successful")
    return mo, pd
```

**Pattern B: Wrap cells in try/except**
```python
@app.cell
def __(mo):
    try:
        result = expensive_computation()
        message = f"✅ Success: {result}"
    except Exception as e:
        message = f"❌ Error in computation: {str(e)}"
        print(f"DEBUG: Full traceback:", file=sys.stderr)
        import traceback
        traceback.print_exc()

    mo.md(message)
    return result,
```

### 3. Dependency Validation

Ensure all required packages are installed:

```bash
#!/bin/bash
# validate_dependencies.sh

# Extract imports from notebook
python -c "
import ast
import sys

with open('$1', 'r') as f:
    tree = ast.parse(f.read())

imports = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        for alias in node.names:
            imports.add(alias.name.split('.')[0])
    elif isinstance(node, ast.ImportFrom):
        if node.module:
            imports.add(node.module.split('.')[0])

for imp in sorted(imports):
    if imp not in ['marimo', '__future__']:
        print(imp)
" | while read package; do
    python -c "import $package" 2>/dev/null || echo "❌ Missing: $package"
done
```

### 4. Interactive Debugging with PDB

Add breakpoints for interactive debugging:

```python
@app.cell
def __(data):
    # Add breakpoint
    import pdb; pdb.set_trace()

    processed = transform_data(data)
    return processed,
```

Run the notebook and interact with the debugger:
```bash
python notebook.py
# Debugger will pause at breakpoint
```

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/test-notebooks.yml
name: Test Marimo Notebooks

on: [push, pull_request]

jobs:
  test-notebooks:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install marimo pandas numpy
        # Add other dependencies

    - name: Test all notebooks
      run: |
        for notebook in notebooks/*.py; do
          echo "Testing $notebook..."
          timeout 300 python "$notebook" || exit 1
        done

    - name: Upload logs on failure
      if: failure()
      uses: actions/upload-artifact@v3
      with:
        name: notebook-logs
        path: '*.log'
```

### Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash

echo "Testing marimo notebooks..."
for notebook in notebooks/*.py; do
    if [ -f "$notebook" ]; then
        echo "Testing $(basename "$notebook")..."
        if ! timeout 60 python "$notebook" > /dev/null 2>&1; then
            echo "❌ Notebook test failed: $notebook"
            echo "Fix the notebook or skip with: git commit --no-verify"
            exit 1
        fi
    fi
done

echo "✅ All notebooks passed"
```

## Advanced Testing Patterns

### Pattern 1: Parameterized Testing

Create notebooks that accept parameters:

```python
# notebook.py
import marimo
import sys

app = marimo.App()

@app.cell
def __():
    import marimo as mo
    # Get command-line argument or use default
    data_file = sys.argv[1] if len(sys.argv) > 1 else "default.csv"
    return mo, data_file

@app.cell
def __(mo, data_file):
    import pandas as pd
    df = pd.read_csv(data_file)
    mo.md(f"Loaded {len(df)} rows from {data_file}")
    return df,

if __name__ == "__main__":
    app.run()
```

Test with different parameters:
```bash
python notebook.py data1.csv
python notebook.py data2.csv
python notebook.py data3.csv
```

### Pattern 2: Conditional Execution

Skip expensive operations during testing:

```python
@app.cell
def __():
    import os
    import marimo as mo

    # Check if in test mode
    IS_TEST = os.getenv('MARIMO_TEST_MODE', 'false') == 'true'
    return mo, IS_TEST

@app.cell
def __(mo, IS_TEST):
    if IS_TEST:
        # Use small sample data
        data = generate_sample_data(n=100)
        mo.md("⚠️ Running in TEST mode with sample data")
    else:
        # Use full dataset
        data = load_full_dataset()
        mo.md("Running with full dataset")

    return data,
```

Run in test mode:
```bash
MARIMO_TEST_MODE=true python notebook.py
```

### Pattern 3: Assertion-Based Testing

Add assertions to validate results:

```python
@app.cell
def __(df, mo):
    # Validate data quality
    try:
        assert len(df) > 0, "DataFrame is empty"
        assert not df.isnull().all().any(), "Column with all nulls found"
        assert df['price'].min() >= 0, "Negative prices found"

        message = "✅ Data validation passed"
    except AssertionError as e:
        message = f"❌ Validation failed: {str(e)}"
        raise  # Re-raise to fail the script

    mo.md(message)
    return
```

### Pattern 4: Output File Validation

Generate output files and validate them:

```python
@app.cell
def __(results, mo):
    import json

    # Save results
    output_file = "results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f)

    # Validate output
    with open(output_file, 'r') as f:
        loaded = json.load(f)

    assert loaded == results, "Output file validation failed"

    mo.md(f"✅ Results saved to {output_file}")
    return
```

## Common Issues and Solutions

### Issue 1: Import Errors

**Problem**: Notebook runs in interactive mode but fails as script.

**Solution**: Ensure all imports are in cells, not at module level:

```python
# ❌ WRONG: Module-level import (outside cells)
import pandas as pd

@app.cell
def __():
    import marimo as mo
    return mo,

# ✅ CORRECT: All imports inside cells
@app.cell
def __():
    import marimo as mo
    import pandas as pd
    return mo, pd
```

### Issue 2: Cell Dependencies Not Resolved

**Problem**: Cells execute in wrong order causing NameErrors.

**Solution**: Ensure proper dependency declaration:

```python
# ✅ CORRECT: Declare dependencies in function signature
@app.cell
def __(mo, df):  # Depends on mo and df from previous cells
    result = process(df)
    mo.md("Processed")
    return result,
```

### Issue 3: Display Output in Scripts

**Problem**: Display output doesn't show when running as script.

**Solution**: Add explicit prints for script mode:

```python
@app.cell
def __(mo, result):
    import sys

    message = f"Processing complete: {result}"

    # Show in marimo UI
    mo.md(message)

    # Also print for script mode
    if not sys.stdout.isatty():
        print(message)

    return
```

### Issue 4: Resource Cleanup

**Problem**: Notebooks don't clean up resources (files, connections).

**Solution**: Use context managers or explicit cleanup:

```python
@app.cell
def __(mo):
    try:
        # Open connection
        conn = database.connect()
        data = conn.query("SELECT * FROM table")
        message = "✅ Data loaded"
    except Exception as e:
        message = f"❌ Error: {e}"
        data = None
    finally:
        # Always cleanup
        if 'conn' in locals():
            conn.close()

    mo.md(message)
    return data,
```

## Best Practices for Testable Notebooks

### 1. Make Notebooks Deterministic

```python
# Set random seeds
import random
import numpy as np

random.seed(42)
np.random.seed(42)
```

### 2. Separate Configuration from Logic

```python
@app.cell
def __():
    # Configuration cell
    CONFIG = {
        'data_path': 'data.csv',
        'output_path': 'results.json',
        'sample_size': 1000,
        'random_seed': 42
    }
    return CONFIG,

@app.cell
def __(CONFIG):
    # Use configuration
    df = pd.read_csv(CONFIG['data_path'])
    return df,
```

### 3. Add Logging

```python
@app.cell
def __():
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    return logger,

@app.cell
def __(logger, data):
    logger.info(f"Processing {len(data)} rows")
    result = process(data)
    logger.info("Processing complete")
    return result,
```

### 4. Document Expected Behavior

```python
@app.cell
def __(mo):
    mo.md("""
    # Data Processing Pipeline

    ## Expected Inputs
    - `data.csv`: Input data with columns [id, value, timestamp]

    ## Expected Outputs
    - `results.json`: Processed results
    - Console: Summary statistics

    ## Expected Behavior
    - Filters out null values
    - Aggregates by timestamp
    - Saves results to JSON

    ## Testing
    Run as script: `python pipeline.py`
    Expected exit code: 0
    Expected runtime: < 30 seconds
    """)
    return
```

## Quick Reference Commands

```bash
# Basic testing
python notebook.py

# Test with timeout
timeout 300 python notebook.py

# Test with error output
python notebook.py 2>&1 | tee error.log

# Test all notebooks
for nb in *.py; do python "$nb" || exit 1; done

# Test with environment variable
MARIMO_TEST_MODE=true python notebook.py

# Test with parameters
python notebook.py --input data.csv --output results.json

# Validate dependencies
python -m pip check

# Check syntax without running
python -m py_compile notebook.py
```

## Instructions for Claude

When helping users test and debug marimo notebooks:

1. **Always suggest running as Python script first**: `python notebook.py` is the fastest way to test
2. **Add proper error handling**: Wrap risky operations in try/except with informative messages
3. **Use timeouts**: Prevent hanging notebooks with `timeout` command
4. **Validate incrementally**: Test cells individually before running entire notebook
5. **Check dependencies**: Ensure all required packages are installed
6. **Make notebooks deterministic**: Set random seeds, use fixed parameters
7. **Add logging**: Help users understand execution flow
8. **Test in CI/CD**: Encourage automated testing in pipelines

## Remember

- **Script execution is key**: `python notebook.py` runs all cells in dependency order
- **Exit codes matter**: Non-zero exit code indicates failure
- **Timeouts prevent hangs**: Always use timeout for automated testing
- **Error handling is critical**: Catch and report errors clearly
- **Dependencies must be explicit**: All imports inside cells
- **Testing early prevents issues**: Test notebooks before sharing

Following these patterns ensures marimo notebooks are robust, testable, and production-ready.
