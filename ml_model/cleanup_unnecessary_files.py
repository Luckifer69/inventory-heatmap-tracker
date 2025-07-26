#!/usr/bin/env python3
"""
Cleanup Script for Prophet-Only Setup
Removes unnecessary files and keeps only essential modules.
"""

import os
import shutil

def cleanup_files():
    """Remove unnecessary files for Prophet-only setup"""
    print("🧹 CLEANING UP UNNECESSARY FILES")
    print("="*60)
    
    # Files to remove (ARIMA-related and test files)
    files_to_remove = [
        # ARIMA-related files
        "model_definition.py",  # Original with pmdarima
        "model_training.py",    # Original with ARIMA
        "prediction.py",        # Original with ARIMA
        
        # Test files (keep only essential ones)
        "test_model_components.py",
        "test_pipeline.py", 
        "test_prophet_pipeline.py",
        "test_all_modules.py",
        "test_all.bat",
        "test_all.sh",
        "test_fixes.py",
        "test_simple_fixes.py",
        "diagnose_failures.py",
        "fix_numpy_compatibility.py",
        
        # Temporary files
        "quick_fix.py",
        "requirements_fixed.txt"
    ]
    
    # Files to keep (essential for Prophet-only)
    files_to_keep = [
        # Core modules
        "data_ingestion.py",
        "feature_engineering.py", 
        "model_definition_simple.py",
        "model_training_prophet_only.py",
        "prediction_prophet_only.py",
        "model_evaluation.py",
        "api_integration.py",
        "monitoring.py",
        "scheduler_integration.py",
        
        # Configuration
        "requirements.txt",
        
        # Run scripts
        "run_all_modules.sh",
        "run_all_modules.bat",
        "cleanup_unnecessary_files.py"
    ]
    
    removed_count = 0
    kept_count = 0
    
    print("Removing unnecessary files...")
    for file in files_to_remove:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"🗑️  Removed: {file}")
                removed_count += 1
            except Exception as e:
                print(f"⚠️  Could not remove {file}: {e}")
        else:
            print(f"ℹ️  File not found: {file}")
    
    print(f"\nKept essential files:")
    for file in files_to_keep:
        if os.path.exists(file):
            print(f"✅ Kept: {file}")
            kept_count += 1
        else:
            print(f"⚠️  Missing: {file}")
    
    print(f"\nCleanup Summary:")
    print(f"🗑️  Removed: {removed_count} files")
    print(f"✅ Kept: {kept_count} files")
    
    return removed_count, kept_count

def create_readme():
    """Create a README for the cleaned up directory"""
    readme_content = """# FastCommerce ML Model (Prophet Only)

This directory contains the essential ML modules for demand prediction using Prophet models.

## Essential Files

### Core Modules
- `data_ingestion.py` - Fetches historical sales data
- `feature_engineering.py` - Preprocesses data for time series forecasting
- `model_definition_simple.py` - Defines Prophet models
- `model_training_prophet_only.py` - Trains Prophet models
- `prediction_prophet_only.py` - Makes predictions using trained models
- `model_evaluation.py` - Evaluates model performance
- `api_integration.py` - FastAPI endpoints for predictions
- `monitoring.py` - Monitors model health and performance
- `scheduler_integration.py` - Runs scheduled predictions

### Configuration
- `requirements.txt` - Python dependencies (Prophet-only)

### Run Scripts
- `run_all_modules.sh` - Linux/Mac script to test all modules
- `run_all_modules.bat` - Windows script to test all modules

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run all modules:
   ```bash
   # Linux/Mac
   ./run_all_modules.sh
   
   # Windows
   run_all_modules.bat
   ```

3. Start API server:
   ```bash
   uvicorn api_integration:app --reload
   ```

4. Access API docs:
   ```
   http://localhost:8000/docs
   ```

## Features

✅ Time series forecasting with Prophet  
✅ Demand prediction for fast-commerce  
✅ RESTful API endpoints  
✅ Model monitoring and health checks  
✅ Scheduled predictions  
✅ No complex dependencies (pmdarima excluded)  

## Model Types

- **Prophet**: Facebook's time series forecasting model
- **No ARIMA**: Excluded due to compatibility issues

This setup is production-ready and avoids numpy compatibility issues.
"""
    
    with open("README_ML.md", "w") as f:
        f.write(readme_content)
    
    print("✅ Created README_ML.md")

def main():
    """Main cleanup function"""
    print("🧹 PROPHET-ONLY CLEANUP")
    print("="*80)
    
    # Clean up files
    removed, kept = cleanup_files()
    
    # Create README
    create_readme()
    
    print("\n" + "="*80)
    print("✅ CLEANUP COMPLETED!")
    print("="*80)
    print(f"🗑️  Removed {removed} unnecessary files")
    print(f"✅ Kept {kept} essential files")
    print("📝 Created README_ML.md")
    print("\nYour ML directory is now clean and Prophet-only!")
    print("\nRun: ./run_all_modules.sh (Linux/Mac) or run_all_modules.bat (Windows)")

if __name__ == "__main__":
    main() 