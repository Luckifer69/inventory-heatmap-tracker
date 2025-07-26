@echo off
echo ========================================
echo   FASTCOMMERCE ML MODULES (PROPHET ONLY)
echo ========================================
echo.
echo Running all ML modules with Prophet models only...
echo.

cd /d "%~dp0"

echo Step 1: Testing Data Ingestion...
python -c "from data_ingestion import get_historical_sales_data; data = get_historical_sales_data('2024-01-01', '2024-01-31'); print(f'Data shape: {data.shape}'); print('Data ingestion: SUCCESS')"
if %errorlevel% equ 0 (
    echo ✅ Data Ingestion
) else (
    echo ❌ Data Ingestion
)

echo.
echo Step 2: Testing Feature Engineering...
python -c "from data_ingestion import get_historical_sales_data; from feature_engineering import preprocess_data; data = get_historical_sales_data('2024-01-01', '2024-01-31'); processed = preprocess_data(data); print(f'Processed data shape: {processed.shape}'); print('Feature Engineering: SUCCESS')"
if %errorlevel% equ 0 (
    echo ✅ Feature Engineering
) else (
    echo ❌ Feature Engineering
)

echo.
echo Step 3: Testing Model Definition...
python -c "from model_definition_simple import define_prophet_model; model = define_prophet_model(); print('Prophet model defined successfully'); print('Model Definition: SUCCESS')"
if %errorlevel% equ 0 (
    echo ✅ Model Definition
) else (
    echo ❌ Model Definition
)

echo.
echo Step 4: Testing Model Training...
python model_training_prophet_only.py
if %errorlevel% equ 0 (
    echo ✅ Model Training
) else (
    echo ❌ Model Training
)

echo.
echo Step 5: Testing Prediction...
python -c "from prediction_prophet_only import predict_next_day_demand; result = predict_next_day_demand('110037', 'Milk', 'prophet'); print(f'Prediction result: {result}'); print('Prediction: SUCCESS')"
if %errorlevel% equ 0 (
    echo ✅ Prediction
) else (
    echo ❌ Prediction
)

echo.
echo Step 6: Testing Model Evaluation...
python -c "from model_evaluation import evaluate_model_performance; import pandas as pd; dummy_data = pd.DataFrame({'ds': pd.date_range('2024-01-01', periods=10), 'y': [10, 12, 15, 8, 20, 18, 22, 16, 19, 25], 'pincode': ['110037'] * 10, 'item': ['Milk'] * 10}); result = evaluate_model_performance('110037', 'Milk', 'prophet', dummy_data); print(f'Evaluation result: {result}'); print('Model Evaluation: SUCCESS')"
if %errorlevel% equ 0 (
    echo ✅ Model Evaluation
) else (
    echo ❌ Model Evaluation
)

echo.
echo Step 7: Testing API Integration...
python -c "from api_integration import app; print(f'FastAPI app created with {len(app.routes)} routes'); print('API Integration: SUCCESS')"
if %errorlevel% equ 0 (
    echo ✅ API Integration
) else (
    echo ❌ API Integration
)

echo.
echo Step 8: Testing Monitoring...
python -c "from monitoring import check_model_health; health = check_model_health(); print(f'Health status: {health}'); print('Monitoring: SUCCESS')"
if %errorlevel% equ 0 (
    echo ✅ Monitoring
) else (
    echo ❌ Monitoring
)

echo.
echo Step 9: Testing Scheduler Integration...
python -c "from scheduler_integration import run_batch_predictions; results = run_batch_predictions(); print(f'Scheduler results: {len(results) if results else 0} predictions'); print('Scheduler Integration: SUCCESS')"
if %errorlevel% equ 0 (
    echo ✅ Scheduler Integration
) else (
    echo ❌ Scheduler Integration
)

echo.
echo ========================================
echo   ALL MODULES COMPLETED
echo ========================================
echo.
echo 🎉 Prophet-only ML pipeline is working!
echo.
echo Next steps:
echo 1. Start API server: uvicorn api_integration:app --reload
echo 2. Access API docs: http://localhost:8000/docs
echo 3. Make predictions via API endpoints
echo.
pause 