import pandas as pd
import sys
import os
from datetime import date, timedelta
import logging

# Configure basic logging for the monitoring script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the parent directory of the current script to sys.path.
# This is crucial for allowing this script to import other modules
# (like 'data_ingestion', 'feature_engineering', 'model_evaluation')
# from the same 'ml_model' directory.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the necessary functions from other modules
try:
    from data_ingestion import get_historical_sales_data
    from feature_engineering import preprocess_data
    from model_evaluation import evaluate_model_performance
except ImportError as e:
    logging.error(f"Could not import required modules: {e}")
    print(f"Warning: Could not import required modules: {e}")



def check_model_health():
    """
    Additional function to check the health of trained models.
    This can be used to verify that models are accessible and not corrupted.
    """
    try:
        import os
        import glob
        
        model_dir = "trained_models"
        if not os.path.exists(model_dir):
            logging.warning("Model directory does not exist.")
            return {"status": "no_models", "count": 0}
        
        model_files = glob.glob(os.path.join(model_dir, "*.joblib"))
        
        if not model_files:
            logging.warning("No model files found in model directory.")
            return {"status": "no_models", "count": 0}
        
        # Try to load a sample model to check if they're accessible
        try:
            import joblib
            sample_model = joblib.load(model_files[0])
            logging.info(f"Successfully loaded sample model: {os.path.basename(model_files[0])}")
            return {"status": "healthy", "count": len(model_files)}
        except Exception as e:
            logging.error(f"Error loading sample model: {e}")
            return {"status": "corrupted", "count": len(model_files), "error": str(e)}
            
    except Exception as e:
        logging.error(f"Error checking model health: {e}")
        return {"status": "error", "error": str(e)}

def monitor_model_performance_daily():
    """
    This function simulates the core logic of a daily scheduled monitoring task.
    It fetches recent actual sales data, preprocesses it, and then re-evaluates
    the performance of the trained ML models against this recent actual data.

    Key aspects of this monitoring include:
    - Tracking prediction accuracy (MAE, RMSE, MAPE) over time.
    - Identifying potential model degradation (when metrics worsen).
    - (Conceptual) Triggering alerts or notifications if performance falls below thresholds.
    - (Conceptual) Storing historical metric data for dashboard visualization.
    """
    try:
        logging.info(f"[{date.today()} {os.getenv('TZ', 'UTC')}] Starting daily ML model performance monitoring...")

        # Define the period for which to fetch actual data for evaluation.
        # We typically evaluate on data that has recently become available (e.g., yesterday's sales).
        # Using the last 30 days of actual data for evaluation provides a good rolling window.
        eval_end_date = date.today() - timedelta(days=1) # Evaluate up to yesterday's actuals
        eval_start_date = eval_end_date - timedelta(days=30) # Look back 30 days for evaluation data

        # Step 1: Fetch recent actual sales data for the evaluation period.
        logging.info(f"Fetching actual sales data for evaluation from {eval_start_date} to {eval_end_date}...")
        try:
            recent_actual_sales_df = get_historical_sales_data(
                eval_start_date.strftime('%Y-%m-%d'),
                eval_end_date.strftime('%Y-%m-%d')
            )
        except Exception as e:
            logging.error(f"Error fetching historical sales data: {e}")
            return

        if recent_actual_sales_df.empty:
            logging.warning("No recent actual sales data available for monitoring. Exiting.")
            return

        # Step 2: Preprocess the fetched actual data, similar to how training data was prepared.
        logging.info("Preprocessing recent actual sales data...")
        try:
            preprocessed_actual_sales = preprocess_data(recent_actual_sales_df)
        except Exception as e:
            logging.error(f"Error preprocessing actual sales data: {e}")
            return

        if preprocessed_actual_sales.empty:
            logging.warning("Preprocessed actual sales data is empty. Cannot perform evaluation. Exiting.")
            return

        # Step 3: Identify unique pincode-item combinations present in the evaluation data.
        try:
            unique_combinations = preprocessed_actual_sales[['pincode', 'item']].drop_duplicates()
            if unique_combinations.empty:
                logging.warning("No unique pincode-item combinations found in preprocessed actual data. Exiting.")
                return
        except Exception as e:
            logging.error(f"Error getting unique combinations: {e}")
            return

        logging.info(f"Found {len(unique_combinations)} unique pincode-item combinations to evaluate.")

        # Step 4: Iterate through each combination and evaluate both Prophet and ARIMA models.
        evaluation_results = []
        for index, row in unique_combinations.iterrows():
            pincode = row['pincode']
            item = row['item']

            logging.info(f"Evaluating models for Pincode: {pincode}, Item: {item}")

            try:
                # Evaluate Prophet model
                prophet_metrics = evaluate_model_performance(
                    pincode, item, model_type='prophet',
                    evaluation_data=preprocessed_actual_sales
                )
                logging.info(f"  Prophet Metrics: {prophet_metrics}")

                # Evaluate ARIMA model
                arima_metrics = evaluate_model_performance(
                    pincode, item, model_type='arima',
                    evaluation_data=preprocessed_actual_sales
                )
                logging.info(f"  ARIMA Metrics: {arima_metrics}")

                # Store results for potential alerting
                evaluation_results.append({
                    'pincode': pincode,
                    'item': item,
                    'prophet_metrics': prophet_metrics,
                    'arima_metrics': arima_metrics,
                    'evaluation_date': date.today()
                })

                # Step 5: (Conceptual) Implement alerting logic.
                # Define thresholds for metrics (e.g., if RMSE exceeds a certain value).
                # If a threshold is crossed, trigger an alert (email, Slack notification, etc.).
                # Example (pseudo-code):
                # THRESHOLD_RMSE = 15.0 # Example threshold
                # if prophet_metrics and prophet_metrics.get("RMSE", float('inf')) > THRESHOLD_RMSE:
                #     logging.critical(f"ALERT: Prophet model for {item} in {pincode} shows high RMSE ({prophet_metrics['RMSE']})!")
                #     send_notification_email(f"ML Model Alert: {item} in {pincode} degraded", "RMSE too high.")

                # Step 6: (Conceptual) Store evaluation metrics in a database.
                # This historical data can then be used to visualize model performance trends
                # on an Admin Dashboard.
                # Example (pseudo-code):
                # save_metrics_to_database(pincode, item, 'prophet', prophet_metrics, date.today())
                # save_metrics_to_database(pincode, item, 'arima', arima_metrics, date.today())

            except Exception as e:
                logging.error(f"Error evaluating models for {pincode}, {item}: {e}")
                continue

        logging.info(f"Daily monitoring complete. Evaluated {len(evaluation_results)} model combinations.")
        return evaluation_results

    except Exception as e:
        logging.error(f"An unexpected error occurred during model monitoring: {e}", exc_info=True)
        return None

# This block allows you to test the script individually.
# When you run `python monitoring.py` from your terminal,
# the code within this block will execute, simulating one run of the daily monitoring job.
if __name__ == "__main__":
    print("--- Running monitoring.py individually (simulating a scheduled monitoring task) ---")
    print("This will fetch recent actual data, preprocess it, and evaluate model performance.")
    print("Ensure you have run `model_training.py` at least once beforehand.")

    try:
        # Check model health first
        print("\n--- Checking Model Health ---")
        health_status = check_model_health()
        print(f"Model Health Status: {health_status}")

        # Call the main function that performs the daily monitoring
        results = monitor_model_performance_daily()

        if results:
            print(f"\n--- Monitoring Results Summary ---")
            print(f"Total evaluations completed: {len(results)}")
            
            # Show some sample results
            for i, result in enumerate(results[:3]):  # Show first 3 results
                print(f"\nResult {i+1}:")
                print(f"  Pincode: {result['pincode']}, Item: {result['item']}")
                print(f"  Prophet: {result['prophet_metrics']}")
                print(f"  ARIMA: {result['arima_metrics']}")
        else:
            print("No monitoring results generated.")

    except Exception as e:
        print(f"Error during individual testing: {e}")

    print("\nIndividual testing of monitoring.py complete.")
    print("In a real deployment, this script would be invoked by an external scheduler (e.g., cron, Celery Beat).")
