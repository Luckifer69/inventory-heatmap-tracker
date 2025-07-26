import pandas as pd
import sys
import os
from datetime import date, timedelta
import logging

# Configure basic logging for the scheduler script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the parent directory of the current script to sys.path.
# This is crucial for allowing this script to import other modules
# (like 'prediction' and 'data_ingestion') from the same 'ml_model' directory.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the necessary functions from other modules
from prediction import predict_next_day_demand
from data_ingestion import get_historical_sales_data # To get a list of active pincode-item combinations

def run_daily_predictions():
    """
    This function simulates the core logic of a daily scheduled task.
    It identifies all active pincode-item combinations and triggers
    next-day demand predictions for each using the ML models.

    In a real system, the generated predictions would then be stored
    in a database (e.g., MongoDB) for the 'Restocking Engine' to access
    and use for automatic replenishment.
    """
    logging.info(f"[{date.today()} {os.getenv('TZ', 'UTC')}] Starting daily ML predictions for tomorrow's demand...")

    try:
        # Step 1: Identify all unique active pincode-item combinations.
        # In a production system, you might query a dedicated 'stores' or 'inventory'
        # collection in your database to get the list of currently active stores and their SKUs,
        # rather than inferring from recent sales data.
        # For this example, we'll fetch a small window of recent sales data to get combinations.
        logging.info("Fetching recent sales data to identify active pincode-item combinations...")
        recent_sales_data = get_historical_sales_data(
            (date.today() - timedelta(days=7)).strftime('%Y-%m-%d'), # Look at last 7 days
            date.today().strftime('%Y-%m-%d')
        )
        unique_combinations = recent_sales_data[['pincode', 'item']].drop_duplicates()

    except Exception as e:
        logging.error(f"Error identifying unique pincode-item combinations: {e}", exc_info=True)
        unique_combinations = pd.DataFrame(columns=['pincode', 'item']) # Fallback to empty DataFrame

    if unique_combinations.empty:
        logging.warning("No unique pincode-item combinations found to predict for. Exiting.")
        return

    logging.info(f"Found {len(unique_combinations)} unique pincode-item combinations to process.")

    all_predictions_results = []
    # Step 2: Iterate through each combination and generate predictions
    for index, row in unique_combinations.iterrows():
        pincode = row['pincode']
        item = row['item']

        # You can choose which model type to use here, or dynamically based on
        # recent model evaluation results (e.g., use the better performing model).
        # For simplicity, let's use Prophet as the primary model.
        prediction_result = predict_next_day_demand(pincode, item, model_type='prophet')

        all_predictions_results.append(prediction_result)

        # Step 3: (Conceptual) Store the prediction results in a database.
        # This is where the output of the ML model becomes an input for the
        # 'Restocking Engine' or the 'Admin Dashboard'.
        # Example (pseudo-code):
        # if prediction_result["status"] == "Success":
        #     save_prediction_to_database(prediction_result)
        # else:
        #     log_prediction_failure(prediction_result)
        logging.info(f"  - Processed {pincode}, {item}: Status - {prediction_result['status']}")

    logging.info(f"Daily predictions complete. Total {len(all_predictions_results)} predictions attempted.")
    # You might want to return or log all_predictions_results for further processing/auditing
    return all_predictions_results

# This block allows you to test the script individually.
# When you run `python scheduler_integration.py` from your terminal,
# the code within this block will execute, simulating one run of the daily job.
if __name__ == "__main__":
    print("--- Running scheduler_integration.py individually (simulating a scheduled task) ---")
    print("This will fetch recent data, identify combinations, and run predictions.")
    print("Ensure you have run `model_training.py` at least once beforehand.")

    # Call the main function that performs the daily predictions
    results = run_daily_predictions()

    print("\n--- Summary of Individual Run Results ---")
    if results:
        for res in results:
            print(f"Pincode: {res['pincode']}, Item: {res['item']}, Predicted: {res['predicted_demand']}, Restock: {res['restock_quantity']}, Status: {res['status']}")
    else:
        print("No predictions were generated.")

    print("\nIndividual testing of scheduler_integration.py complete.")
    print("In a real deployment, this script would be invoked by an external scheduler (e.g., cron, Celery Beat).")
