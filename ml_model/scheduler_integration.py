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
try:
    from prediction import predict_next_day_demand
    from data_ingestion import get_historical_sales_data # To get a list of active pincode-item combinations
except ImportError as e:
    logging.error(f"Could not import required modules: {e}")
    print(f"Warning: Could not import required modules: {e}")
    # Fallback function if prediction module is not available
    def predict_next_day_demand(pincode, item, model_type='prophet'):
        return {
            "pincode": pincode,
            "item": item,
            "predicted_demand": 0,
            "restock_quantity": 0,
            "status": "Error: Prediction module not available"
        }

def run_daily_predictions():
    """
    This function simulates the core logic of a daily scheduled task.
    It identifies all active pincode-item combinations and triggers
    next-day demand predictions for each using the ML models.

    In a real system, the generated predictions would then be stored
    in a database (e.g., MongoDB) for the 'Restocking Engine' to access
    and use for automatic replenishment.
    """
    try:
        logging.info(f"[{date.today()} {os.getenv('TZ', 'UTC')}] Starting daily ML predictions for tomorrow's demand...")

        # Step 1: Identify all unique active pincode-item combinations.
        # In a production system, you might query a dedicated 'stores' or 'inventory'
        # collection in your database to get the list of currently active stores and their SKUs,
        # rather than inferring from recent sales data.
        # For this example, we'll fetch a small window of recent sales data to get combinations.
        logging.info("Fetching recent sales data to identify active pincode-item combinations...")
        try:
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
            return []

        logging.info(f"Found {len(unique_combinations)} unique pincode-item combinations to process.")

        all_predictions_results = []
        successful_predictions = 0
        failed_predictions = 0

        # Step 2: Iterate through each combination and generate predictions
        for index, row in unique_combinations.iterrows():
            pincode = row['pincode']
            item = row['item']

            try:
                # You can choose which model type to use here, or dynamically based on
                # recent model evaluation results (e.g., use the better performing model).
                # For simplicity, let's use Prophet as the primary model.
                prediction_result = predict_next_day_demand(pincode, item, model_type='prophet')

                all_predictions_results.append(prediction_result)

                # Track success/failure counts
                if prediction_result["status"] == "Success":
                    successful_predictions += 1
                else:
                    failed_predictions += 1

                # Step 3: (Conceptual) Store the prediction results in a database.
                # This is where the output of the ML model becomes an input for the
                # 'Restocking Engine' or the 'Admin Dashboard'.
                # Example (pseudo-code):
                # if prediction_result["status"] == "Success":
                #     save_prediction_to_database(prediction_result)
                # else:
                #     log_prediction_failure(prediction_result)
                logging.info(f"  - Processed {pincode}, {item}: Status - {prediction_result['status']}")

            except Exception as e:
                logging.error(f"Error processing prediction for {pincode}, {item}: {e}")
                failed_predictions += 1
                # Add a failed result to maintain consistency
                all_predictions_results.append({
                    "pincode": pincode,
                    "item": item,
                    "predicted_demand": 0,
                    "restock_quantity": 0,
                    "status": f"Error: {str(e)}"
                })

        logging.info(f"Daily predictions complete. Total {len(all_predictions_results)} predictions attempted.")
        logging.info(f"Successful: {successful_predictions}, Failed: {failed_predictions}")
        
        # You might want to return or log all_predictions_results for further processing/auditing
        return all_predictions_results

    except Exception as e:
        logging.error(f"Unexpected error in daily predictions: {e}", exc_info=True)
        return []

def run_batch_predictions(pincode_item_list=None, model_type='prophet'):
    """
    Run predictions for a specific list of pincode-item combinations.
    This can be useful for testing or for running predictions on demand.
    
    Args:
        pincode_item_list: List of tuples [(pincode, item), ...] or None to use default
        model_type: Type of model to use ('prophet' or 'arima')
    
    Returns:
        List of prediction results
    """
    try:
        if pincode_item_list is None:
            # Use default combinations
            pincode_item_list = [
                ('110037', 'Milk'),
                ('110037', 'Bread'),
                ('400092', 'Milk'),
                ('400092', 'Eggs'),
                ('400053', 'Apples')
            ]
        
        logging.info(f"Running batch predictions for {len(pincode_item_list)} combinations using {model_type} model...")
        
        results = []
        for pincode, item in pincode_item_list:
            try:
                result = predict_next_day_demand(pincode, item, model_type=model_type)
                results.append(result)
                logging.info(f"  - {pincode}, {item}: {result['status']}")
            except Exception as e:
                logging.error(f"Error predicting for {pincode}, {item}: {e}")
                results.append({
                    "pincode": pincode,
                    "item": item,
                    "predicted_demand": 0,
                    "restock_quantity": 0,
                    "status": f"Error: {str(e)}"
                })
        
        return results
        
    except Exception as e:
        logging.error(f"Error in batch predictions: {e}")
        return []

# This block allows you to test the script individually.
# When you run `python scheduler_integration.py` from your terminal,
# the code within this block will execute, simulating one run of the daily job.
if __name__ == "__main__":
    print("--- Running scheduler_integration.py individually (simulating a scheduled task) ---")
    print("This will fetch recent data, identify combinations, and run predictions.")
    print("Ensure you have run `model_training.py` at least once beforehand.")

    try:
        # Call the main function that performs the daily predictions
        results = run_daily_predictions()

        print("\n--- Summary of Individual Run Results ---")
        if results:
            successful = sum(1 for r in results if r['status'] == 'Success')
            failed = len(results) - successful
            
            print(f"Total predictions: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")
            
            print("\nSample Results:")
            for i, res in enumerate(results[:5]):  # Show first 5 results
                print(f"  {i+1}. Pincode: {res['pincode']}, Item: {res['item']}")
                print(f"     Predicted: {res['predicted_demand']}, Restock: {res['restock_quantity']}")
                print(f"     Status: {res['status']}")
        else:
            print("No predictions were generated.")

        # Test batch predictions
        print("\n--- Testing Batch Predictions ---")
        batch_results = run_batch_predictions()
        if batch_results:
            print(f"Batch predictions completed: {len(batch_results)} results")
            for res in batch_results[:3]:  # Show first 3
                print(f"  - {res['pincode']}, {res['item']}: {res['status']}")

    except Exception as e:
        print(f"Error during individual testing: {e}")

    print("\nIndividual testing of scheduler_integration.py complete.")
    print("In a real deployment, this script would be invoked by an external scheduler (e.g., cron, Celery Beat).")
