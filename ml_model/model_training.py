import pandas as pd
from prophet import Prophet
import pmdarima as pm # For auto_arima, used to find optimal ARIMA parameters
import os
import joblib # For saving and loading Python objects, including trained models

# Import functions from other modules for individual testing
# In a full pipeline, data would be passed directly between modules.
from data_ingestion import get_historical_sales_data
from feature_engineering import preprocess_data

# Define a directory to save the trained models
MODEL_SAVE_DIR = "trained_models"

# Ensure the model save directory exists
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
print(f"Model save directory '{MODEL_SAVE_DIR}' ensured.")

def train_prophet_model(data: pd.DataFrame, pincode: str, item: str) -> Prophet:
    """
    Trains a Prophet model for a specific (pincode, item) combination.

    Args:
        data (pd.DataFrame): Preprocessed sales data. Must contain 'ds' (datetime),
                             'y' (sales quantity), 'pincode', and 'item' columns.
        pincode (str): The pincode for which to train the model.
        item (str): The item for which to train the model.

    Returns:
        Prophet: A trained Prophet model instance. Returns None if no data is available
                 for the given pincode-item combination.
    """
    print(f"\n--- Training Prophet model for Pincode: {pincode}, Item: {item} ---")

    # Filter data for the specific combination
    filtered_data = data[(data['pincode'] == pincode) & (data['item'] == item)].copy()

    if filtered_data.empty:
        print(f"No data found for Pincode: {pincode}, Item: {item}. Skipping Prophet training.")
        return None

    # Prophet requires 'ds' and 'y' columns
    prophet_df = filtered_data[['ds', 'y']].copy()

    # Initialize the Prophet model
    # Parameters are chosen based on typical sales data characteristics.
    # seasonality_mode='multiplicative' is often good for sales where seasonal
    # fluctuations grow with the overall trend.
    # changepoint_prior_scale controls trend flexibility.
    model = Prophet(
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05,
        daily_seasonality=False, # Assuming daily data, so no need for intra-day seasonality
        weekly_seasonality=True,
        yearly_seasonality=True
    )

    # If you added regressors in feature_engineering.py (e.g., 'is_weekend'),
    # you must add them to the Prophet model here before fitting.
    # Example:
    # if 'is_weekend' in filtered_data.columns:
    #     model.add_regressor('is_weekend')
    #     prophet_df['is_weekend'] = filtered_data['is_weekend'] # Ensure regressor is in df

    try:
        model.fit(prophet_df)
        print(f"Prophet model trained successfully for Pincode: {pincode}, Item: {item}.")

        # Save the trained model to disk
        model_filename = os.path.join(MODEL_SAVE_DIR, f'prophet_model_{pincode}_{item}.joblib')
        joblib.dump(model, model_filename)
        print(f"Prophet model saved to {model_filename}")
        return model
    except Exception as e:
        print(f"Error training Prophet model for {pincode}, {item}: {e}")
        return None

def train_arima_model(data: pd.DataFrame, pincode: str, item: str):
    """
    Trains an ARIMA model using pmdarima's auto_arima for a specific (pincode, item) combination.
    auto_arima automatically finds the best ARIMA (p,d,q)(P,D,Q,s) parameters
    by searching through combinations and evaluating them using information criteria (AIC, BIC).

    Args:
        data (pd.DataFrame): Preprocessed sales data. Must contain 'ds' (datetime),
                             'y' (sales quantity), 'pincode', and 'item' columns.
        pincode (str): The pincode for which to train the model.
        item (str): The item for which to train the model.

    Returns:
        pmdarima.ARIMA: A trained auto_arima model instance. Returns None if no data
                        is available for the given pincode-item combination.
    """
    print(f"\n--- Training ARIMA model for Pincode: {pincode}, Item: {item} ---")

    # Filter data for the specific combination
    filtered_data = data[(data['pincode'] == pincode) & (data['item'] == item)].copy()

    if filtered_data.empty:
        print(f"No data found for Pincode: {pincode}, Item: {item}. Skipping ARIMA training.")
        return None

    # ARIMA models typically work directly on the time series values ('y').
    # Ensure the data is sorted by date.
    series = filtered_data.sort_values('ds')['y'].values

    if len(series) < 10: # ARIMA needs a reasonable amount of data
        print(f"Not enough data points ({len(series)}) for ARIMA for {pincode}, {item}. Skipping.")
        return None

    try:
        # Use auto_arima to find the best ARIMA model parameters.
        # test='adf': Uses ADF test to determine the integration order 'd'.
        # m=7: Specifies a weekly seasonality (7 days).
        # seasonal=True: Enables seasonal components.
        # stepwise=True: Uses a stepwise algorithm to speed up the search.
        # trace=False: Set to True to see the search process (can be verbose).
        model = pm.auto_arima(series,
                              start_p=1, start_q=1,
                              test='adf',       # Use adftest to find optimal 'd'
                              max_p=5, max_q=5, # Maximum p and q
                              m=7,              # Seasonality (e.g., 7 for weekly)
                              d=None,           # Let model determine 'd'
                              seasonal=True,    # Enable seasonal ARIMA
                              start_P=0, start_Q=0,
                              max_P=2, max_Q=2,
                              D=None,           # Let model determine 'D'
                              trace=False,
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)

        print(f"ARIMA model trained successfully for Pincode: {pincode}, Item: {item}.")

        # Save the trained model to disk
        model_filename = os.path.join(MODEL_SAVE_DIR, f'arima_model_{pincode}_{item}.joblib')
        joblib.dump(model, model_filename)
        print(f"ARIMA model saved to {model_filename}")
        return model
    except Exception as e:
        print(f"Error training ARIMA model for {pincode}, {item}: {e}")
        return None

# This block allows you to test the script individually.
# When you run `python model_training.py` from your terminal,
# the code within this block will execute.
if __name__ == "__main__":
    print("--- Running model_training.py individually ---")

    # --- Hardcoded Inputs for Individual Testing ---
    # 1. Get raw data
    print("Step 1: Ingesting raw data...")
    raw_sales_df_for_test = get_historical_sales_data("2024-01-01", "2024-03-31")

    # 2. Preprocess the raw data
    print("\nStep 2: Preprocessing raw data...")
    preprocessed_sales_df = preprocess_data(raw_sales_df_for_test)

    if preprocessed_sales_df.empty:
        print("Preprocessed data is empty. Cannot proceed with training.")
    else:
        # 3. Identify unique (pincode, item) combinations to train models for
        unique_combinations = preprocessed_sales_df[['pincode', 'item']].drop_duplicates()
        print(f"\nStep 3: Found {len(unique_combinations)} unique pincode-item combinations for training.")

        trained_prophet_models_count = 0
        trained_arima_models_count = 0

        # 4. Iterate and train models for each combination
        for index, row in unique_combinations.iterrows():
            pincode = row['pincode']
            item = row['item']

            # Train Prophet model
            prophet_model = train_prophet_model(preprocessed_sales_df, pincode, item)
            if prophet_model:
                trained_prophet_models_count += 1

            # Train ARIMA model
            arima_model = train_arima_model(preprocessed_sales_df, pincode, item)
            if arima_model:
                trained_arima_models_count += 1

        print("\n--- Individual Training Summary ---")
        print(f"Total Prophet models trained and saved: {trained_prophet_models_count}")
        print(f"Total ARIMA models trained and saved: {trained_arima_models_count}")
        print(f"Models saved to: {os.path.abspath(MODEL_SAVE_DIR)}")
        print("\nIndividual testing of model_training.py complete.")
