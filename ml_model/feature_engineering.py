import pandas as pd
# Import data_ingestion to allow for individual testing of this module
# In a real pipeline, preprocess_data would receive a DataFrame directly.
from data_ingestion import get_historical_sales_data

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the raw sales data for time series forecasting.
    This function takes the raw sales DataFrame (from data_ingestion)
    and transforms it into a format suitable for the forecasting models (Prophet/ARIMA).

    Key preprocessing steps include:
    1. Ensuring the 'date' column is in datetime format.
    2. Grouping data by 'pincode' and 'item' to create individual time series.
    3. Resampling each series to a daily frequency, filling missing days with zero sales.
       This is crucial for time series models that expect continuous data.
    4. Renaming columns to 'ds' (date) and 'y' (sales quantity) as required by Prophet.
    5. Adding additional time-based features (e.g., day of week, month, weekend flag)
       that can be used as regressors in some models or for deeper analysis.

    Args:
        df (pd.DataFrame): Raw sales DataFrame with at least 'date', 'pincode',
                           'item', and 'sales_quantity' columns.

    Returns:
        pd.DataFrame: A preprocessed DataFrame suitable for Prophet/ARIMA models.
                      It will contain 'ds', 'y', 'pincode', 'item', and
                      additional time-based features.
    """
    print("Starting data preprocessing...")

    if df.empty:
        print("Input DataFrame is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    # Ensure 'date' column is datetime
    df['date'] = pd.to_datetime(df['date'])

    processed_data_list = []

    # Iterate through each unique combination of pincode and item
    # Each combination represents a distinct time series to be forecasted.
    unique_combinations = df[['pincode', 'item']].drop_duplicates()

    for index, row in unique_combinations.iterrows():
        pincode = row['pincode']
        item = row['item']

        # Filter data for the current pincode-item combination
        group_df = df[(df['pincode'] == pincode) & (df['item'] == item)].copy()

        if group_df.empty:
            continue # Skip if no data for this combination

        # Set 'date' as index for resampling, then resample to daily frequency.
        # Fill any missing dates within the range with 0 sales, assuming no sales means 0 demand.
        # This is critical for creating a continuous time series.
        daily_sales = group_df.set_index('date')['sales_quantity'].resample('D').sum().reset_index()

        # Rename columns for Prophet compatibility ('ds' for datetime, 'y' for value)
        daily_sales = daily_sales.rename(columns={'date': 'ds', 'sales_quantity': 'y'})

        # Add pincode and item back to the DataFrame for identification
        daily_sales['pincode'] = pincode
        daily_sales['item'] = item

        # --- Add additional time-based features ---
        # These features can be used as exogenous variables (regressors) in models
        # like Prophet or for more complex ARIMA models.
        daily_sales['day_of_week'] = daily_sales['ds'].dt.dayofweek # Monday=0, Sunday=6
        daily_sales['day_of_year'] = daily_sales['ds'].dt.dayofyear
        daily_sales['month'] = daily_sales['ds'].dt.month
        daily_sales['quarter'] = daily_sales['ds'].dt.quarter
        daily_sales['is_weekend'] = (daily_sales['ds'].dt.dayofweek >= 5).astype(int) # 1 if weekend, 0 otherwise

        # You could add holiday indicators here if you have a holiday calendar
        # daily_sales['is_holiday'] = daily_sales['ds'].isin(your_holiday_list).astype(int)

        processed_data_list.append(daily_sales)

    if not processed_data_list:
        print("No data was processed. Check input DataFrame and unique combinations.")
        return pd.DataFrame()

    # Concatenate all processed individual time series into a single DataFrame
    processed_df = pd.concat(processed_data_list, ignore_index=True)

    print(f"Data preprocessing complete. Final DataFrame shape: {processed_df.shape}")
    return processed_df

# This block allows you to test the script individually.
# When you run `python feature_engineering.py` from your terminal,
# the code within this block will execute.
if __name__ == "__main__":
    # --- Hardcoded Inputs for Individual Testing ---
    # First, simulate or get some raw data using data_ingestion.py's function
    print("--- Running data_ingestion.py to get raw data for testing ---")
    raw_sales_df_for_test = get_historical_sales_data("2024-01-01", "2024-03-31")
    print("\nRaw data head:")
    print(raw_sales_df_for_test.head())
    print(f"Raw data shape: {raw_sales_df_for_test.shape}")

    # Now, pass the raw data to the preprocess_data function
    print("\n--- Running preprocess_data on the raw data ---")
    preprocessed_sales_df = preprocess_data(raw_sales_df_for_test)

    # Print a preview of the preprocessed data
    print("\n--- Sample of Preprocessed Sales Data (from individual test) ---")
    print(preprocessed_sales_df.head())

    # Print data types to verify new columns and correct types
    print("\n--- Data Types of Preprocessed Data (from individual test) ---")
    print(preprocessed_sales_df.dtypes)

    # Verify the number of unique combinations and the structure
    print(f"\nUnique Pincode-Item Combinations in Preprocessed Data: {preprocessed_sales_df[['pincode', 'item']].drop_duplicates().shape[0]}")
    print(f"Total records in Preprocessed Data: {preprocessed_sales_df.shape[0]}")

    # Example: Check a specific series
    print("\n--- Checking a specific series (e.g., Milk in Pincode 110037) ---")
    milk_110037_series = preprocessed_sales_df[
        (preprocessed_sales_df['pincode'] == '110037') &
        (preprocessed_sales_df['item'] == 'Milk')
    ].sort_values('ds')
    print(milk_110037_series.head())
    print(f"Length of Milk in 110037 series: {len(milk_110037_series)}")
    # Check for continuity (should be daily without gaps)
    if not milk_110037_series.empty:
        time_diffs = milk_110037_series['ds'].diff().dropna()
        if (time_diffs == pd.Timedelta(days=1)).all():
            print("Series for Milk in 110037 is continuous daily.")
        else:
            print("Series for Milk in 110037 has gaps or irregular frequency.")
