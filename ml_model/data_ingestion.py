import pandas as pd
# from pymongo import MongoClient # Uncomment this line if you are connecting to MongoDB

def get_historical_sales_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical sales data.
    In a real-world scenario, this function would connect to a database (like MongoDB
    as suggested in the project PDF) and query actual sales records.

    For this project's demonstration and initial development, we will simulate
    daily sales data for a few sample pincodes and items.

    Args:
        start_date (str): The start date for data retrieval in 'YYYY-MM-DD' format.
        end_date (str): The end date for data retrieval in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the historical sales data.
                      It will have columns: 'date', 'pincode', 'item', 'sales_quantity'.
    """
    print(f"Fetching sales data from {start_date} to {end_date}...")

    # --- Placeholder for MongoDB Connection (Uncomment and configure for real use) ---
    # try:
    #     # Replace with your MongoDB connection string and database/collection names
    #     client = MongoClient('mongodb://localhost:27017/')
    #     db = client['fast_commerce_db']
    #     sales_collection = db['sales_data']
    #
    #     # Example query: Fetch sales within the date range
    #     # This query assumes your sales data has a 'timestamp' field
    #     query_results = sales_collection.find({
    #         'timestamp': {
    #             '$gte': pd.to_datetime(start_date),
    #             '$lte': pd.to_datetime(end_date)
    #         }
    #     })
    #     df = pd.DataFrame(list(query_results))
    #     # Ensure 'date' column is correctly formatted and rename if necessary
    #     df['date'] = pd.to_datetime(df['timestamp']).dt.date # Or just use timestamp if preferred
    #     df = df[['date', 'pincode', 'item', 'sales_quantity']] # Adjust column names as per your schema
    #
    # except Exception as e:
    #     print(f"Error connecting to MongoDB or fetching data: {e}")
    #     print("Proceeding with simulated data for demonstration.")
    #     df = _simulate_sales_data(start_date, end_date)
    # ---------------------------------------------------------------------------------

    # --- Simulated Data Generation (Used if MongoDB connection is not active or for testing) ---
    # This function generates synthetic sales data to allow the ML pipeline
    # to be developed and tested without a live database connection.
    df = _simulate_sales_data(start_date, end_date)
    # ---------------------------------------------------------------------------------

    print(f"Data ingestion complete. Fetched {df.shape[0]} records.")
    return df

def _simulate_sales_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Helper function to simulate sales data for demonstration purposes.
    This generates random daily sales for a few predefined items and pincodes.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    pincodes = ['110037', '400092', '400053'] # Sample pincodes
    items = ['Milk', 'Eggs', 'Bread', 'Apples', 'Bananas'] # Sample items

    data = []
    for date in dates:
        for pincode in pincodes:
            for item in items:
                import random
                # Simulate varying sales quantities
                sales = random.randint(10, 60)
                if date.dayofweek >= 5: # Increase sales on weekends
                    sales = random.randint(20, 80)
                if item == 'Milk' and pincode == '400092': # Higher demand for Milk in a specific pincode
                    sales += random.randint(10, 30)
                if item == 'Eggs' and pincode == '110037': # Lower demand for Eggs in another
                    sales = max(5, sales - random.randint(5, 20))

                data.append({'date': date, 'pincode': pincode, 'item': item, 'sales_quantity': sales})

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date']) # Ensure 'date' is datetime object
    return df

# This block allows you to test the script individually.
# When you run `python data_ingestion.py` from your terminal,
# the code within this block will execute.
if __name__ == "__main__":
    # --- Hardcoded Inputs for Individual Testing ---
    start_date_example = "2024-01-01"
    end_date_example = "2024-03-31"

    # Call the data ingestion function with the hardcoded dates
    sales_df = get_historical_sales_data(start_date_example, end_date_example)

    # Print a preview of the fetched/simulated data to verify
    print("\n--- Sample of Ingested Sales Data (from individual test) ---")
    print(sales_df.head())

    # Print data types to verify correct parsing
    print("\n--- Data Types of Ingested Data (from individual test) ---")
    print(sales_df.dtypes)

    # Verify the number of unique items and pincodes
    print(f"\nUnique Pincodes: {sales_df['pincode'].nunique()}")
    print(f"Unique Items: {sales_df['item'].nunique()}")
    print(f"Total records: {sales_df.shape[0]}")
