# Import the preprocessing functions from the 'data_preprocessing_def' module
from data_preprocessing_def import handle_aka_name, handle_license, fill_missing_facility_type, drop_risk_rows, \
    process_city_column, process_state_column, process_zip_column, process_inspection_type_column, \
    fill_missing_violations, fill_missing_lat_long, extract_information_from_date


def data_preprocessing(data):
    # Apply each preprocessing function to the data

    # Step 1: Handle 'aka_name' column to fill missing values and standardize names
    data = handle_aka_name(data)

    # Step 2: Handle 'license_' column to fill missing values and ensure consistent format
    data = handle_license(data)

    # Step 3: Fill missing values in the 'facility_type' column based on other information
    data = fill_missing_facility_type(data)

    # Step 4: Drop rows with missing values in the 'risk' column
    data = drop_risk_rows(data)

    # Step 5: Process the 'city' column using external data from 'zip_code_database.csv'
    data = process_city_column(data, 'zip_code_database.csv')

    # Step 6: Process the 'state' column to ensure consistent format
    data = process_state_column(data)

    # Step 7: Process the 'zip' column to ensure consistent format
    data = process_zip_column(data)

    # Step 8: Process the 'inspection_type' column to ensure consistent format
    data = process_inspection_type_column(data)

    # Step 9: Fill missing values in the 'violations' column based on other information
    data = fill_missing_violations(data)

    # Step 10: Fill missing latitude and longitude values based on other information
    data = fill_missing_lat_long(data)

    # Step 11: Extracting information from the date column (inspection_date)
    data = extract_information_from_date(data)

    # Return the preprocessed DataFrame
    return data
