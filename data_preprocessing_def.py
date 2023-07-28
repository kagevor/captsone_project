import pandas as pd
import numpy as np


def handle_aka_name(data):
    """
    Fill missing 'aka_name' values with the corresponding 'dba_name'.

    Parameters:
        data (pandas.DataFrame): The DataFrame containing the 'aka_name' and 'dba_name' columns.

    Returns:
        pandas.DataFrame: The DataFrame with missing 'aka_name' values filled using 'dba_name'.
    """
    data['aka_name'] = data['aka_name'].fillna(data['dba_name'])
    return data

def handle_license(data):
    """
    Handle new license numbers for rows with a license value of 0 or NaN.

    Parameters:
        data (pandas.DataFrame): The DataFrame containing the 'license_', 'dba_name', and 'address' columns.

    Returns:
        pandas.DataFrame: The DataFrame with new license numbers assigned and missing or 0 license numbers filled.
    """
    # Starting id for new license numbers
    new_license_id = int(data['license_'].max() + 1)

    # Get rows with license value of 0 or NaN, and select relevant columns
    license_zero = data[(data['license_'] == 0) | (data['license_'].isna())][['dba_name', 'address']].copy().drop_duplicates()

    # Assign new license numbers to rows with license value of 0 or NaN
    license_zero['new_license'] = range(new_license_id, new_license_id + len(license_zero))

    # Merge the DataFrame with the new license numbers to the original DataFrame
    data = data.merge(license_zero, on=['dba_name', 'address'], how='left')

    # Replace 0 with NaN in the 'license_' column
    data['license_'] = data['license_'].apply(lambda x: np.nan if x == 0 else x)

    # Fill missing 'license_' values with the newly assigned license numbers
    data['license_'] = data['license_'].fillna(value=data['new_license'])

    # Drop the temporary column 'new_license'
    data.drop(columns=['new_license'], inplace=True)

    return data

def fill_missing_facility_type(data):
    """
    Fill missing 'facility_type' values in the DataFrame.

    Parameters:
        data (pandas.DataFrame): The DataFrame containing the 'facility_type' column.

    Returns:
        pandas.DataFrame: The DataFrame with missing 'facility_type' values filled.
    """
    # Get the value counts of each facility type and sort in descending order
    frequencies_facility_type = data['facility_type'].value_counts(normalize=True)
    # Identify the most dominant facility type ('Restaurant') and its percentage
    dominant_facility_type = frequencies_facility_type.idxmax()
    dominant_percentage = frequencies_facility_type.max() * 100

    # Fill missing 'facility_type' values with the dominant facility type
    data['facility_type'].fillna(dominant_facility_type, inplace=True)

    return data



def drop_risk_rows(data):
    """
    Drop rows with NaN values and the "All" category from the 'risk' column.

    Parameters:
        data (pandas.DataFrame): The DataFrame containing the 'risk' column.

    Returns:
        pandas.DataFrame: The DataFrame with rows containing NaN or "All" in the 'risk' column dropped.
    """
    # Drop rows with NaN or "All" in the 'risk' column
    data = data[data['risk'].notna() & (data['risk'] != 'All')]

    return data


def process_city_column(data, zip_codes_file_path):
    """
    Process the 'city' column in the DataFrame.

    Parameters:
        data (pandas.DataFrame): The DataFrame containing the 'city' and 'state' columns.
        zip_codes_file_path (str): The file path to the CSV file containing the ZIP codes table.

    Returns:
        pandas.DataFrame: The DataFrame with processed 'city' column.
    """
    # Read the ZIP codes table from the CSV file
    zip_codes_df = pd.read_csv(zip_codes_file_path, dtype={'zip': float})

    # Merge data with the ZIP codes DataFrame to get city names based on ZIP codes
    data = pd.merge(data, zip_codes_df[['zip', 'primary_city']], on='zip', how='left')

    # Fill missing 'city' values with the 'primary_city' values
    data['city'] = data['city'].fillna(data['primary_city'])

    # Drop the temporary 'primary_city' column
    data.drop('primary_city', axis=1, inplace=True)

    # Fill missing 'city' values with 'CHICAGO' where 'state' is 'IL'
    data.loc[(data['city'].isna()) & (data['state'] == 'IL'), 'city'] = 'CHICAGO'

    # Capitalize the first letter of each city name
    data['city'] = data['city'].str.capitalize()

    # Standardize city names containing 'icago' (case-insensitive) to 'Chicago'
    data.loc[data['city'].str.contains('icago', case=False), 'city'] = 'Chicago'

    return data

def process_state_column(data):
    """
    Process the 'state' column in the DataFrame.

    Parameters:
        data (pandas.DataFrame): The DataFrame containing the 'state' and 'city' columns.

    Returns:
        pandas.DataFrame: The DataFrame with processed 'state' column and filtered data for 'IL'.
    """
    # Fill missing 'state' values with 'IL' where 'city' is 'CHICAGO'
    data.loc[(data['state'].isna()) & (data['city'] == 'CHICAGO'), 'state'] = 'IL'

    # Fill remaining missing 'state' values with 'Not Available'
    data['state'].fillna('Not Available', inplace=True)

    # Filter the data to retain only the rows with 'IL' in the 'state' column
    data = data[data['state'] == 'IL']

    # Drop the 'state' column
    data.drop('state', axis=1, inplace=True)

    return data

def process_zip_column(data):
    """
    Process the 'zip' column in the DataFrame.

    Parameters:
        data (pandas.DataFrame): The DataFrame containing the 'zip' column.

    Returns:
        pandas.DataFrame: The DataFrame with processed 'zip' column.
    """
    # Find the most frequent zip code
    most_frequent_zip = data['zip'].mode()[0]

    # Count the occurrences of missing 'zip' values for each city
    missing_zip_by_city = data[data['zip'].isna()]['city'].value_counts()

    # Fill missing 'zip' values with the most frequent zip code
    data['zip'].fillna(most_frequent_zip, inplace=True)

    # Convert the 'zip' column to the integer data type and then back to the string data type
    data['zip'] = data['zip'].astype(int)
    data['zip'] = data['zip'].astype(str)

    return data

def process_inspection_type_column(data):
    """
    Process the 'inspection_type' column in the DataFrame.

    Parameters:
        data (pandas.DataFrame): The DataFrame containing the 'inspection_type' column.

    Returns:
        pandas.DataFrame: The DataFrame with processed 'inspection_type' column.
    """
    # Fill missing 'inspection_type' values with 'Unknown'
    data['inspection_type'].fillna('Unknown', inplace=True)

    # Convert 'inspection_type' values to lowercase
    data['inspection_type'] = data['inspection_type'].str.lower()

    # Standardize certain inspection types
    data.loc[data['inspection_type'].str.contains('canvas', case=False), 'inspection_type'] = 'canvass'
    data.loc[data['inspection_type'].str.contains('licen', case=False), 'inspection_type'] = 'license'
    data.loc[data['inspection_type'].str.contains('complai', case=False), 'inspection_type'] = 'complaint'
    data.loc[data['inspection_type'].str.contains('task', case=False), 'inspection_type'] = 'task force'
    data.loc[data['inspection_type'].str.contains('kids', case=False), 'inspection_type'] = 'kids cafe'
    data.loc[data['inspection_type'].str.contains('out of', case=False), 'inspection_type'] = 'out of business'
    data.loc[data['inspection_type'].str.contains('reinspection ', case=False), 'inspection_type'] = 'recent inspection'

    # Suspected Food Poisoning replacements
    sfp_values = data['inspection_type'].str.lower().str.contains('food|sfp', regex=True)
    data.loc[sfp_values, 'inspection_type'] = 'suspected food poisoning'

    # Define a helper function to merge categories based on keywords
    def merge_categories(keyword, target_category):
        categories_containing_keyword = data['inspection_type'].str.lower().str.contains(keyword)
        data.loc[categories_containing_keyword, 'inspection_type'] = target_category

    merge_categories('recent inspection', 'Recent Inspection')
    merge_categories('out of business', 'Out of Business')
    merge_categories('no entry', 'No Entry')

    # Capitalize the first letter of each inspection type
    data['inspection_type'] = data['inspection_type'].str.title()

    # Known inspection types
    known_types = ['License', 'Canvass', 'Complaint', 'Non-Inspection', 'Suspected Food Poisoning', 'Consultation',
                   'Tag Removal', 'Recent Inspection', 'Out Of Business', 'Task Force', 'No Entry']

    # Classify the rest as 'Unknown'
    data.loc[~data['inspection_type'].isin(known_types), 'inspection_type'] = 'Unknown'

    return data

def fill_missing_violations(data):
    """
    Fill missing 'violations' values in the DataFrame.

    Parameters:
        data (pandas.DataFrame): The DataFrame containing the 'violations' column.

    Returns:
        pandas.DataFrame: The DataFrame with missing 'violations' values filled.
    """
    # Fill missing 'violations' values with 'Not Available'
    data['violations'].fillna('Not Available', inplace=True)

    return data


def fill_missing_lat_long(data):
    """
    Fill missing 'latitude' and 'longitude' values in the DataFrame with their means.
    Recreate the 'location' column by combining 'latitude' and 'longitude'.

    Parameters:
        data (pandas.DataFrame): The DataFrame containing the 'latitude' and 'longitude' columns.

    Returns:
        pandas.DataFrame: The DataFrame with missing 'latitude' and 'longitude' values filled and 'location' column recreated.
    """
    # Calculate the mean of 'latitude'
    latitude_mean = data['latitude'].mean()

    # Fill missing 'latitude' values with the mean
    data['latitude'].fillna(latitude_mean, inplace=True)

    # Calculate the mean of 'longitude'
    longitude_mean = data['longitude'].mean()

    # Fill missing 'longitude' values with the mean
    data['longitude'].fillna(longitude_mean, inplace=True)

    # Recreate the 'location' column by combining 'latitude' and 'longitude' into tuples
    data['location'] = list(zip(data['latitude'], data['longitude']))

    return data


def extract_information_from_date(data):
    """
    Extract additional date-related information from the 'inspection_date' column.

    Parameters:
        data (pandas.DataFrame): The DataFrame containing the 'inspection_date' column.

    Returns:
        pandas.DataFrame: The DataFrame with additional date-related columns extracted.
    """
    # Convert 'inspection_date' to datetime format
    data['inspection_date'] = pd.to_datetime(data['inspection_date'])

    # Extract 'inspection_year', 'inspection_month', 'inspection_season', and 'inspection_weekday'
    data['inspection_year'] = data['inspection_date'].dt.year
    data['inspection_month'] = data['inspection_date'].dt.month
    data['inspection_season'] = data['inspection_month'] % 12 // 3 + 1
    data['inspection_weekday'] = data['inspection_date'].dt.weekday

    return data




