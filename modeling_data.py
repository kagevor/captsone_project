import pandas as pd
import ast

import data_eda


def load_and_preprocess_data():
    # Load the dataset
    data = data_eda.data

    # Convert 'inspection_type' column into dummy variables using one-hot encoding
    data = pd.get_dummies(data, columns=['inspection_type'], prefix='', prefix_sep='')
    data = pd.get_dummies(data, columns=['facility_type_grouped'], prefix='', prefix_sep='')
    risk_encoded = pd.get_dummies(data['risk'], prefix='risk')
    data = pd.concat([data, risk_encoded], axis=1)

    food_codes = [11, 12, 13, 14, 15, 17, 23, 26, 27, 28, 30, 31, 37, 39, 42]
    facility_codes = [10, 18, 19, 20, 21, 22, 33, 35, 36, 38, 41, 43, 44, 48, 50, 51, 53, 55, 56, 59, 60, 62]
    sanitary_codes = [2, 8, 16, 40, 45, 46, 47, 49, 52, 54]
    staff_codes = [1, 3, 7, 9, 25, 57, 58]
    unknown_codes = [4, 5, 6, 24, 29, 32, 61, 63]

    codes_violation = [food_codes, facility_codes, sanitary_codes, staff_codes, unknown_codes]

    # Function to safely evaluate and convert the 'violation Codes' string to a list
    def safe_eval(code_str):
        try:
            return ast.literal_eval(code_str)
        except (SyntaxError, ValueError):
            return []

    # Function to count the violations for each row based on the codes
    def count_violations(row):
        violation_codes = safe_eval(row['violation Codes'])
        count_per_group = []
        for group_codes in codes_violation:
            count_per_group.append(sum(code in group_codes for code in violation_codes))
        return pd.Series(count_per_group, index=['food', 'facility', 'sanitary', 'staff', 'unknown'])

    # Apply the count_violations function to the DataFrame
    data[['violation_food', 'violation_facility', 'violation_sanitary', 'violation_staff',
          'violation_unknown']] = data.apply(count_violations, axis=1)

    # Function to map inspection outcomes to 'Pass' or 'Fail'
    def map_to_pass_or_fail(outcome):
        if outcome in ['Fail', 'Business Not Located', 'Out of Business', 'No Entry', 'Not Ready']:
            return 0
        else:
            return 1

    # Create the new 'result' column by applying the mapping function to 'inspection_outcome'
    data['result'] = data['results'].apply(map_to_pass_or_fail)

    return data
