from data_loading import load_data
from data_preprocessing import data_preprocessing
import violation_analysis
import risk_analysis
import safety_analysis
import geopandas as gpd


# Load and clean the data
data = load_data()
data = data_preprocessing(data)

# Data Preprocessing
data['violation Codes'] = data['violations'].apply(violation_analysis.extract_violation_codes)

# Load the zip code boundaries as a GeoDataFrame
zip_code_boundaries = gpd.read_file('Boundaries - ZIP Codes.geojson')

# Data Visualization and Analysis
violation_analysis.plot_violations_stacked_bars(data, 'Distribution of most common violations')
risk_analysis.visualize_risk_counts(data)
risk_analysis.create_risk_distribution_map(data)
safety_analysis.map_safety_scores_by_inspection_year(data)
safety_analysis.map_safety_scores_by_zip_code(data, zip_code_boundaries)
safety_analysis.map_safety_scores_by_zip_code_and_restaurants(data, zip_code_boundaries)

# Save the processed data to a CSV file
# data.to_csv('data_eda.csv', index=False)

# def save_data_to_csv(data, file_path):
#     data.to_csv(file_path, index=False)