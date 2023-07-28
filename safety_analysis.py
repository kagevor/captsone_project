import matplotlib.pyplot as plt
import folium
from branca.colormap import linear
from folium import plugins

def map_safety_scores_by_inspection_year(data):
    # Function to map safety scores by inspection year
    # Create a map centered on Chicago
    map_chicago = folium.Map(location=[41.8781, -87.6298], zoom_start=11)

    # Group by 'inspection_year' and calculate the mean safety score for each group
    grouped_data = data.groupby('inspection_year')['safety_score'].mean().reset_index()

    # Bar plot of safety scores by inspection_year type
    plt.figure(figsize=(12, 6))
    plt.bar(grouped_data['inspection_year'], grouped_data['safety_score'], color='orange')
    plt.xlabel('Inspection Year')
    plt.ylabel('Mean Safety Score')
    plt.title('Mean Safety Score by Inspection Year')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def map_safety_scores_by_zip_code(data, zip_code_boundaries):
    # Function to map safety scores by zip code
    # Merge the zip code boundaries and data DataFrames on the 'zip' column
    merged_data = zip_code_boundaries.merge(data.groupby('zip')['safety_score'].mean().reset_index(), on='zip', how='left')

    # Create a map centered on Chicago
    map_chicago = folium.Map(location=[41.8781, -87.6298], zoom_start=11)

    # Calculate the min and max safety scores for the color scale
    min_score = merged_data['safety_score'].min()
    max_score = merged_data['safety_score'].max()

    # Create a gradient color scheme based on the safety scores
    colormap = linear.YlOrRd_09.scale(min_score, max_score)

    # Create a style function to shape the zip code areas on the map with the gradient color scheme
    def style_function(feature):
        safety_score = feature['properties']['safety_score']
        return {
            'fillColor': colormap(safety_score),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7
        }

    # Create a GeoJSON layer for the zip code boundaries with the style function and tooltip
    folium.GeoJson(merged_data,
                   tooltip=folium.GeoJsonTooltip(fields=['zip', 'safety_score'],
                                                 aliases=['Zip Code', 'Mean Safety Score'],
                                                 labels=True),
                   style_function=style_function).add_to(map_chicago)

    # Add the color scale to the map
    colormap.caption = 'Mean Safety Score'
    map_chicago.add_child(colormap)

    # Add title to the map
    title_html = '''
                 <h3 align="center" style="font-size:16px"><b>Mean Safety Score by Zip Code</b></h3>
                 '''
    map_chicago.get_root().html.add_child(folium.Element(title_html))

    # Display the map
    return map_chicago

def map_safety_scores_by_zip_code_and_restaurants(data, zip_code_boundaries):
    # Function to map safety scores by zip code and restaurants that passed inspection in 2023
    # Filter data for specific criteria
    filtered_data = data[
        (data['facility_type'] == 'Restaurant') &
        (data['results'] == 'Pass') &
        (data['zip'].isin(zip_code_boundaries['zip'])) &
        (data['inspection_year'] == 2023)
    ]

    # Create a map centered on Chicago
    map_chicago = folium.Map(location=[41.8781, -87.6298], zoom_start=11)

    # Create a MarkerCluster layer to group the markers
    restaurants_map = plugins.MarkerCluster().add_to(map_chicago)

    # Define a colormap for safety scores
    colormap = linear.YlOrRd_09.scale(filtered_data['safety_score'].min(), filtered_data['safety_score'].max())

    # Add markers for each restaurant to the MarkerCluster
    for lat, lon, score, label in zip(filtered_data['latitude'], filtered_data['longitude'], filtered_data['safety_score'], filtered_data['dba_name']):
        # Get the color for the marker based on the safety score
        color = colormap(score)
        # Create a CircleMarker for each restaurant location
        folium.CircleMarker(
            location=(lat, lon),
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=label
        ).add_to(restaurants_map)

    # Add the colormap to the map
    colormap.caption = 'Mean Safety Score'
    map_chicago.add_child(colormap)

    # Add title to the map
    title_html = '''
                <h3 align="center" style="font-size:16px"><b>Safety Score by Zip Code and Restaurants that Passed Inspection in 2023</b></h3>
                '''
    map_chicago.get_root().html.add_child(folium.Element(title_html))

    # Display the map
    return map_chicago
