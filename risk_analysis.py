import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
from branca.colormap import linear

def visualize_risk_counts(data):
    # Function to visualize the count of inspections by risk
    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x='risk', order=data['risk'].value_counts().index, palette='viridis')
    plt.xlabel('Risk Level')
    plt.ylabel('Count')
    plt.title('Inspections by Risk Level')
    plt.show()

def create_risk_distribution_map(data):
    # Function to create a map with markers for the distribution of inspections by risk
    # Center the map on Chicago
    map_chicago = folium.Map(location=[41.8781, -87.6298], zoom_start=11)

    # Create a MarkerCluster layer to group the markers
    risk_distribution_map = plugins.MarkerCluster().add_to(map_chicago)

    # Define a colormap for risk levels
    colormap = linear.YlOrRd_09.scale(data['risk'].nunique())

    # Add markers for each inspection to the MarkerCluster
    for lat, lon, risk, label in zip(data['latitude'], data['longitude'], data['risk'], data['dba_name']):
        # Get the color for the marker based on the risk level
        color = colormap(data['risk'].unique().tolist().index(risk))
        # Create a CircleMarker for each inspection location
        folium.CircleMarker(
            location=(lat, lon),
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=label
        ).add_to(risk_distribution_map)

    # Add the colormap to the map
    colormap.caption = 'Risk Level'
    map_chicago.add_child(colormap)

    # Add title to the map
    title_html = '''
                <h3 align="center" style="font-size:16px"><b>Distribution of Inspections by Risk Level</b></h3>
                '''
    map_chicago.get_root().html.add_child(folium.Element(title_html))

    # Display the map
    return map_chicago
