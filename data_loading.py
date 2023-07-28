import pandas as pd
import requests
import io


def load_data():
    # Define the URL from which the data will be fetched
    url = "https://data.cityofchicago.org/resource/4ijn-s7e5.csv"

    # Define parameters to limit the number of records fetched (1,000,000 in this case)
    params = {"$limit": 1000000}

    # Send a GET request to the URL with the specified parameters to fetch the data
    response = requests.get(url, params=params)

    # Read the fetched CSV data into a pandas DataFrame
    data = pd.read_csv(io.StringIO(response.text))

    # Return the loaded DataFrame
    return data


