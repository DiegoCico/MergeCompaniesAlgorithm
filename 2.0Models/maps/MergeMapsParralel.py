import pandas as pd
from geopy import Nominatim
from geopy.exc import GeopyError
from geopy.geocoders import ArcGIS
from geopy.distance import geodesic
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import ssl
import time
from datetime import datetime
from fuzzywuzzy import fuzz

# Configuration constants
NAME_THRESHOLD = 80  # Fuzzy matching threshold for company names
DISTANCE_THRESHOLD = 1000  # Maximum allowable distance (in miles) for grouping companies
RETRY = 3  # Number of retries for geocoding
DELAY = 2  # Delay between retries (seconds)

# Initialize ArcGIS geocoder
geolocator = Nominatim(user_agent="geo_app")

def standardize(text):
    """
    Standardizes input text: converts to uppercase, removes extra spaces, and trims whitespace.
    """
    text = str(text).upper()
    return ' '.join(text.split()).strip()

def get_geolocation(address, retries=RETRY, delay=DELAY):
    """
    Geocodes an address using ArcGIS. Retries on failure with exponential backoff.
    """
    for attempt in range(retries):
        try:
            location = geolocator.geocode(address)
            if location:
                return (location.latitude, location.longitude)
            else:
                return (0, 0)  # Default if geocoding fails
        except GeopyError:
            time.sleep(delay * (2 ** attempt))  # Exponential backoff
    return (0, 0)

def parallel_geocoding(addresses):
    """
    Executes geocoding in parallel for a list of addresses using multiprocessing.
    """
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(get_geolocation, addresses), total=len(addresses), desc="Geocoding"))
    return results

def are_similar(name1, name2):
    """
    Checks if two company names are similar using fuzzy matching.
    """
    return fuzz.ratio(name1, name2) > NAME_THRESHOLD

def merge_companies(df):
    """
    Groups companies with similar names and proximity within DISTANCE_THRESHOLD miles.
    """
    location_index = 1
    df['Location Index'] = None
    visited = set()

    for i, row1 in df.iterrows():
        if i in visited:
            continue

        group = [i]
        visited.add(i)

        for j, row2 in df.iterrows():
            if j in visited or i == j:
                continue

            if are_similar(row1['Company Name'], row2['Company Name']):
                distance = geodesic((row1['Latitude'], row1['Longitude']),
                                    (row2['Latitude'], row2['Longitude'])).miles
                if distance <= DISTANCE_THRESHOLD:
                    group.append(j)
                    visited.add(j)

        if len(group) > 1:
            for index in group:
                df.at[index, 'Location Index'] = location_index
            location_index += 1

    return df

def process_data_with_arcgis(input_csv, output_csv):
    """
    Processes data by standardizing, geocoding, and merging similar companies based on names and proximity.
    """
    start_time = time.time()
    print(f"Process started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    try:
        df = pd.read_csv(input_csv)
        df = df.head(10)
    except FileNotFoundError:
        print(f"Error: The file '{input_csv}' was not found.")
        return

    df.columns = df.columns.str.strip()

    # Standardize addresses
    print("Standardizing addresses...")
    df['Standardized Address'] = df['first3_addresses'].apply(standardize)

    # Perform geocoding in parallel
    print("Geocoding addresses in parallel...")
    df['Coordinates'] = parallel_geocoding(df['Standardized Address'])

    # Split coordinates into latitude and longitude
    df['Latitude'] = df['Coordinates'].apply(lambda x: x[0])
    df['Longitude'] = df['Coordinates'].apply(lambda x: x[1])
    df.drop(columns=['Coordinates', 'Standardized Address'], inplace=True)

    # Standardize company names
    print("Standardizing company names...")
    df['Company Name'] = df['shipper_name'].apply(standardize)

    # Merge similar companies
    print("Merging similar companies...")
    merged_df = merge_companies(df)

    # Save to CSV
    try:
        merged_df.to_csv(output_csv, index=False)
        print(f"Processed data saved to '{output_csv}'")
    except Exception as e:
        print(f"Error saving CSV: {e}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Process completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {elapsed_time:.2f} seconds")

# Example usage
if __name__ == "__main__":
    process_data_with_arcgis(
        input_csv='../import_yeti.csv',
        output_csv='./merged_companies.csv'
    )
