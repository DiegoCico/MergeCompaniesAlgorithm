import pandas as pd
from geopy.geocoders import ArcGIS
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import ssl
from urllib.request import urlopen
import time
from datetime import datetime

# Custom SSL context to bypass certifi
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Initialize ArcGIS geocoder
geolocator = ArcGIS(timeout=10, ssl_context=ssl_context)

def standardize(text):
    """Standardize and truncate text to remove variations."""
    text = str(text).upper()
    standardized_text = ' '.join(text.split()).strip()
    return standardized_text[:200]  # Truncate to 200 characters

def get_geolocation_arcgis(address):
    """Get latitude and longitude for a given address using ArcGIS."""
    try:
        location = geolocator.geocode(address)
        if location:
            return (location.latitude, location.longitude)
        else:
            return (0, 0)  # Default coordinates if not found
    except Exception as e:
        print(f"Error geocoding address '{address}': {e}")
        return (0, 0)

def parallel_geocoding_arcgis(addresses):
    """Perform geocoding in parallel."""
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(get_geolocation_arcgis, addresses), total=len(addresses), desc="Geocoding"))
    return results

def process_data_with_arcgis(input_csv, output_csv):
    start_time = time.time()
    print(f"Process started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: The file '{input_csv}' was not found.")
        return

    df.columns = df.columns.str.strip()

    # Standardize addresses
    print("Standardizing addresses...")
    df['Standardized Address'] = df['first3_addresses'].apply(standardize)

    # Perform geocoding in parallel
    print("Geocoding addresses in parallel...")
    df['Coordinates'] = parallel_geocoding_arcgis(df['Standardized Address'])

    # Split coordinates into latitude and longitude for clarity
    df['Latitude'] = df['Coordinates'].apply(lambda x: x[0])
    df['Longitude'] = df['Coordinates'].apply(lambda x: x[1])
    df.drop(columns=['Coordinates', 'Standardized Address'], inplace=True)

    # Save to CSV
    try:
        df.to_csv(output_csv, index=False)
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
        output_csv='./processed_data_with_arcgis.csv'
    )
