import pandas as pd
from geopy.geocoders import Nominatim
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
from datetime import datetime

def standardize(text):
    """Standardize text to remove variations."""
    text = str(text).upper()
    return ' '.join(text.split()).strip()

def get_geolocation(address):
    """Get latitude and longitude for a given address."""
    try:
        geolocator = Nominatim(user_agent="geo_match")
        location = geolocator.geocode(address)
        if location:
            return (location.latitude, location.longitude)
        else:
            return (0, 0)  # Default coordinates if not found
    except Exception:
        return (0, 0)

def parallel_geocoding(addresses):
    """Perform geocoding in parallel."""
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(get_geolocation, addresses), total=len(addresses), desc="Geocoding"))
    return results

def process_data_with_geopy(input_csv, output_csv):
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
    df['Coordinates'] = parallel_geocoding(df['Standardized Address'])

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
    process_data_with_geopy(
        input_csv='../import_yeti.csv',
        output_csv='./processed_data_with_geopy.csv'
    )