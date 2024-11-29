import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeopyError
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

# Configuration constants
RETRY = 3  # Number of retries for geocoding
DELAY = 2  # Delay between retries (seconds)

# Initialize Nominatim geocoder
geolocator = Nominatim(user_agent="geo_app")

def preprocess_address(address):
    """
    Preprocesses an address to improve geocoding results.
    """
    address = address.upper()
    # Replace ambiguous abbreviations
    address = address.replace("ROOM", "").replace("RM", "").replace("BLDG", "BUILDING")
    address = address.replace("AND E", "AND EAST").replace("RD", "ROAD").replace("FN", "")
    # Add separators for clarity
    if "CHANGSHA" in address and "CHINA" not in address:
        address += ", CHINA"
    return ' '.join(address.split()).strip()

def get_geolocation(address, retries=RETRY, delay=DELAY):
    """
    Geocodes an address using Nominatim. Retries on failure.
    """
    preprocessed = preprocess_address(address)
    for attempt in range(retries):
        try:
            location = geolocator.geocode(preprocessed)
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

def process_data(input_csv, output_csv):
    """
    Processes data by preprocessing, geocoding, and saving the results.
    """
    # Load data
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: The file '{input_csv}' was not found.")
        return

    # Limit to a manageable size for testing
    df = df.head(10)

    df.columns = df.columns.str.strip()

    # Standardize and preprocess addresses
    print("Preprocessing addresses...")
    df['Preprocessed Address'] = df['first3_addresses'].apply(preprocess_address)  # Replace 'address_column_name'

    # Perform geocoding in parallel
    print("Geocoding addresses in parallel...")
    df['Coordinates'] = parallel_geocoding(df['Preprocessed Address'])

    # Split coordinates into latitude and longitude
    df['Latitude'] = df['Coordinates'].apply(lambda x: x[0])
    df['Longitude'] = df['Coordinates'].apply(lambda x: x[1])
    df.drop(columns=['Coordinates', 'Preprocessed Address'], inplace=True)

    # Save results to CSV
    try:
        df.to_csv(output_csv, index=False)
        print(f"Processed data saved to '{output_csv}'")
    except Exception as e:
        print(f"Error saving CSV: {e}")

if __name__ == "__main__":
    # Replace 'input.csv' with the name of your input file
    process_data(
        input_csv='../import_yeti.csv',
        output_csv='./geocoded_output.csv'
    )
