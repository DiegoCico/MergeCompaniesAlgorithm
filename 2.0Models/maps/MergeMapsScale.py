import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import asyncio
from aiohttp import ClientSession
from fuzzywuzzy import fuzz
from tqdm import tqdm
from datetime import datetime

NAME_THRESHOLD = 80
DISTANCE_THRESHOLD = 50
RETRY = 3
DELAY = 2


def standardize(text):
    """
    Standardizes the input text by converting it to uppercase, removing extra spaces,
    and trimming whitespace.

    Args:
        text (str): The input text to standardize.

    Returns:
        str: The standardized text.
    """
    text = str(text).upper()
    return ' '.join(text.split()).strip()


async def fetch_geocode(session, geolocator, address):
    """
    Asynchronously fetches the geolocation for a given address.

    Args:
        session (ClientSession): An aiohttp client session.
        geolocator (Nominatim): Geolocator object for geocoding.
        address (str): Address to geocode.

    Returns:
        tuple: Latitude and longitude of the address, or (0, 0) if geocoding fails.
    """
    for attempt in range(RETRY):
        try:
            location = await asyncio.get_event_loop().run_in_executor(None, geolocator.geocode, address)
            if location:
                return (location.latitude, location.longitude)
        except Exception:
            await asyncio.sleep(DELAY * (2 ** attempt))  # Exponential backoff
    return (0, 0)


async def async_geocode(addresses):
    """
    Asynchronously geocodes a list of addresses.

    Args:
        addresses (list): A list of addresses to geocode.

    Returns:
        list: A list of tuples containing latitude and longitude for each address.
    """
    geolocator = Nominatim(user_agent="geo_match")
    async with ClientSession() as session:
        tasks = [fetch_geocode(session, geolocator, address) for address in addresses]
        results = []
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Geocoding"):
            results.append(await future)
        return results


def are_similar(name1, name2):
    """
    Determines if two company names are similar based on a fuzzy matching algorithm.

    Args:
        name1 (str): The first company name.
        name2 (str): The second company name.

    Returns:
        bool: True if the similarity score exceeds the threshold, otherwise False.
    """
    return fuzz.ratio(name1, name2) > NAME_THRESHOLD


def merge_companies(df):
    """
    Merges companies with similar names within a 50-mile radius into the same group.

    Args:
        df (pandas.DataFrame): DataFrame containing 'Company Name', 'Latitude', and 'Longitude' columns.

    Returns:
        pandas.DataFrame: The updated DataFrame with a new 'Location Index' column.
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


def process_data_with_async_geopy(input_csv, output_csv):
    """
    Processes an input CSV file by standardizing addresses and company names, performing geocoding,
    and merging companies with similar names within a geographic proximity.

    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to save the processed output CSV file.

    Returns:
        None
    """
    start_time = datetime.now()
    print(f"Process started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

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

    # Perform asynchronous geocoding
    print("Geocoding addresses asynchronously...")
    addresses = df['Standardized Address'].tolist()
    df['Coordinates'] = asyncio.run(async_geocode(addresses))

    # Split coordinates into latitude and longitude for clarity
    df['Latitude'] = df['Coordinates'].apply(lambda x: x[0])
    df['Longitude'] = df['Coordinates'].apply(lambda x: x[1])
    df.drop(columns=['Coordinates', 'Standardized Address'], inplace=True)

    # Standardize company names
    print("Standardizing company names...")
    df['Company Name'] = df['Company Name'].apply(standardize)

    # Merge similar companies
    print("Merging similar companies...")
    merged_df = merge_companies(df)

    # Save to CSV
    try:
        merged_df.to_csv(output_csv, index=False)
        print(f"Processed data saved to '{output_csv}'")
    except Exception as e:
        print(f"Error saving CSV: {e}")

    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    print(f"Process completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {elapsed_time:.2f} seconds")


# Example usage
if __name__ == "__main__":
    process_data_with_async_geopy(
        input_csv='../import_yeti.csv',
        output_csv='./merged_companies_async.csv'
    )
