import pandas as pd
from rapidfuzz import fuzz
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from multiprocessing import Pool, cpu_count
import re

# Initialize geocoder
geolocator = Nominatim(user_agent="geo_filter")

def standardize(text):
    text = str(text).upper()
    text = re.sub(r'\b(LTD|INTL|CO|LLC|INC|CORP)\b', '', text)
    text = re.sub(r'\bLEVEL\s?\d+\b|\bNEO\d*\b', '', text)
    text = re.sub(r'\b(STREET|ST|AVENUE|AVE|ROAD|RD|BOULEVARD|BLVD|DRIVE|DR)\b', 'RD', text)
    text = re.sub(r'\bHOI\s?BUN\s?ROAD\b', 'BUN RD', text)
    text = re.sub(r'\b(KWUN\s?TONG|KOWLOON|HONG\s?KONG|CHINA|HK|CN)\b', '', text)
    text = re.sub(r'\bPO\s?BOX\b', 'POBOX', text)
    text = re.sub(r'\b(TEL/FAX|PHONE|FAX|TEL)\b', '', text)
    text = re.sub(r'\b\w\b', '', text)
    text = re.sub(r'\d{5,}', '', text)
    text = ' '.join(sorted(text.split()))
    text = re.sub(r'[\s,.-]', '', text)
    return text.strip()

def get_coordinates(address):
    try:
        location = geolocator.geocode(address)
        return (location.latitude, location.longitude) if location else None
    except Exception as e:
        return None

def calculate_similarity_chunk(chunk_data, all_data, name_weight, name_threshold, address_threshold, max_distance):
    results = []
    for i, row_i in chunk_data.iterrows():
        standardized_name_i = row_i['Standardized Name']
        coord_i = row_i['Coordinates']

        for j, row_j in all_data.iterrows():
            if i >= j:
                continue

            standardized_name_j = row_j['Standardized Name']
            coord_j = row_j['Coordinates']

            # Print the company names and addresses being compared
            print(f"Comparing:\n  Company 1: {row_i['shipper_name']} | Address: {row_i['first3_addresses']}\n"
                  f"  Company 2: {row_j['shipper_name']} | Address: {row_j['first3_addresses']}")

            # Skip if coordinates are missing or not within distance
            if not coord_i or not coord_j:
                print("  Skipped due to missing coordinates.")
                continue

            distance = geodesic(coord_i, coord_j).miles
            print(f"  Distance: {distance:.2f} miles")
            if distance > max_distance:
                print("  Skipped due to distance exceeding threshold.")
                continue

            name_score = fuzz.token_sort_ratio(standardized_name_i, standardized_name_j) * name_weight
            address_score = fuzz.token_sort_ratio(row_i['Standardized Address'], row_j['Standardized Address'])
            overall_similarity = (name_score + address_score) / (1 + name_weight)

            print(f"  Name Similarity: {name_score / name_weight:.2f}, Address Similarity: {address_score:.2f}, "
                  f"Overall Similarity: {overall_similarity:.2f}")

            if name_score >= name_threshold and address_score >= address_threshold:
                print("  Match found!")
                results.append((i, j, name_score / name_weight, address_score, overall_similarity))
            else:
                print("  No match.")
    return results

def process_shipper_data_optimized(input_csv, output_csv, name_threshold=80, address_threshold=60, name_weight=1.4, max_distance=50):
    start_time = pd.Timestamp.now()

    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()

    # Standardize text
    df['Standardized Name'] = df['shipper_name'].apply(standardize)
    df['Standardized Address'] = df['first3_addresses'].apply(standardize)

    # Get coordinates
    df['Coordinates'] = df['first3_addresses'].apply(get_coordinates)

    # Process without multiprocessing for debugging
    results = calculate_similarity_chunk(df, df, name_weight, name_threshold, address_threshold, max_distance)

    # Flatten results
    matches = results

    # Assign matches to DataFrame
    df['Location Index'] = -1
    df['Name Confidence'] = None
    df['Address Confidence'] = None
    df['Overall Similarity'] = None

    for i, j, name_conf, addr_conf, overall_sim in matches:
        if df.at[i, 'Location Index'] == -1 and df.at[j, 'Location Index'] == -1:
            new_index = max(df['Location Index'].max() + 1, 0)
            df.at[i, 'Location Index'] = new_index
            df.at[j, 'Location Index'] = new_index
        elif df.at[i, 'Location Index'] != -1:
            df.at[j, 'Location Index'] = df.at[i, 'Location Index']
        elif df.at[j, 'Location Index'] != -1:
            df.at[i, 'Location Index'] = df.at[j, 'Location Index']

        df.at[i, 'Name Confidence'] = name_conf
        df.at[i, 'Address Confidence'] = addr_conf
        df.at[i, 'Overall Similarity'] = overall_sim
        df.at[j, 'Name Confidence'] = name_conf
        df.at[j, 'Address Confidence'] = addr_conf
        df.at[j, 'Overall Similarity'] = overall_sim

    df = df.drop(columns=['Standardized Name', 'Standardized Address', 'Coordinates'])
    df.to_csv(output_csv, index=False)

    end_time = pd.Timestamp.now()
    processing_time = (end_time - start_time).total_seconds()

    print(f"Processed data saved to '{output_csv}'")
    print(f"Processing Time: {processing_time:.2f} seconds")
    print(f"Total Matches Found: {len(matches)}")

if __name__ == '__main__':
    process_shipper_data_optimized(
        './import_yeti.csv',
        './imported_data-MAPS.csv'
    )
