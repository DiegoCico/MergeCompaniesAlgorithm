import pandas as pd
from rapidfuzz import fuzz
from multiprocessing import Pool, cpu_count
import re

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

def calculate_similarity_chunk(chunk_data, all_data, name_weight, name_threshold, address_threshold):
    results = []
    for i, row_i in chunk_data.iterrows():
        standardized_name_i = row_i['Standardized Name']
        standardized_address_i = row_i['Standardized Address']

        for j, row_j in all_data.iterrows():
            if i >= j:
                continue

            name_score = fuzz.token_sort_ratio(standardized_name_i, row_j['Standardized Name']) * name_weight
            address_score = fuzz.token_sort_ratio(standardized_address_i, row_j['Standardized Address'])
            overall_similarity = (name_score + address_score) / (1 + name_weight)

            if name_score >= name_threshold and address_score >= address_threshold:
                results.append((i, j, name_score / name_weight, address_score, overall_similarity))
    return results

def process_shipper_data_optimized(input_csv, output_csv, low_score_csv, name_threshold=80, address_threshold=60, name_weight=1.4):
    start_time = pd.Timestamp.now()

    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()

    # Standardize text in parallel
    df['Standardized Name'] = df['shipper_name'].apply(standardize)
    df['Standardized Address'] = df['first3_addresses'].apply(standardize)

    # Split data into chunks for multiprocessing
    num_cores = cpu_count()
    chunk_size = len(df) // num_cores
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    # Use multiprocessing to calculate similarities
    with Pool(num_cores) as pool:
        results = pool.starmap(
            calculate_similarity_chunk,
            [(chunk, df, name_weight, name_threshold, address_threshold) for chunk in chunks]
        )

    # Flatten results
    matches = [match for chunk_result in results for match in chunk_result]

    # Create a new DataFrame for low overall similarity scores
    low_similarity_data = []

    df['Location Index'] = -1
    df['Name Confidence'] = None
    df['Address Confidence'] = None
    df['Overall Similarity'] = None

    for i, j, name_conf, addr_conf, overall_sim in matches:
        if overall_sim < 68:  # Filter low similarity scores
            low_similarity_data.append({
                'Company Name 1': df.at[i, 'shipper_name'],
                'Address 1': df.at[i, 'first3_addresses'],
                'Company Name 2': df.at[j, 'shipper_name'],
                'Address 2': df.at[j, 'first3_addresses'],
                'Name Confidence': name_conf,
                'Address Confidence': addr_conf,
                'Overall Similarity': overall_sim
            })

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

    # Save DataFrame with low similarity rows to CSV
    low_similarity_df = pd.DataFrame(low_similarity_data)
    low_similarity_df.to_csv(low_score_csv, index=False)

    # Drop standardized columns and save the main dataset
    df = df.drop(columns=['Standardized Name', 'Standardized Address'])
    df.to_csv(output_csv, index=False)

    end_time = pd.Timestamp.now()
    processing_time = (end_time - start_time).total_seconds()

    print(f"Processed data saved to '{output_csv}'")
    print(f"Low similarity data saved to '{low_score_csv}'")
    print(f"Processing Time: {processing_time:.2f} seconds")
    print(f"Total Matches Found: {len(matches)}")
    print(f"Total Low Similarity Records: {len(low_similarity_data)}")

if __name__ == '__main__':
    process_shipper_data_optimized(
        '../import_yeti.csv',
        './processed_data.csv',
        './low_similarity_data.csv')
