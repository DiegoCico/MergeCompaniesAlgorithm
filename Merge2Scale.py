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


def process_shipper_data_optimized(input_csv, output_csv, name_threshold=80, address_threshold=58, name_weight=1.3):
    start_time = pd.Timestamp.now()

    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()

    # Standardize text in parallel
    df['Standardized Name'] = df['Shipper Name'].apply(standardize)
    df['Standardized Address'] = df['Shipper Address'].apply(standardize)

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

    # Assign matches to DataFrame
    df['Location Index'] = -1
    df['Name Confidence'] = None
    df['Address Confidence'] = None
    df['Overall Similarity'] = None

    total_name_confidence = 0
    total_address_confidence = 0
    total_overall_similarity = 0

    lowest_name_confidence = float('inf')
    highest_name_confidence = float('-inf')
    lowest_address_confidence = float('inf')
    highest_address_confidence = float('-inf')
    lowest_overall_similarity = float('inf')
    highest_overall_similarity = float('-inf')

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

        total_name_confidence += name_conf
        total_address_confidence += addr_conf
        total_overall_similarity += overall_sim

        lowest_name_confidence = min(lowest_name_confidence, name_conf)
        highest_name_confidence = max(highest_name_confidence, name_conf)
        lowest_address_confidence = min(lowest_address_confidence, addr_conf)
        highest_address_confidence = max(highest_address_confidence, addr_conf)
        lowest_overall_similarity = min(lowest_overall_similarity, overall_sim)
        highest_overall_similarity = max(highest_overall_similarity, overall_sim)

    avg_name_confidence = total_name_confidence / len(matches) if matches else 0
    avg_address_confidence = total_address_confidence / len(matches) if matches else 0
    avg_overall_similarity = total_overall_similarity / len(matches) if matches else 0

    df = df.drop(columns=['Standardized Name', 'Standardized Address'])
    df.to_csv(output_csv, index=False)

    end_time = pd.Timestamp.now()
    processing_time = (end_time - start_time).total_seconds()

    # Print Metrics
    print(f"Processed data saved to '{output_csv}'")
    print(f"Processing Time: {processing_time:.2f} seconds")
    print(f"Total Matches Found: {len(matches)}")
    print(f"Average Name Confidence: {avg_name_confidence:.2f}")
    print(f"Average Address Confidence: {avg_address_confidence:.2f}")
    print(f"Average Overall Similarity: {avg_overall_similarity:.2f}")
    print(f"Lowest Name Confidence: {lowest_name_confidence:.2f}")
    print(f"Highest Name Confidence: {highest_name_confidence:.2f}")
    print(f"Lowest Address Confidence: {lowest_address_confidence:.2f}")
    print(f"Highest Address Confidence: {highest_address_confidence:.2f}")
    print(f"Lowest Overall Similarity: {lowest_overall_similarity:.2f}")
    print(f"Highest Overall Similarity: {highest_overall_similarity:.2f}")


if __name__ == '__main__':
    process_shipper_data_optimized('./shipper_name_duplicates_labelled.csv', 'merge2More_optimized_metrics.csv')
