import pandas as pd
import re
from rapidfuzz import fuzz
import time


def standardize(text):
    """
    Standardizes a given text for consistency in address and name matching.
    """
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


def process_shipper_data(input_csv, output_csv, name_threshold=80, address_threshold=58, name_weight=1.3):
    """
    Processes shipper data to identify potential duplicate entries by comparing
    names and addresses with standardized text.
    """
    # Start timing
    start_time = time.time()

    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()

    df['Standardized Name'] = df['Shipper Name'].apply(standardize)
    df['Standardized Address'] = df['Shipper Address'].apply(standardize)
    df['Location Index'] = -1
    df['Name Confidence'] = None
    df['Address Confidence'] = None
    df['Overall Similarity'] = None

    for i, row_i in df.iterrows():
        for j, row_j in df.iterrows():
            if i >= j:
                continue

            name_score = fuzz.token_sort_ratio(row_i['Standardized Name'], row_j['Standardized Name']) * name_weight
            address_score = fuzz.token_sort_ratio(row_i['Standardized Address'], row_j['Standardized Address'])

            overall_similarity = (name_score + address_score) / (1 + name_weight)

            if name_score >= name_threshold and address_score >= address_threshold:
                if df.at[i, 'Location Index'] == -1 and df.at[j, 'Location Index'] == -1:
                    new_index = max(df['Location Index']) + 1
                    df.at[i, 'Location Index'] = new_index
                    df.at[j, 'Location Index'] = new_index
                elif df.at[i, 'Location Index'] != -1:
                    df.at[j, 'Location Index'] = df.at[i, 'Location Index']
                elif df.at[j, 'Location Index'] != -1:
                    df.at[i, 'Location Index'] = df.at[j, 'Location Index']

                df.at[i, 'Name Confidence'] = name_score / name_weight
                df.at[i, 'Address Confidence'] = address_score
                df.at[i, 'Overall Similarity'] = overall_similarity
                df.at[j, 'Name Confidence'] = name_score / name_weight
                df.at[j, 'Address Confidence'] = address_score
                df.at[j, 'Overall Similarity'] = overall_similarity

    df = df.drop(columns=['Standardized Name', 'Standardized Address'])
    df.to_csv(output_csv, index=False)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print timing and confirmation
    print(f"Processed data saved to '{output_csv}'")
    print(f"Processing Time: {elapsed_time:.2f} seconds")


# Run the process
process_shipper_data('./shipper_name_duplicates_labelled.csv', 'merge2More.csv')
