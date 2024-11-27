import pandas as pd
import re
from difflib import SequenceMatcher

def standardize(text):
    # Convert to uppercase for consistency
    text = str(text).upper()

    # Remove non-distinct terms and standardize keywords
    text = re.sub(r'\bLTD\b|\bINTL\b|\bCO\b|\bLLC\b|\bINC\b|\bCORP\b', '', text)
    text = re.sub(r'\bLEVEL\s?\d+\b', '', text)
    text = re.sub(r'\bNEO\b', '', text)
    text = re.sub(r'\bPO\s?BOX\b', 'POBOX', text)
    text = re.sub(r'\b(STREET|ST|AVENUE|AVE|ROAD|RD|BOULEVARD|BLVD|DRIVE|DR)\b', 'RD', text)
    text = re.sub(r'\bHOI\s?BUN\s?ROAD\b', 'HOIBUNRD', text)
    text = re.sub(r'\bEAST\s?WING\b', 'EASTWING', text)
    text = re.sub(r'\b(KWUN\s?TONG|KOWLOON|HONG\s?KONG|CHINA|HK|CN)\b', '', text)
    text = re.sub(r'\bTEL/FAX\b|\bPHONE\b|\bFAX\b|\bTEL\b', '', text)
    text = re.sub(r'\b\w\b', '', text)
    text = re.sub(r'\d{5,}', '', text)

    # Split, sort, and join to make order irrelevant
    text = ' '.join(sorted(text.split()))

    # Remove any remaining punctuation after sorting
    text = re.sub(r'[\s,.-]', '', text)

    return text.strip()


def process_shipper_data(input_csv, output_csv, name_threshold=85, address_threshold=50):
    """
    Processes shipper data for fuzzy matching to identify duplicate or similar entries
    based on 'Shipper Name' and 'Shipper Address' columns. Adds 'Location Index',
    'Name Confidence', and 'Address Confidence' columns, and saves the updated data to a new CSV file.
    """
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()

    if 'Shipper Name' not in df.columns or 'Shipper Address' not in df.columns:
        print("Error: Required columns 'Shipper Name' and 'Shipper Address' are not present in the CSV file.")
        return

    df['Standardized Name'] = df['Shipper Name'].apply(standardize)
    df['Standardized Address'] = df['Shipper Address'].apply(standardize)
    df['Location Index'] = -1
    df['Name Confidence'] = None
    df['Address Confidence'] = None

    for i, row_i in df.iterrows():
        for j, row_j in df.iterrows():
            if i >= j:
                continue

            # Using SequenceMatcher to calculate similarity scores
            name_score = SequenceMatcher(None, row_i['Standardized Name'], row_j['Standardized Name']).ratio() * 100
            address_score = SequenceMatcher(None, row_i['Standardized Address'], row_j['Standardized Address']).ratio() * 100

            if name_score >= name_threshold and address_score >= address_threshold:
                if df.at[i, 'Location Index'] == -1 and df.at[j, 'Location Index'] == -1:
                    new_index = max(df['Location Index']) + 1
                    df.at[i, 'Location Index'] = new_index
                    df.at[j, 'Location Index'] = new_index
                elif df.at[i, 'Location Index'] != -1:
                    df.at[j, 'Location Index'] = df.at[i, 'Location Index']
                elif df.at[j, 'Location Index'] != -1:
                    df.at[i, 'Location Index'] = df.at[j, 'Location Index']

                df.at[i, 'Name Confidence'] = name_score
                df.at[i, 'Address Confidence'] = address_score
                df.at[j, 'Name Confidence'] = name_score
                df.at[j, 'Address Confidence'] = address_score

    df = df.drop(columns=['Standardized Name', 'Standardized Address'])

    df.to_csv(output_csv, index=False)
    print(f"Processed data saved to '{output_csv}'")


process_shipper_data('./shipper_name_duplicates_labelled.csv', 'merge2Sequence.csv')

