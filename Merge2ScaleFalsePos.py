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
    names and addresses with standardized text. Includes metrics for false positives
    and processing time.
    """
    start_time = time.time()

    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()

    df['Standardized Name'] = df['Shipper Name'].apply(standardize)
    df['Standardized Address'] = df['Shipper Address'].apply(standardize)
    df['Location Index'] = -1
    df['Name Confidence'] = None
    df['Address Confidence'] = None
    df['Overall Similarity'] = None
    df['False Positive'] = None  # Add a column for false positives

    # Track results for summary
    total_matches = 0
    total_name_confidence = 0
    total_address_confidence = 0
    total_overall_similarity = 0

    for i, row_i in df.iterrows():
        for j, row_j in df.iterrows():
            if i >= j:
                continue

            name_score = fuzz.token_sort_ratio(row_i['Standardized Name'], row_j['Standardized Name']) * name_weight
            address_score = fuzz.token_sort_ratio(row_i['Standardized Address'], row_j['Standardized Address'])

            overall_similarity = (name_score + address_score) / (1 + name_weight)

            # Check if the match meets the thresholds
            is_match = name_score >= name_threshold and address_score >= address_threshold
            is_false_positive = name_score >= name_threshold and address_score < (address_threshold - 10)

            if is_match:
                if df.at[i, 'Location Index'] == -1 and df.at[j, 'Location Index'] == -1:
                    new_index = max(df['Location Index']) + 1
                    df.at[i, 'Location Index'] = new_index
                    df.at[j, 'Location Index'] = new_index
                elif df.at[i, 'Location Index'] != -1:
                    df.at[j, 'Location Index'] = df.at[i, 'Location Index']
                elif df.at[j, 'Location Index'] != -1:
                    df.at[i, 'Location Index'] = df.at[j, 'Location Index']

                # Update confidence levels
                df.at[i, 'Name Confidence'] = name_score / name_weight
                df.at[i, 'Address Confidence'] = address_score
                df.at[i, 'Overall Similarity'] = overall_similarity
                df.at[j, 'Name Confidence'] = name_score / name_weight
                df.at[j, 'Address Confidence'] = address_score
                df.at[j, 'Overall Similarity'] = overall_similarity

                # Mark false positive
                df.at[i, 'False Positive'] = "Yes" if is_false_positive else "No"
                df.at[j, 'False Positive'] = "Yes" if is_false_positive else "No"

                # Update summary metrics
                total_matches += 1
                total_name_confidence += name_score / name_weight
                total_address_confidence += address_score
                total_overall_similarity += overall_similarity

    # Remove standardized columns for output
    df = df.drop(columns=['Standardized Name', 'Standardized Address'])
    df.to_csv(output_csv, index=False)

    # Calculate processing time
    end_time = time.time()
    processing_time_ms = (end_time - start_time) * 1000

    # Calculate averages
    avg_name_confidence = total_name_confidence / total_matches if total_matches > 0 else 0
    avg_address_confidence = total_address_confidence / total_matches if total_matches > 0 else 0
    avg_overall_similarity = total_overall_similarity / total_matches if total_matches > 0 else 0

    # Print metrics
    print(f"Processed data saved to '{output_csv}'")
    print(f"Processing Time: {processing_time_ms:.2f} milliseconds")
    print(f"Total Matches Found: {total_matches}")
    print(f"Average Name Confidence: {avg_name_confidence:.2f}")
    print(f"Average Address Confidence: {avg_address_confidence:.2f}")
    print(f"Average Overall Similarity: {avg_overall_similarity:.2f}")

process_shipper_data('./shipper_name_duplicates_labelled.csv', 'merge2More_with_false_positive.csv')
