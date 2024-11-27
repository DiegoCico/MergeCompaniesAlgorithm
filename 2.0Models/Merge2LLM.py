import pandas as pd
from rapidfuzz import fuzz
from langchain import OpenAI
from langchain.prompts import PromptTemplate
import re
from multiprocessing import Pool, cpu_count

# Initialize OpenAI LLM via LangChain
llm = OpenAI(model="text-davinci-003", temperature=0)

# LLM Prompt Template for Text Refinement
refinement_prompt = PromptTemplate(
    input_variables=["text"],
    template="Standardize the following text by removing noise, company suffixes, and unnecessary components. "
             "Keep only relevant business information:\n\n{text}"
)

def refine_with_llm(text):
    """
    Uses the LLM to refine the input text based on the defined prompt.
    """
    if not text or not text.strip():
        return ""
    try:
        refined_text = llm(refinement_prompt.format(text=text))
        return refined_text.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return text  # Fallback if the LLM call fails

def standardize(text, use_llm=False):
    """
    Standardizes text using regex and optionally refines it with LLM.
    """
    if not text:
        return ""
    text = str(text).upper()
    patterns = [
        r'\b(LTD|INTL|CO|LLC|INC|CORP)\b',
        r'\bLEVEL\s?\d+\b|\bNEO\d*\b',
        r'\b(STREET|ST|AVENUE|AVE|ROAD|RD|BOULEVARD|BLVD|DRIVE|DR)\b',
        r'\bHOI\s?BUN\s?ROAD\b',
        r'\b(KWUN\s?TONG|KOWLOON|HONG\s?KONG|CHINA|HK|CN)\b',
        r'\bPO\s?BOX\b',
        r'\b(TEL/FAX|PHONE|FAX|TEL)\b',
        r'\d{5,}',
        r'\b\w\b',
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text)
    text = ' '.join(sorted(text.split()))
    text = re.sub(r'[\s,.-]', '', text).strip()

    # Use LLM refinement if enabled
    if use_llm:
        text = refine_with_llm(text)

    return text

def process_shipper_data_with_llm(input_csv, output_csv, use_llm=False, name_threshold=80, address_threshold=60, name_weight=1.4):
    """
    Processes shipper data, standardizing text with optional LLM refinement.
    """
    df = pd.read_csv(input_csv).fillna("")
    df.columns = df.columns.str.strip()

    # Standardize columns
    df['Standardized Name'] = df['shipper_name'].apply(lambda x: standardize(x, use_llm))
    df['Standardized Address'] = df['first3_addresses'].apply(lambda x: standardize(x, use_llm))

    # Calculate similarities
    results = []
    for i, row_i in df.iterrows():
        for j, row_j in df.iterrows():
            if i >= j:
                continue
            name_score = fuzz.token_sort_ratio(row_i['Standardized Name'], row_j['Standardized Name']) * name_weight
            address_score = fuzz.token_sort_ratio(row_i['Standardized Address'], row_j['Standardized Address'])
            overall_similarity = (name_score + address_score) / (1 + name_weight)
            if name_score >= name_threshold and address_score >= address_threshold:
                results.append((i, j, name_score / name_weight, address_score, overall_similarity))

    # Assign similarity scores back to DataFrame
    match_df = pd.DataFrame(results, columns=["Row 1", "Row 2", "Name Confidence", "Address Confidence", "Overall Similarity"])
    match_df.to_csv(output_csv, index=False)
    print(f"Processed data saved to '{output_csv}' with {len(results)} matches.")

if __name__ == '__main__':
    process_shipper_data_with_llm(
        input_csv='./import_yeti.csv',
        output_csv='./imported_data_LLM.csv',
        use_llm=True  # Enable LLM refinement
    )
