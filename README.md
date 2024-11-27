# README for Data Processing Scripts

## Overview
This script processes a dataset of company information and addresses to standardize text fields, geocode addresses, and identify similar companies based on names and locations. Two variations of the algorithm are included, leveraging different approaches to geocoding and similarity matching.

---

## Features
1. **Standardization**:
    - Converts company names and addresses to uppercase, removes extra spaces, and trims unnecessary characters.

2. **Geocoding**:
    - **Variation 1**: Uses `Nominatim` geocoder with multiprocessing for parallel geocoding.
    - **Variation 2**: Implements asynchronous geocoding using `aiohttp` for faster processing.

3. **Similarity Matching**:
    - Compares company names using fuzzy matching algorithms to determine similarity.
    - Groups companies within a specified distance threshold.

4. **Distance Filtering**:
    - Filters companies based on geographic proximity using geodesic distance calculations (default: 50 miles).

5. **Low Similarity Tracking**:
    - Identifies and separates records with low similarity scores for further review.

---

## Input Data
The input CSV file should contain the following columns:
- `Company Name`: Name of the company.
- `first3_addresses`: Address details.

### Example Input:
| Company Name       | first3_addresses           |
|--------------------|----------------------------|
| Example Co.        | 123 Main St, Boston, MA    |
| Sample LLC         | 124 Main Rd, Boston, MA    |
| Another LLC        | 789 Broadway, New York, NY |

---

## Output Files
1. **Processed Data**:
    - A CSV file containing the processed data with additional columns:
        - `Latitude`: Geographical latitude of the address.
        - `Longitude`: Geographical longitude of the address.
        - `Location Index`: Group index for similar companies.

2. **Low Similarity Data**:
    - A CSV file listing records with overall similarity scores below a specified threshold (default: 68).

---

## Algorithm Variations

### Variation 1: Parallel Geocoding
- **Approach**:
    - Uses `multiprocessing.Pool` to perform parallel geocoding of addresses.
    - Suitable for environments where CPU-intensive operations can be distributed across multiple cores.

- **Output Example**:
  | Company Name       | first3_addresses           | Latitude   | Longitude  | Location Index |
  |--------------------|----------------------------|------------|------------|----------------|
  | Example Co.        | 123 Main St, Boston, MA    | 42.3601    | -71.0589   | 1              |
  | Sample LLC         | 124 Main Rd, Boston, MA    | 42.3611    | -71.0599   | 1              |
  | Another LLC        | 789 Broadway, New York, NY | 40.7128    | -74.0060   | 2              |

### Variation 2: Asynchronous Geocoding
- **Approach**:
    - Implements asynchronous geocoding using `aiohttp` and `asyncio` for concurrent address resolution.
    - Efficient for large datasets where latency is critical.

- **Output Example**:
  | Company Name       | first3_addresses           | Latitude   | Longitude  | Location Index |
  |--------------------|----------------------------|------------|------------|----------------|
  | Example Co.        | 123 Main St, Boston, MA    | 42.3601    | -71.0589   | 1              |
  | Sample LLC         | 124 Main Rd, Boston, MA    | 42.3611    | -71.0599   | 1              |
  | Another LLC        | 789 Broadway, New York, NY | 40.7128    | -74.0060   | 2              |

---

## Debugging
- Add print statements in `merge_companies` to observe how companies are grouped.
- For asynchronous geocoding, monitor address processing in real-time using `tqdm`.

---

## Notes
- Ensure geocoding requests do not exceed the rate limits of `Nominatim` by managing retries and delays.
- Adjust thresholds for name similarity, address similarity, and distance based on the dataset characteristics.