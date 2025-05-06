import pandas as pd
import gzip
from collections import defaultdict

def generate_subsequences(seq):
    """Generates subsequences of length 2, including all repeated identical pairs."""
    subsequences = []
    for i in range(len(seq) - 1):
        subsequences.append(tuple(seq[i:i+2]))  # No exclusion of identical pairs
    return subsequences

def gsp_mining(file_path, min_count=2, num_days=31, rows_to_read=None):
    with gzip.open(file_path, 'rt') as f:
        data = pd.read_csv(f)
   
    if rows_to_read:
        data = data.head(rows_to_read)

    data = data[data['d'] < num_days]
   
    # Group data by 'd' (day) to create sequences for each day
    grouped = data.groupby('d')
   
    # Initialize dictionaries to hold subsequences and their counts
    subsequence_counts = defaultdict(int)
    subsequence_days = defaultdict(set)  # To track on which days subsequences appear

    for day, group in grouped:
        # Extract the sequence of (x, y) pairs for the day
        sequence = list(zip(group['x'], group['y']))
       
        # Generate subsequences of length 2
        subsequences = generate_subsequences(sequence)
        for subseq in subsequences:
            subsequence_counts[subseq] += 1
            subsequence_days[subseq].add(day)
   
    # Filter subsequences by count >= 2 and calculate support
    filtered_subsequences = []
    for subseq, count in subsequence_counts.items():
        if count >= min_count:
            support = len(subsequence_days[subseq]) / num_days  # Support is #days containing subsequence / total days
            filtered_subsequences.append((subseq, count, support))
   
    # Sort by count and support
    filtered_subsequences.sort(key=lambda x: (-x[1], -x[2]))  # Sort by count (descending), then by support (descending)
   
    return filtered_subsequences

def print_filtered_subsequences(file_name, filtered_subsequences):
    print(f"Results for {file_name}:")
    for subseq, count, support in filtered_subsequences:
        print(f"Subsequence: {subseq}, Count: {count}, Support: {support:.2f}")
    print("\n" + "="*50 + "\n")

files = [
    'C:/Users/chow1/Downloads/cityA_groundtruthdata.csv.gz',
    'C:/Users/chow1/Downloads/cityB_challengedata.csv.gz',
    'C:/Users/chow1/Downloads/cityC_challengedata.csv.gz',
    'C:/Users/chow1/Downloads/cityD_challengedata.csv.gz'
]
rows_to_read_dict = {
    'C:/Users/chow1/Downloads/cityA_groundtruthdata.csv.gz': 600,
    'C:/Users/chow1/Downloads/cityB_challengedata.csv.gz': 600,
    'C:/Users/chow1/Downloads/cityC_challengedata.csv.gz': 400,
    'C:/Users/chow1/Downloads/cityD_challengedata.csv.gz': 800,
}

# Apply Generalised Sequential Pattern mining to each dataset
for file in files:
    rows_to_read = rows_to_read_dict[file]  # Get the specific row limit for each file
    filtered_subsequences = gsp_mining(file, rows_to_read=rows_to_read)
    print_filtered_subsequences(file, filtered_subsequences)