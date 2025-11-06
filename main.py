# Main script for the geopolitical event predictor
from datasets import load_dataset

if __name__ == "__main__":
    # Load the dataset from local files
    data_dir = "/Users/ashishmishra/geopolitical_predictor/data/newsroom_processed"
    print(f"Loading the Newsroom dataset from local directory: {data_dir}...")
    dataset = load_dataset("newsroom", data_dir=data_dir, trust_remote_code=True)

    # Take a small sample from the 'train' split
    print("Taking a sample of 5 articles from the 'train' split...")
    sample = dataset['train'].take(5)

    # Print the sample
    for row in sample:
        print(row)
        print("-" * 50)

    print("Sample inspection complete.")
