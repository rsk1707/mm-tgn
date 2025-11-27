import pandas as pd
import numpy as np
from pathlib import Path

# CONFIG
DATA_DIR = Path('..../datasets/movielens-32m/movielens-original/ml-32m') # Point to your unzipped folder
OUTPUT_DIR = Path('../datasets/movielens-32m/movielens-modern/ml-modern')
MIN_YEAR = 2018
MIN_INTERACTIONS = 10 # 10-core filtering
TARGET_SIZE = 1_000_000 # Max interactions

def create_modern_subset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"1. Loading ratings from {MIN_YEAR} onwards...")
    # timestamps are in seconds. 
    # Jan 1 2018 = 1514764800
    min_ts = 1514764800 
    
    # Read in chunks to avoid OOM
    chunks = []
    for chunk in pd.read_csv(DATA_DIR / 'ratings.csv', chunksize=1_000_000):
        filtered = chunk[chunk['timestamp'] >= min_ts]
        chunks.append(filtered)
    
    df = pd.concat(chunks)
    print(f"   Initial modern interactions: {len(df)}")
    
    print("2. Performing Iterative K-Core Filtering...")
    # We need a dense graph where every user and item has at least k interactions
    while True:
        user_counts = df['userId'].value_counts()
        item_counts = df['movieId'].value_counts()
        
        valid_users = user_counts[user_counts >= MIN_INTERACTIONS].index
        valid_items = item_counts[item_counts >= MIN_INTERACTIONS].index
        
        # If we have reached stability, break
        if len(valid_users) == df['userId'].nunique() and len(valid_items) == df['movieId'].nunique():
            break
            
        # Filter
        df = df[df['userId'].isin(valid_users) & df['movieId'].isin(valid_items)]
        print(f"   Reduced to: {len(df)} interactions...")
        
        if len(df) == 0:
            raise ValueError("Criteria too strict! Lower MIN_INTERACTIONS.")

    print("3. Final Size Check")
    # If still too big, take the most recent TARGET_SIZE
    if len(df) > TARGET_SIZE:
        print(f"   Capping at {TARGET_SIZE} most recent interactions...")
        df = df.sort_values('timestamp', ascending=False).head(TARGET_SIZE)
        
    # 4. Save the Subset
    print(f"   Saving final dataset: {len(df)} rows.")
    print(f"   Users: {df['userId'].nunique()}, Movies: {df['movieId'].nunique()}")
    df.to_csv(OUTPUT_DIR / 'ratings.csv', index=False)
    
    # 5. Filter Metadata Files to match this subset
    # We only want movies/links that exist in our new ratings file
    valid_movie_ids = df['movieId'].unique()
    
    print("   Filtering movies.csv and links.csv...")
    movies = pd.read_csv(DATA_DIR / 'movies.csv')
    movies = movies[movies['movieId'].isin(valid_movie_ids)]
    movies.to_csv(OUTPUT_DIR / 'movies.csv', index=False)
    
    links = pd.read_csv(DATA_DIR / 'links.csv')
    links = links[links['movieId'].isin(valid_movie_ids)]
    links.to_csv(OUTPUT_DIR / 'links.csv', index=False)
    
    print("Done! Use 'data/ml-modern' as your dataset source.")

if __name__ == "__main__":
    create_modern_subset()