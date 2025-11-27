import pandas as pd
from pathlib import Path
import os

# CONFIG
DATA_DIR = Path("../data/movielens-32m/movielens-modern/ml-modern")
POSTERS_DIR = Path("../data/movielens-32m/movielens-modern/ml-modern-posters")

def verify_dataset():
    print(f"🔍 Verifying dataset quality in: {DATA_DIR}")
    
    # 1. Load Interactions (The Source of Truth)
    ratings_path = DATA_DIR / "ratings.csv"
    if not ratings_path.exists():
        print("❌ CRITICAL: ratings.csv not found!")
        return
    
    df_ratings = pd.read_csv(ratings_path)
    # Ensure IDs are strings for consistent comparison
    active_movies = set(df_ratings['movieId'].astype(str))
    active_users = set(df_ratings['userId'].astype(str))
    
    print(f"\n📊 INTERACTION STATS:")
    print(f"   - Total Interactions: {len(df_ratings):,}")
    print(f"   - Unique Users: {len(active_users):,}")
    print(f"   - Unique Movies (in interactions): {len(active_movies):,}")
    
    # 2. Check Enriched Metadata (Text)
    enriched_path = DATA_DIR / "enriched.csv"
    if not enriched_path.exists():
        print("❌ CRITICAL: enriched.csv not found!")
        return
    
    df_enriched = pd.read_csv(enriched_path)
    enriched_movies = set(df_enriched['movieId'].astype(str))
    
    # Check for empty plots
    # We consider a plot "empty" if it is NaN or length < 5 chars
    valid_plots = df_enriched[df_enriched['tmdb_overview'].str.len() > 5]['movieId'].astype(str)
    valid_plots_set = set(valid_plots)
    
    print(f"\n📝 METADATA STATS:")
    print(f"   - Enriched Movies Found: {len(enriched_movies):,}")
    print(f"   - Movies with Valid Plot Summaries: {len(valid_plots_set):,}")
    
    # 3. Check Images (Posters)
    if not POSTERS_DIR.exists():
        print("❌ CRITICAL: posters/ directory not found!")
        return
    
    # Get all .jpg files and strip extension to get ID
    image_files = set(f.stem for f in POSTERS_DIR.glob("*.jpg"))
    
    print(f"\n🖼️  IMAGE STATS:")
    print(f"   - Poster Images Found: {len(image_files):,}")
    
    # 4. The Alignment Check (The most important part)
    print(f"\n🔗 ALIGNMENT CHECK (Active Movies vs. Features):")
    
    missing_text = active_movies - enriched_movies
    missing_images = active_movies - image_files
    
    print(f"   - Movies in ratings but MISSING TEXT: {len(missing_text)} ({len(missing_text)/len(active_movies):.2%})")
    print(f"   - Movies in ratings but MISSING IMAGE: {len(missing_images)} ({len(missing_images)/len(active_movies):.2%})")
    
    # 5. User Density Check (Cold Start sanity)
    user_counts = df_ratings['userId'].value_counts()
    min_inter = user_counts.min()
    print(f"\n👥 USER DENSITY:")
    print(f"   - Min interactions per user: {min_inter}")
    if min_inter < 5:
        print("   ⚠️  WARNING: Some users have very few interactions. TGN might struggle.")
    else:
        print("   ✅ User density looks good (>= 5).")

    # 6. Final Verdict
    print(f"\n🎯 FINAL VERDICT:")
    if len(missing_images) > 0.5 * len(active_movies):
        print("   ❌ STOP: You are missing images for >50% of your active movies.")
    elif len(missing_text) > 0.5 * len(active_movies):
        print("   ❌ STOP: You are missing text for >50% of your active movies.")
    else:
        print("   ✅ GO AHEAD: Data is consistent enough for embeddings.")
        print("      (Missing items will be handled by Zero-Padding in the encoder script)")

if __name__ == "__main__":
    verify_dataset()