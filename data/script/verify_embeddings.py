import numpy as np
import os
import argparse
from pathlib import Path

def verify(dataset, text_model, image_model, features_dir):
    print(f"🔍 Verifying: {dataset}")
    print(f"   - Models: Text='{text_model}' | Image='{image_model}'")
    print(f"   - Dir:    {features_dir}")
    
    base_path = Path(features_dir)
    
    # Construct filenames based on the convention from generate_features.py
    txt_path = base_path / f"{dataset}_text_{text_model}.npy"
    img_path = base_path / f"{dataset}_image_{image_model}.npy"
    ids_path = base_path / f"{dataset}_ids.npy"
    
    # 1. Check Existence
    missing = []
    for p in [txt_path, img_path, ids_path]:
        if not p.exists():
            missing.append(str(p))
    
    if missing:
        print("\n❌ CRITICAL: Missing files!")
        for m in missing:
            print(f"   - {m}")
        return

    # 2. Load Data
    try:
        txt = np.load(txt_path)
        img = np.load(img_path)
        ids = np.load(ids_path)
    except Exception as e:
        print(f"\n❌ Error loading .npy files: {e}")
        return
    
    # 3. Check Shapes
    print(f"\n📊 SHAPE CHECK:")
    print(f"   - IDs:   {ids.shape[0]}")
    print(f"   - Text:  {txt.shape} (N, Dim)")
    print(f"   - Image: {img.shape} (N, Dim)")
    
    # Validation logic
    rows_match = (txt.shape[0] == img.shape[0] == ids.shape[0])
    if not rows_match:
        print("❌ ALIGNMENT ERROR: Row counts do not match!")
        return
    else:
        print("✅ Alignment OK (Rows match).")

    # 4. Check Zero Vectors (Missing Data Handling)
    # A vector is "missing" if its norm is 0 (or very close to it)
    zero_rows_img = np.where(~img.any(axis=1))[0]
    zero_rows_txt = np.where(~txt.any(axis=1))[0]
    
    print(f"\nEMPTY/MISSING DATA CHECK:")
    print(f"   - Missing Images (Zero Vectors): {len(zero_rows_img)} ({len(zero_rows_img)/len(img):.2%})")
    print(f"   - Missing Texts (Zero Vectors):  {len(zero_rows_txt)} ({len(zero_rows_txt)/len(txt):.2%})")
    
    if len(zero_rows_img) > 0:
        print(f"     -> Sample missing indices: {zero_rows_img[:5]}")

    # 5. Value Distribution (Sanity Check)
    print(f"\nSTATS CHECK (Should be non-zero / normalized):")
    # Avoid mean of empty array if everything is zero
    if len(txt) > 0:
        print(f"   - Text Mean: {np.mean(txt):.4f} | Min: {np.min(txt):.4f} | Max: {np.max(txt):.4f}")
        print(f"   - Image Mean: {np.mean(img):.4f} | Min: {np.min(img):.4f} | Max: {np.max(img):.4f}")

    print("\n✅ Verification Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify generated .npy embedding files")
    
    parser.add_argument("--features-dir", required=True, help="Path to the folder containing .npy files")
    parser.add_argument("--dataset", required=True, help="Dataset prefix (e.g., ml-modern)")
    parser.add_argument("--text-model", default="baseline", help="Text model suffix (e.g., baseline, efficient)")
    parser.add_argument("--image-model", default="clip", help="Image model suffix (e.g., clip, siglip)")
    
    args = parser.parse_args()
    
    verify(args.dataset, args.text_model, args.image_model, args.features_dir)