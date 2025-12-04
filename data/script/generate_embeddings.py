import os
import re
import ast
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional, Union
import open_clip
from sentence_transformers import SentenceTransformer

# --- CHECK IMAGEBIND AVAILABILITY ---
try:
    from imagebind import data
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType
    IMAGEBIND_AVAILABLE = True
except ImportError:
    IMAGEBIND_AVAILABLE = False

# --- MODEL REGISTRY ---
TEXT_MODELS = {
    "baseline": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "type": "sbert",
        "desc": "Standard SBERT (Fast, Reliable)"
    },
    "efficient": {
        "name": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "type": "sbert",
        "desc": "Alibaba Qwen2 1.5B (SOTA efficient)",
        "trust_remote": False  # Fixed for stability
    },
    "imagebind": {
        "name": "imagebind_huge",
        "type": "imagebind",
        "desc": "Meta ImageBind (Aligned Space)"
    }
}

IMAGE_MODELS = {
    "clip": {
        "name": "ViT-L-14",
        "pretrained": "laion2b_s32b_b82k",
        "type": "open_clip",
        "desc": "OpenAI CLIP ViT-L"
    },
    "siglip": {
        "name": "ViT-SO400M-14-SigLIP", 
        "pretrained": "webli",
        "type": "open_clip",
        "desc": "Google SigLIP (SOTA Vision)"
    },
    "imagebind": {
        "name": "imagebind_huge",
        "type": "imagebind",
        "desc": "Meta ImageBind (Aligned Space)"
    }
}

class UniversalEncoder:
    def __init__(self, text_key: str, image_key: str, device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Initializing Encoders on {self.device}...")
        
        self.txt_conf = TEXT_MODELS[text_key]
        self.img_conf = IMAGE_MODELS[image_key]

        # 1. Initialize Text Encoder
        if self.txt_conf['type'] == 'sbert':
            print(f"   - Loading Text: {self.txt_conf['name']}")
            self.text_model = SentenceTransformer(
                self.txt_conf['name'], 
                trust_remote_code=self.txt_conf.get('trust_remote', False),
                device=self.device
            )
        
        # 2. Initialize Image Encoder (OpenCLIP)
        if self.img_conf['type'] == 'open_clip':
            print(f"   - Loading Image: {self.img_conf['name']}")
            self.img_model, _, self.img_preprocess = open_clip.create_model_and_transforms(
                self.img_conf['name'], 
                pretrained=self.img_conf['pretrained'], 
                device=self.device
            )
            self.img_model.eval()
            with torch.no_grad():
                d = torch.randn(1, 3, 224, 224).to(self.device)
                self.img_dim = self.img_model.encode_image(d).shape[-1]

        # 3. Initialize ImageBind
        if 'imagebind' in [self.txt_conf['type'], self.img_conf['type']]:
            if not IMAGEBIND_AVAILABLE:
                raise ImportError("❌ ImageBind not installed! Run: pip install git+https://github.com/facebookresearch/ImageBind.git")
            
            print(f"   - Loading ImageBind (Huge)...")
            self.ib_model = imagebind_model.imagebind_huge(pretrained=True)
            self.ib_model.eval()
            self.ib_model.to(self.device)
            self.img_dim = 1024

    def encode_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        print(f"   - Encoding {len(texts)} texts...")
        
        if self.txt_conf['type'] == 'sbert':
            if "Qwen" in self.txt_conf['name']:
                texts = [f"Instruct: Retrieve semantic text embeddings.\nQuery: {t}" for t in texts]
            
            return self.text_model.encode(
                texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
            )
            
        elif self.txt_conf['type'] == 'imagebind':
            all_embs = []
            for i in tqdm(range(0, len(texts), batch_size), desc="Text Batches"):
                batch_texts = texts[i : i+batch_size]
                inputs = {ModalityType.TEXT: data.load_and_transform_text(batch_texts, self.device)}
                with torch.no_grad():
                    out = self.ib_model(inputs)
                    feats = out[ModalityType.TEXT]
                    feats /= feats.norm(dim=-1, keepdim=True)
                    all_embs.append(feats.cpu().numpy())
            return np.concatenate(all_embs, axis=0)

    def encode_images(self, image_paths: List[Path], batch_size: int = 32) -> np.ndarray:
        print(f"   - Encoding {len(image_paths)} images...")
        all_embs = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Image Batches"):
            batch_paths = image_paths[i : i+batch_size]
            valid_paths = [str(p) for p in batch_paths if Path(p).exists()]
            valid_indices = [idx for idx, p in enumerate(batch_paths) if Path(p).exists()]
            
            batch_out = np.zeros((len(batch_paths), self.img_dim), dtype=np.float32)
            
            if valid_paths:
                with torch.no_grad():
                    if self.img_conf['type'] == 'open_clip':
                        imgs = [Image.open(p).convert('RGB') for p in valid_paths]
                        tensors = torch.stack([self.img_preprocess(img) for img in imgs]).to(self.device)
                        feats = self.img_model.encode_image(tensors)
                        
                    elif self.img_conf['type'] == 'imagebind':
                        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(valid_paths, self.device)}
                        feats = self.ib_model(inputs)[ModalityType.VISION]

                    feats /= feats.norm(dim=-1, keepdim=True)
                    feats_np = feats.cpu().numpy()

                    for v_idx, f_vec in zip(valid_indices, feats_np):
                        batch_out[v_idx] = f_vec
            
            all_embs.append(batch_out)
            
        return np.concatenate(all_embs, axis=0)

def main():
    parser = argparse.ArgumentParser(description="Universal Multimodal Embedding Generator")
    # Dataset Config
    parser.add_argument("--csv-path", required=True, help="Path to metadata CSV (e.g., enriched.csv)")
    parser.add_argument("--image-dir", required=True, help="Directory containing images (e.g., posters/)")
    parser.add_argument("--output-dir", required=True, help="Where to save .npy files")
    parser.add_argument("--dataset-name", required=True, help="Prefix for output files (e.g., amazon-sports)")
    # Column Mapping
    parser.add_argument("--id-col", required=True, help="Column name for Item ID (e.g., movieId, asin)")
    parser.add_argument("--text-col", required=True, help="Column name for Text (e.g., overview, description)")
    parser.add_argument("--text-model", choices=TEXT_MODELS.keys(), default="baseline")
    parser.add_argument("--image-model", choices=IMAGE_MODELS.keys(), default="clip")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Loading Data from: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    
     # --- VALIDATION BLOCK ---

    if args.id_col not in df.columns:
        raise ValueError(f"❌ ID Column '{args.id_col}' not found in CSV. Available: {list(df.columns)}")
    if args.text_col not in df.columns:
        print(f"⚠️  Warning: Primary text column '{args.text_col}' not found. Will rely on metadata (Title/Genre/Year).")
    
    # -----------------------------------------------------------
    # UNIVERSAL METADATA ENRICHMENT
    # -----------------------------------------------------------
    print("📝 Augmenting Text with Metadata (Universal Logic)...")
    
    def parse_stringified_list(value: Union[str, list]) -> str:
        """
        Robustly parse stringified Python lists from CSV columns.
        
        Handles Amazon's category format: "['Sports', 'Gear']"
        Also handles: nested lists, pipe-separated, and plain strings.
        
        Args:
            value: String like "['Cat1', 'Cat2']" or actual list or plain string
        
        Returns:
            Cleaned comma-separated string: "Cat1, Cat2"
        """
        if isinstance(value, list):
            # Already a list - flatten if nested
            flat = []
            for item in value:
                if isinstance(item, list):
                    flat.extend(str(x) for x in item)
                else:
                    flat.append(str(item))
            return ", ".join(flat)
        
        val_str = str(value).strip()
        
        # Handle nan/empty
        if val_str.lower() in ('nan', 'none', ''):
            return ''
        
        # Handle pipe-separated (MovieLens style): "Action|Comedy|Drama"
        if '|' in val_str and not val_str.startswith('['):
            return val_str.replace('|', ', ')
        
        # Try parsing as Python literal (safe for lists/tuples/strings)
        if val_str.startswith('[') or val_str.startswith('('):
            try:
                parsed = ast.literal_eval(val_str)
                # Recursively handle nested lists
                return parse_stringified_list(parsed)
            except (ValueError, SyntaxError):
                # Fallback: regex to extract quoted strings
                # Matches 'text' or "text" patterns
                matches = re.findall(r"['\"]([^'\"]+)['\"]", val_str)
                if matches:
                    return ", ".join(matches)
                # Last resort: strip brackets and clean
                cleaned = val_str.strip('[]()').replace("'", "").replace('"', '')
                return cleaned
        
        # Plain string - return as-is
        return val_str
    
    def create_rich_text(row):
        # 1. TITLE (Common across all datasets)
        # Check standard variations: tmdb_title, ml_title, title, name
        title = str(row.get('tmdb_title', row.get('ml_title', row.get('title', row.get('name', '')))))
        if title.lower() == 'nan': title = ''
        
        # 2. YEAR / TIME
        # MovieLens: ml_year | Goodreads: publication_year | Amazon: usually missing
        year = str(row.get('ml_year', row.get('publication_year', row.get('year', ''))))
        if year.lower() == 'nan': year = ''
        
        # 3. GENRES / CATEGORIES (with robust parsing)
        # MovieLens: ml_genres (pipe-sep) | Amazon: categories (stringified list) | Goodreads: genres
        raw_cats = row.get('tmdb_genres', row.get('categories', row.get('genres', '')))
        cats = parse_stringified_list(raw_cats)
            
        # 4. EXTRAS (Brand, Author, Price range, etc.)
        extras = []
        
        # Amazon Brand
        brand = str(row.get('brand', ''))
        if brand.lower() not in ('nan', ''):
            extras.append(f"Brand: {brand}")
        
        # Goodreads Author
        authors = str(row.get('authors', row.get('author', '')))
        if authors.lower() not in ('nan', ''):
            # Authors can also be stringified lists
            authors_clean = parse_stringified_list(authors)
            if authors_clean:
                extras.append(f"Author: {authors_clean}")
        
        # Amazon Price (useful context)
        price = str(row.get('price', ''))
        if price.lower() not in ('nan', '') and price != '0':
            extras.append(f"Price: ${price}")

        # 5. MAIN DESCRIPTION/TEXT
        main_text = str(row.get(args.text_col, ''))
        if main_text.lower() == 'nan': main_text = ''
        
        # ================================================================
        # SMART TEXT HANDLING FOR MINIMAL AMAZON DATA
        # If we only have the main text field (no title, categories, etc.)
        # treat the text as the product name/title for better embeddings
        # ================================================================
        has_metadata = bool(title or year or cats or extras)
        
        # CONSTRUCT PROMPT
        parts = []
        
        if has_metadata:
            # Rich metadata available (MovieLens, enriched Amazon, etc.)
            if title: parts.append(f"Title: {title}")
            if year: parts.append(f"Year: {year}")
            if cats: parts.append(f"Categories: {cats}")
            parts.extend(extras)
            if main_text: parts.append(f"Description: {main_text}")
        else:
            # Minimal data (Amazon raw_text only) - treat as product info
            # The raw_text is typically: "Product Name - Category" or just product name
            if main_text:
                # Try to extract category hints from hyphen-separated parts
                # e.g., "BenchMaster Pocket Guide - Fly Fishing - Fishing"
                if ' - ' in main_text:
                    text_parts = [p.strip() for p in main_text.split(' - ')]
                    product_name = text_parts[0]
                    potential_cats = text_parts[1:] if len(text_parts) > 1 else []
                    
                    parts.append(f"Product: {product_name}")
                    if potential_cats:
                        parts.append(f"Category: {', '.join(potential_cats)}")
                else:
                    # Plain product name
                    parts.append(f"Product: {main_text}")
        
        return ". ".join(parts) if parts else "Unknown product"

    texts = df.apply(create_rich_text, axis=1).tolist()
    ids = df[args.id_col].astype(str).tolist()
    img_paths = [Path(args.image_dir) / f"{iid}.jpg" for iid in ids]
    
    # Run Encoding
    encoder = UniversalEncoder(args.text_model, args.image_model)
    print("🔄 Generating Text Embeddings...")
    txt_emb = encoder.encode_text(texts, batch_size=args.batch_size)
    print("🔄 Generating Image Embeddings...")
    img_emb = encoder.encode_images(img_paths, batch_size=args.batch_size)
    
    print("💾 Saving Files...")
    np.save(out_dir / f"{args.dataset_name}_text_{args.text_model}.npy", txt_emb)
    np.save(out_dir / f"{args.dataset_name}_image_{args.image_model}.npy", img_emb)
    np.save(out_dir / f"{args.dataset_name}_ids.npy", np.array(ids))
    
    print(f"\n✅ Done! Saved to {out_dir}")
    print(f"   - IDs: {len(ids)}")
    print(f"   - Text Shape: {txt_emb.shape}")
    print(f"   - Image Shape: {img_emb.shape}")

if __name__ == "__main__":
    main()