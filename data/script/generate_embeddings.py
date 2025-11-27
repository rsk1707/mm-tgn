import os
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional
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
        "trust_remote": False
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
            # Determine dimension dynamically
            with torch.no_grad():
                d = torch.randn(1, 3, 224, 224).to(self.device)
                self.img_dim = self.img_model.encode_image(d).shape[-1]

        # 3. Initialize ImageBind (Monolithic Model)
        if 'imagebind' in [self.txt_conf['type'], self.img_conf['type']]:
            if not IMAGEBIND_AVAILABLE:
                raise ImportError("❌ ImageBind not installed! Run: pip install git+https://github.com/facebookresearch/ImageBind.git")
            
            print(f"   - Loading ImageBind (Huge)...")
            # We load one shared model instance
            self.ib_model = imagebind_model.imagebind_huge(pretrained=True)
            self.ib_model.eval()
            self.ib_model.to(self.device)
            self.img_dim = 1024

    def encode_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        print(f"   - Encoding {len(texts)} texts...")
        
        if self.txt_conf['type'] == 'sbert':
            # Add instruction for Qwen models to boost performance
            if "Qwen" in self.txt_conf['name']:
                texts = [f"Instruct: Retrieve semantic text embeddings.\nQuery: {t}" for t in texts]
            
            return self.text_model.encode(
                texts, 
                batch_size=batch_size, 
                show_progress_bar=True, 
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
        elif self.txt_conf['type'] == 'imagebind':
            all_embs = []
            for i in tqdm(range(0, len(texts), batch_size), desc="Text Batches"):
                batch_texts = texts[i : i+batch_size]
                inputs = {ModalityType.TEXT: data.load_and_transform_text(batch_texts, self.device)}
                with torch.no_grad():
                    out = self.ib_model(inputs)
                    # Normalize
                    feats = out[ModalityType.TEXT]
                    feats /= feats.norm(dim=-1, keepdim=True)
                    all_embs.append(feats.cpu().numpy())
            return np.concatenate(all_embs, axis=0)

    def encode_images(self, image_paths: List[Path], batch_size: int = 32) -> np.ndarray:
        print(f"   - Encoding {len(image_paths)} images...")
        all_embs = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Image Batches"):
            batch_paths = image_paths[i : i+batch_size]
            
            # Filter for valid images in this batch
            valid_paths = [str(p) for p in batch_paths if Path(p).exists()]
            valid_indices = [idx for idx, p in enumerate(batch_paths) if Path(p).exists()]
            
            # Prepare Output Batch (Zero-filled)
            batch_out = np.zeros((len(batch_paths), self.img_dim), dtype=np.float32)
            
            if valid_paths:
                with torch.no_grad():
                    if self.img_conf['type'] == 'open_clip':
                        # OpenCLIP Preprocessing
                        imgs = [Image.open(p).convert('RGB') for p in valid_paths]
                        tensors = torch.stack([self.img_preprocess(img) for img in imgs]).to(self.device)
                        feats = self.img_model.encode_image(tensors)
                        
                    elif self.img_conf['type'] == 'imagebind':
                        # ImageBind Preprocessing
                        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(valid_paths, self.device)}
                        feats = self.ib_model(inputs)[ModalityType.VISION]

                    # Normalize & Move to CPU
                    feats /= feats.norm(dim=-1, keepdim=True)
                    feats_np = feats.cpu().numpy()

                    # Map back to original batch positions
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
    
    # Model Config
    parser.add_argument("--text-model", choices=TEXT_MODELS.keys(), default="baseline")
    parser.add_argument("--image-model", choices=IMAGE_MODELS.keys(), default="clip")
    parser.add_argument("--batch-size", type=int, default=32)
    
    args = parser.parse_args()
    
    # Setup
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Loading Data from: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    
    # Validation
    if args.id_col not in df.columns:
        raise ValueError(f"ID Column '{args.id_col}' not found in CSV. Available: {list(df.columns)}")
    if args.text_col not in df.columns:
        print(f"⚠️ Warning: Text Column '{args.text_col}' not found. Filling with empty strings.")
        texts = [""] * len(df)
    else:
        texts = df[args.text_col].fillna("").astype(str).tolist()
        
    ids = df[args.id_col].astype(str).tolist()
    
    # Assume image filename is simply "{ID}.jpg"
    # This is the standard we enforced in the cleaning step
    img_paths = [Path(args.image_dir) / f"{iid}.jpg" for iid in ids]
    
    # Run Encoding
    encoder = UniversalEncoder(args.text_model, args.image_model)
    
    print("🔄 Generating Text Embeddings...")
    txt_emb = encoder.encode_text(texts, batch_size=args.batch_size)
    
    print("🔄 Generating Image Embeddings...")
    img_emb = encoder.encode_images(img_paths, batch_size=args.batch_size)
    
    # Save
    print("💾 Saving Files...")
    base_name = f"{args.dataset_name}_{args.text_model}_{args.image_model}"
    
    np.save(out_dir / f"{args.dataset_name}_text_{args.text_model}.npy", txt_emb)
    np.save(out_dir / f"{args.dataset_name}_image_{args.image_model}.npy", img_emb)
    np.save(out_dir / f"{args.dataset_name}_ids.npy", np.array(ids))
    
    print(f"\n✅ Done! Saved to {out_dir}")
    print(f"   - IDs: {len(ids)}")
    print(f"   - Text Shape: {txt_emb.shape}")
    print(f"   - Image Shape: {img_emb.shape}")

if __name__ == "__main__":
    main()