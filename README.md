# 🕸️ MM-TGN: Multimodal Temporal Graph Network Pipeline

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-Temporal-green.svg)](https://pyg.org/)
[![Status](https://img.shields.io/badge/Status-Data_Engineering_Complete-success.svg)]()

A robust, end-to-end data engineering framework for **Multimodal Temporal Graph Networks (MM-TGN)**. This repository handles the ingestion, enrichment, and feature extraction of large-scale temporal interaction datasets (MovieLens, Amazon, Goodreads) using State-of-the-Art (SOTA) Vision-Language models.

---

## 🚀 Key Features

* **Universal Architecture:** A single pipeline agnostic to domain (Movies, Books, E-commerce).
* **SOTA Feature Extraction:** Integrated support for:
    * **LLMs:** Alibaba Qwen2-1.5B (Instruction-tuned text embeddings).
    * **Dense Vision:** Google SigLIP (Sigmoid Loss for Language Image Pre-training).
    * **Unified Modality:** Meta ImageBind (Aligned audio/visual/text space).
* **Fault Tolerance:** Async scrapers with rate-limit handling and zero-padding logic for missing multimodal assets (<0.03% data loss).
* **TGN-Ready:** Automatically formats outputs for temporal graph learning (Node Features + Edge Lists).

---

## 🏗️ Pipeline Architecture

```mermaid
graph LR
    A[Raw Data] --> B(Preprocessing & Filtering)
    B --> C{Async Scraper}
    C -->|Posters| D[Image Storage]
    C -->|Plot Summaries| E[Enriched Metadata]
    D & E --> F[Universal Encoder]
    F -->|Qwen/SigLIP/ImageBind| G[Feature Matrices .npy]
    G --> H[TGN Formatter]
    H --> I((Ready for Training))
