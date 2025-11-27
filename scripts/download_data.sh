#!/bin/bash

# Script to download a Google Drive folder recursively
# Usage: bash scripts/download_data.sh

# Folder ID of 'Multimodal_project/data' on Google Drive (replace this)
FOLDER_ID=https://drive.google.com/drive/folders/1ZVv7lAI-7-YLSrdbgtkGn8LiveTrSpUf?usp=drive_link

# Destination folder
DEST_DIR="data"

# Create destination directory if not exists
mkdir -p ${DEST_DIR}

# Check if gdown is installed, if not prompt to install
if ! command -v gdown &> /dev/null
then
    echo "gdown is not installed. Please install it with:"
    echo "pip install gdown"
    exit 1
fi

echo "Downloading Google Drive folder into ${DEST_DIR}..."
gdown --folder --id ${FOLDER_ID} -O ${DEST_DIR}

echo "All downloads finished!"
