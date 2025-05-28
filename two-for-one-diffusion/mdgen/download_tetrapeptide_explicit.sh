#!/bin/bash

# Set variables
BUCKET_PATH="gs://mdgen-public/4AA_sims"
REMOTE_USER="sanjeevr"
REMOTE_HOST="escher.lbl.gov"
REMOTE_DIR="/data/sanjeevr/4AA_sim"

# Create a temporary local directory for downloads
LOCAL_TEMP_DIR="temp_4AA_sims"
mkdir -p "$LOCAL_TEMP_DIR"

# Download all files one at a time
gsutil ls "$BUCKET_PATH" | while read -r file; do
    echo "Processing file: $file"
    
    # Download the file to the temporary local directory
    gsutil cp "$file" "$LOCAL_TEMP_DIR"

    # Extract the file name from the full path
    file_name=$(basename "$file")
    local_file="$LOCAL_TEMP_DIR/$file_name"

    # SCP the file to the remote server
    scp "$local_file" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"

    # Check if SCP was successful before deleting the file
    if [ $? -eq 0 ]; then
        echo "Successfully transferred $file_name. Deleting local copy."
        rm "$local_file"
    else
        echo "Failed to transfer $file_name. Keeping local copy for debugging."
    fi
done

# Cleanup: Remove the temporary directory if it's empty
rmdir "$LOCAL_TEMP_DIR" 2>/dev/null
