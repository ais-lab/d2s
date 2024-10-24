OUTPUT_FILE="logs.zip"
FILE_ID="1zQkHChBffXX_ZIu8ytiWuG19Mnr9pBuO"

# Download the file from Google Drive using gdown and save it in the target folder
gdown --id $FILE_ID -O $OUTPUT_FILE

# Unzip the downloaded file in the target folder
unzip $OUTPUT_FILE

# Remove the zip file after extraction
rm $OUTPUT_FILE

echo "Download, extraction, and cleanup completed."