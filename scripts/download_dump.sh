#!/bin/bash
# Download Wikipedia pagelinks SQL dump.
#
# This grabs the "pagelinks" table from the latest English Wikipedia dump.
# Warning: the compressed file is ~6GB, uncompressed is much larger.
# For testing, you can use a smaller wiki (e.g., Simple English).
#
# Usage:
#   bash download_dump.sh [lang]    # lang defaults to "simplewiki"

set -euo pipefail

LANG="${1:-simplewiki}"  # use simplewiki for manageable size
DATE="latest"
BASE_URL="https://dumps.wikimedia.org/${LANG}/${DATE}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
mkdir -p "$DATA_DIR"

echo "Downloading ${LANG} pagelinks dump..."
PAGELINKS_FILE="${LANG}-${DATE}-pagelinks.sql.gz"
wget -c "${BASE_URL}/${PAGELINKS_FILE}" -O "${DATA_DIR}/${PAGELINKS_FILE}" || {
    echo "Download failed. Generating synthetic graph for testing..."
    python3 "${SCRIPT_DIR}/generate_synthetic.py" "${DATA_DIR}/edges.csv"
    exit 0
}

echo "Downloaded to ${DATA_DIR}/${PAGELINKS_FILE}"
echo "Next: python3 scripts/parse_dump.py ${DATA_DIR}/${PAGELINKS_FILE}"
