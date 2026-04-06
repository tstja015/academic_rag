#!/bin/bash
set -e
echo 'Setting up conda environment: rag'
conda create -n rag python=3.11 -y
echo 'Activating environment...'
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rag
echo 'Installing dependencies...'
pip install --upgrade pip
pip install -r requirements.txt
echo ''
echo 'Done. Next steps:'
echo '  conda activate rag'
echo '  Edit config.py -- add your PDF paths to PAPER_DIRS'
echo '  python ingest.py'
echo '  python query.py'