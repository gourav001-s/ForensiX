#!/bin/bash
# Deepfake & Image Forensics Toolkit - setup (Debian/Ubuntu/Kali)
# Developed by Gourav Shrivastava
set -e
echo "Installing OPTIONAL system forensic tools (the app works without them)..."
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip libimage-exiftool-perl binwalk steghide ruby ruby-dev build-essential ffmpeg || true
sudo gem install zsteg || echo "zsteg install skipped"

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

echo "Optional ML classifier:  pip install -r requirements-ml.txt  (see models/README.txt)"
echo "Optional C2PA verification is bundled in the same requirements-ml.txt (c2pa-python)"
echo "Setup complete. Run:  source venv/bin/activate && python3 app.py"
echo "Open http://127.0.0.1:5001"
