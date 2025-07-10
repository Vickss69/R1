#!/bin/bash
apt-get update
apt-get install -y build-essential cmake libopenblas-dev liblapack-dev
python -m pip install --upgrade pip
pip install -r requirements.txt