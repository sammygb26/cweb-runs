#!/bin/bash

python3.11 -m venv venv --prompt "cweb-runs"

source venv/bin/activate

pip install -r ./cweb/requirements.txt
pip install -e ./cweb/