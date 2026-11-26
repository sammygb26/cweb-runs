cd $HOME/cweb-runs

python3 -m venv venv --prompt "cweb-runs"

source venv/bin/activate

pip install -r ./cweb/requirements.txt
pip install -e ./cweb/