Rscript R/processo_arrivi.r

cd Python/pl_euristica_offline
pip install -r requirements.txt
python3 main.py

cd ../pl_online
pip install -r requirements.txt
python3 main.py