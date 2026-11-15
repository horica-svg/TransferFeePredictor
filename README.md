# TransferFeePredictor

LightGBM model predicting player market value (EUR) from Transfermarkt-style CSVs.

1. Put the raw CSVs in `data/raw/` (not tracked by git).
2. `pip install -r requirements.txt`
3. From the project root: `python src/build_dataset.py`, then `python src/train.py`.

Output: `data/processed/dataset_processed.csv`, `models/lgbm_market_value.pkl`.
