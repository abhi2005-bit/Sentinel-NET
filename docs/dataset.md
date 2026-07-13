# Dataset

Use the ULB Credit Card Fraud Detection dataset at `datasets/creditcard.csv`.

Required columns:

`Time`, `V1` through `V28`, `Amount`, `Class`

The loader in `src/data/data_loader.py` validates existence, columns, missing
values, class distribution, and stratified splitting.
