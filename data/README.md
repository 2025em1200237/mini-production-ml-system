# Data

`paysim.csv` (about 470 MB) is not included. Download it from Kaggle:
"Synthetic Financial Datasets For Fraud Detection" (ealaxi/paysim1), rename the
file to `paysim.csv` and place it in this folder. A copy one level up
(`../data/paysim.csv`) is also picked up automatically.

Included here:

- `sample_transactions.csv` - 2,300 rows drawn from the test window (steps > 600):
  40 fraud, 1,960 legitimate in-scope, 300 out-of-scope. Used by the benchmark and the demos.
- `sample_with_bad_rows.csv` - 20 rows, three of them deliberately invalid, to show the
  rejects path of the batch CLI.
