# Data

## Included: ODDS Benchmark Datasets

The `odds/` folder contains 7 datasets from the ODDS library (Rayana, 2016) in MATLAB `.mat` format:

| File | Samples | Features | Anomaly Rate |
|------|---------|----------|--------------|
| thyroid.mat | 3,772 | 6 | 2.5% |
| cardio.mat | 1,831 | 21 | 9.6% |
| arrhythmia.mat | 452 | 274 | 14.6% |
| ionosphere.mat | 351 | 33 | 35.9% |
| pima.mat | 768 | 8 | 34.9% |
| wbc.mat | 378 | 30 | 5.6% |
| vowels.mat | 1,456 | 12 | 3.4% |

Source: http://odds.cs.stonybrook.edu/

## Not Included: Large Kaggle Datasets

The following datasets are too large to include. Download them and place in a `kaggle/` subfolder:

1. **SAML-D** (~950 MB)
   - https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml
   - Download `SAML-D.csv` -> `kaggle/SAML-D.csv`

2. **IEEE-CIS Fraud Detection** (~300 MB)
   - https://www.kaggle.com/competitions/ieee-fraud-detection
   - Download and extract -> `kaggle/IEEE-CIS Fraud Detection/train_transaction.csv`

3. **PaySim** (~470 MB)
   - https://www.kaggle.com/datasets/ealaxi/paysim1
   - Download CSV -> `kaggle/Synthetic Financial Datasets For Fraud Detection.csv`

4. **Credit Card Fraud** (~144 MB)
   - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   - Download -> `kaggle/Credit Card Fraud Detection.csv`
