# Malicious URLs detection with autoencoder neural network

This repository contains the code source of Detecting malicious URLs using an autoencoder neural network.
An article descrbing how it works are available at https://www.linkedin.com/pulse/anomaly-detection-autoencoder-neural-network-applied-urls-daboubi/

## Merge Inversion blocklist into url_data.csv

```python merge_url_data.py```

## To build and test a model you can run

```python train_and_test_urls_autoencoder.py```

## If you would like to generated new enriched data you can run

```python enrich_urls_data.py```

## Dataset sources

- https://www.kaggle.com/datasets/antonyj453/urldataset
- https://www.kaggle.com/datasets/dfydata/the-online-plain-text-english-dictionary-opted
- https://github.com/elliotwutingfeng/Inversion-DNSBL-Blocklists
