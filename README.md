# Malicious URLs detection with autoencoder neural network

This repository contains the source code of Detecting malicious URLs using an autoencoder neural network.
An article describing how it works is available at https://www.linkedin.com/pulse/anomaly-detection-autoencoder-neural-network-applied-urls-daboubi/

## Requirements

- Python 3.9
- x64 CPU
- Tensorflow-compatible NVIDIA GPU

## Install required libraries

```bash
pip3 install -r requirements.txt
```

## Merge Inversion blocklist (Google_hostnames.txt) with url_data.csv

```bash
python merge_url_data.py
```

## Generated new enriched data

```bash
python enrich_urls_data.py
```

## Build and test a model

```bash
python train_and_test_urls_autoencoder.py
```

## Dataset sources

- https://www.kaggle.com/datasets/antonyj453/urldataset
- https://www.kaggle.com/datasets/dfydata/the-online-plain-text-english-dictionary-opted
- https://github.com/elliotwutingfeng/Inversion-DNSBL-Blocklists
