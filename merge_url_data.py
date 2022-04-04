# Merge Inversion blocklist into url_data.csv

from more_itertools import sort_together

with open('url_data.csv', 'r') as g:
    kaggle_urls_and_labels: list[list[str]] = [row.rsplit(',', 1) for row in g.read().splitlines()[1:]]
kaggle_urls, kaggle_labels = zip(*kaggle_urls_and_labels)

# Inversion blocklist
with open('Google_hostnames.txt', 'r') as h:
    inversion_urls: set[str] = set(h.read().splitlines())

urls_and_labels: list[list[str]] = []

for url, label in zip(kaggle_urls, kaggle_labels):
    if url not in inversion_urls:
        urls_and_labels.append([url, label])
for url in inversion_urls:
    urls_and_labels.append([url, "bad"])
urls, labels = zip(*urls_and_labels)

(urls, labels) = sort_together((urls, labels))

with open('url_data_combined.csv', 'w') as f:
    f.write("url,label\n")
    for url, label in zip(urls, labels):
        f.write(f"{url.strip()},{label.strip()}\n")
