# Version: 1.0 - 2018/06/29
# Contact: walid.daboubi@gmail.com

import pandas as pd
import ray
from more_itertools import chunked, flatten

from parallel_compute import execute_with_ray

NUMBERS = set("0123456789")
SPEC_CHARS = set(["+", '"', "*", "#", "%", "&", "(", ")", "=", "?", "^", "-", ".", "!", "~", "_", ">", "<"])


def create_dictionary_words() -> set[str]:
    df = pd.read_csv('OPTED-Dictionary.csv')
    dictionary_words: set[str] = set(df['Word'].apply(str))

    return dictionary_words


async def enrich_row_chunk(chunk, dictionary_words):
    enriched_rows = []
    for row in chunk:
        label = "1" if "bad" in row[1].lower() else "0"
        spec_chars = 0
        depth = 0
        numericals_count = 0
        word_count = 0
        url = str(row[0])

        url_lower = url.lower()
        word_count = len(set([word for word in dictionary_words if word in url_lower]))
        for c in url:
            if c in SPEC_CHARS:
                spec_chars += 1
            elif c in ["/"]:
                depth += 1
            elif c in NUMBERS:
                numericals_count += 1
        enriched_rows.append(f"{len(url)},{spec_chars},0,{depth},{numericals_count},{word_count},{label}")
    return enriched_rows


if __name__ == "__main__":

    dictionary_words = create_dictionary_words()

    rows = (list(row) for row in pd.read_csv("url_data_combined.csv").itertuples(index=False))
    row_chunks = [(chunk,) for chunk in chunked(rows, 1000)]

    ray.shutdown()
    ray.init(include_dashboard=False)
    data = ["len,spec_chars,domain,depth,numericals_count,word_count,label"] + list(
        flatten(execute_with_ray(enrich_row_chunk, row_chunks, object_store={"dictionary_words": dictionary_words}))
        )
    with open("url_enriched_data.csv", "w") as enriched_csv:
        enriched_csv.write('\n'.join(data))
    ray.shutdown()
