import sys
import re
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import pandas as pd
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import pyarrow as pa
import pyarrow.parquet as pq
from unstructured.partition.html import partition_html
from unstructured.cleaners.core import (
    clean,
    clean_non_ascii_chars,
    replace_unicode_quotes,
    group_broken_paragraphs
)
from lxml import etree

def remove_processing_instructions(html_text):
    """Remove processing instructions from HTML content using lxml."""
    try:
        parser = etree.HTMLParser(remove_pis=True)
        tree = etree.fromstring(html_text.encode('utf-8'), parser)
        return etree.tostring(tree, encoding='unicode', method='html')
    except Exception as e:
        # If parsing fails, return the original text
        tqdm.write(f"Error removing processing instructions: {e}", file=sys.stderr)
        return html_text

def process_document(doc):
    try:
        # Extract the 'description' field
        description = doc.get('_source', {}).get('description', '')
        description = remove_processing_instructions(description)

        # Process 'description' field using unstructured.io
        try:
            elements = partition_html(text=description, chunking_strategy='by_title')
        except Exception as e:
            tqdm.write(f"Exception in partition_html for doc {doc.get('_id')}: {e}", file=sys.stderr)
            return None

        docs = []
        chunk_count = len(elements)

        for i, element in enumerate(elements):
            try:
                # Skip elements that are processing instructions
                if hasattr(element, 'element') and isinstance(element.element, etree._ProcessingInstruction):
                    continue

                if hasattr(element, 'text') and element.text:
                    cleaned_text = clean(
                        element.text,
                        extra_whitespace=True,
                        dashes=True,
                        bullets=True,
                        trailing_punctuation=False,
                        lowercase=False
                    )
                    cleaned_text = clean_non_ascii_chars(cleaned_text)
                    cleaned_text = replace_unicode_quotes(cleaned_text)
                    cleaned_text = group_broken_paragraphs(cleaned_text)

                    docs.append({
                        "text": cleaned_text,
                        "es_id": doc["_id"],
                        "title": doc["_source"].get("title", ""),
                        "location": doc["_source"].get("location", ""),
                        "location_name": doc["_source"].get("location_name", ""),
                        "modified": doc["_source"].get("modified", ""),
                        "published": doc["_source"].get("published", ""),
                        "source": doc["_source"].get("source", ""),
                        "type": doc["_source"].get("type", ""),
                        "identifier": doc["_source"].get("identifier", ""),
                        "url": doc["_source"].get("url", ""),
                        "chunk_index": i,
                        "chunk_count": chunk_count
                    })
                else:
                    # Skip elements without text
                    continue
            except Exception as e:
                tqdm.write(f"Exception processing element in doc {doc.get('_id')}: {e}", file=sys.stderr)
                continue

        return docs if docs else None
    except Exception as e:
        # Log exception and return None
        tqdm.write(f"Exception processing doc {doc.get('_id')}: {e}", file=sys.stderr)
        return None

def main():
    # Connect to Elasticsearch
    es = Elasticsearch("http://localhost:9200")
    index_name = "jodal_documents7"
    parquet_file_3_gemeentes = '3_gemeentes.parquet'
    parquet_file_overijssel = 'overijssel.parquet'
    parquet_file_all = 'all.parquet'
    query_3_gemeentes = {
        "query": {
            "bool": {
                "must": [
                    {"exists": {"field": "description"}}
                ],
                "minimum_should_match": 1,
                "should": [
                    {
                        "match_phrase": {
                        "location": "GM0141"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM1896"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0180"
                        }
                    }
                ]
            }
        }
    }
    query_overijssel = {
        "query": {
            "bool": {
                "must": [
                    {"exists": {"field": "description"}}
                ],
                "minimum_should_match": 1,
                "should": [
                    {
                        "match_phrase": {
                        "location": "GM0193"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0141"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0166"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM1774"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0183"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0173"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0150"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM1700"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0164"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0153"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0148"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM1708"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0168"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0160"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0189"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0177"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM1742"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0180"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM1896"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0175"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM1735"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0147"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0163"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0158"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM1773"
                        }
                    }
                ]
            }
        }
    }

    query_all = {
        "query": {
            "bool": {
                "must": [
                    {"exists": {"field": "description"}}
                ]
            }
        }
    }

    query = query_all
    parquet_file = parquet_file_all

    # Use scan to retrieve documents
    es_scan = scan(client=es, index=index_name, query=query)

    # Get total document count for progress bar
    total_docs = es.count(index=index_name, body=query)['count']

    max_workers = multiprocessing.cpu_count()
    pool = Pool(processes=max_workers)
    processed_docs_iter = pool.imap_unordered(process_document, es_scan, chunksize=100)

    processed_docs = []
    batch_size = 1000  # Adjust based on your memory capacity
    batch_num = 0

    # Use a unique directory for batch Parquet files
    output_dir = 'output_parquet_batches'
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with tqdm(total=total_docs) as pbar:
        for docs_list in processed_docs_iter:
            # Update the progress bar for each Elasticsearch document processed
            pbar.update(1)
            if docs_list:
                processed_docs.extend(docs_list)
                if len(processed_docs) >= batch_size:
                    df = pd.DataFrame(processed_docs)
                    # Remove columns that are problematic for Parquet
                    for col in df.columns:
                        if df[col].apply(lambda x: isinstance(x, dict) and not x).all():
                            df.drop(columns=[col], inplace=True)
                    # Write each batch to a separate Parquet file
                    batch_file = os.path.join(output_dir, f'output_part{batch_num}.parquet')
                    df.to_parquet(batch_file, engine='pyarrow', index=False)
                    processed_docs = []
                    batch_num += 1
    # Write any remaining documents
    if processed_docs:
        df = pd.DataFrame(processed_docs)
        # Remove columns that are problematic for Parquet
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, dict) and not x).all():
                df.drop(columns=[col], inplace=True)
        batch_file = os.path.join(output_dir, f'output_part{batch_num}.parquet')
        df.to_parquet(batch_file, engine='pyarrow', index=False)
        processed_docs = []

    pool.close()
    pool.join()

    # Combine all the Parquet files into a single dataset
    import glob
    all_files = glob.glob(os.path.join(output_dir, 'output_part*.parquet'))
    tables = [pq.read_table(f) for f in all_files]
    # Update the following line to address the FutureWarning
    combined_table = pa.concat_tables(tables, promote_options='default')
    pq.write_table(combined_table, parquet_file)
    
    # Optionally, clean up batch files
    for f in all_files:
        os.remove(f)
    os.rmdir(output_dir)

if __name__ == "__main__":
    main()
