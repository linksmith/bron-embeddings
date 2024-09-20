# index/utils.py

import sys
import re
from lxml import etree
import contextlib
import os

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Suppress stdout and stderr."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def format_time(seconds):
    """Format time in seconds to HH:MM:SS."""
    hours = int(seconds) // 3600
    minutes = (int(seconds) % 3600) // 60
    seconds = int(seconds) % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def remove_processing_instructions(html_text):
    """Remove processing instructions from HTML content using lxml."""
    try:
        with suppress_stdout_stderr():
            parser = etree.HTMLParser(remove_pis=True)
            tree = etree.fromstring(html_text.encode('utf-8'), parser)
            return etree.tostring(tree, encoding='unicode', method='html')
    except Exception:
        # If parsing fails, return the original text
        return html_text

def process_document(doc):
    from unstructured.partition.html import partition_html
    from unstructured.cleaners.core import (
        clean,
        clean_non_ascii_chars,
        replace_unicode_quotes,
        group_broken_paragraphs
    )
    error_count = 0  # Initialize error counter
    try:
        # Extract the 'description' field
        description = doc.get('_source', {}).get('description', '')
        try:
            description = remove_processing_instructions(description)
        except Exception:
            # If parsing fails, keep the original description
            error_count += 1

        # Process 'description' field using unstructured.io
        try:
            with suppress_stdout_stderr():
                elements = partition_html(
                    text=description,
                    chunking_strategy='by_title',
                    combine_text_under_n_chars=1024,
                    max_characters=4096,
                    new_after_n_chars=3072,
                    overlap=256
                )
        except Exception:
            # Error in partition_html
            error_count += 1
            return None, error_count  # Return None and error count

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

                    title = doc["_source"].get("title", "")
                    location_name = doc["_source"].get("location_name", "")
                    cleaned_text = f'Titel: {title} \n Gemeente: {location_name} \n Tekst: \n\n {cleaned_text}'

                    docs.append({
                        "text": cleaned_text,
                        "es_id": doc["_id"],
                        "title": title,
                        "location": doc["_source"].get("location", ""),
                        "location_name": location_name,
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
            except Exception:
                # Error processing element
                error_count += 1
                continue

        return (docs if docs else None), error_count
    except Exception:
        # Log exception and return None
        error_count += 1
        return None, error_count
