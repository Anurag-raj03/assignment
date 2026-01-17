import re

def clean_results(results):
    cleaned = []

    for r in results:
        text = r["text"]

        # normalize
        text = text.replace("\n", " ").strip()
        text = re.sub(r"\s+", " ", text)

        # keep short but meaningful industrial tokens
        if len(text) >= 2:
            cleaned.append({
                "text": text,
                "bbox": r["bbox"]
            })

    return cleaned
