import pandas as pd
import json

INPUT_CSV = "messages_with_full_llm_response.csv"
OUTPUT_JSONL = "tg_knowledge_base.jsonl"

df = pd.read_csv(INPUT_CSV)

df_filtered = df[df['llm_relevance_response'] > 7]

with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
    for _, row in df_filtered.iterrows():
        record = {
            "text": row['text'],
            "metadata": {
                "link": row['link'].strip() if pd.notna(row['link']) else "",
                "date": row['date'] if pd.notna(row['date']) else ""
            }
        }
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(f"✅ Saved {len(df_filtered)} records to {OUTPUT_JSONL}")