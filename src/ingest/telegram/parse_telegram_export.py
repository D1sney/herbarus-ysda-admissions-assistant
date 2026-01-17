import json
import pandas as pd

INPUT_JSON = 'result.json'
OUTPUT_CSV = 'messages_parsed.csv'
CHANNEL_USERNAME = '@vse_v_shad'

def extract_text(text_field):
    """
    Извлекает текст из поля text, которое может быть строкой или списком словарей.
    """
    if isinstance(text_field, str):
        return text_field
    elif isinstance(text_field, list):
        result = []
        for item in text_field:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                result.append(item.get('text', ''))
        return ''.join(result)
    else:
        return ''

def main():
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    messages = data.get('messages', [])
    rows = []

    for msg in messages:
        if msg.get('type') != 'message':
            continue

        text = extract_text(msg.get('text', ''))
        date = msg.get('date')
        msg_id = msg.get('id')
        
        link = f"https://t.me/{CHANNEL_USERNAME.lstrip('@')}/{msg_id}"

        rows.append({
            'text': text,
            'date': date,
            'link': link
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"Сохранено {len(rows)} сообщений в {OUTPUT_CSV}")

if __name__ == '__main__':
    main()