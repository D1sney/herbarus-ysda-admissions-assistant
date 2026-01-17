# Парсинг Telegram-канала для RAG-базы знаний

Подготовка базы знаний в формате CSV для дальнейшего использования в RAG (Retrieval-Augmented Generation) 

## Воспроизведение результата

1. **Экспортируйте историю канала**:
   - В Telegram Desktop: `Канал` → `Ещё` → `Экспорт истории чата`.
   - Выберите формат `JSON`, без вложений.

2. **Поместите файл экспорта**:
   - Переименуйте файл экспорта в `result.json` и положите в папку `src/ingest/telegram/`.

3. **Запустите скрипт**:
   ```bash
   cd src/ingest/telegram
   python parse_telegram_export.py
   ```
4. **Результат**:
   - Файл messages_parsed.csv будет создан в той же папке.
   - Содержит столбцы: text, date, link.