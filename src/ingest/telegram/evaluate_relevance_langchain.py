import pandas as pd
import os
import time
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from tqdm import tqdm

load_dotenv()

INPUT_CSV = "src/ingest/telegram/messages_parsed.csv"
OUTPUT_CSV = "src/ingest/telegram/messages_with_full_llm_response.csv"
CACHE_CSV = "src/ingest/telegram/relevance_cache.csv"

REQUEST_DELAY = 2

http_client = httpx.Client(timeout=30.0)

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0.00001,
    http_client=http_client
)

if os.path.exists(CACHE_CSV):
    cache_df = pd.read_csv(CACHE_CSV, encoding='utf-8')
    cache = dict(zip(cache_df['text'], cache_df['response']))
else:
    cache = {}

def save_cache():
    cache_df = pd.DataFrame(list(cache.items()), columns=['text', 'response'])
    cache_df.to_csv(CACHE_CSV, index=False, encoding='utf-8')

def get_full_llm_response(text):
    if text in cache:
        return cache[text]
    prompt = f"""
Evaluate ONLY the relevance of the following text for answering questions about the admissions process to YSDA (Yandex School of Data Analysis).

Consider whether the text contains explicit information about:
- Stages of admission (application, testing, exams, interviews, results),
- Specific dates or deadlines,
- Entry requirements or eligibility criteria,
- Contacts of curators, teachers, or support channels,
- Open house events (days of open doors) and how to attend them,
- Descriptions of admission tracks or sections (e.g., classical vs. alternative track),
- Official procedures, instructions, or links to application pages,
- Information about courses relevant to applicants.

Ignore accuracy, writing style, formatting, or unrelated content. Focus solely on topic relevance.

Return a single integer from 0 to 10, where 10 means the text is fully relevant and directly useful for answering applicant questions, and 0 means it is completely unrelated.  

Text:  
{text}
"""
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        result = response.content.strip()
        cache[text] = result
        save_cache()
        time.sleep(REQUEST_DELAY)
        return result
    except Exception as e:
        print("Stopping to avoid data corruption.")
        raise 

df = pd.read_csv(INPUT_CSV)

print("Running LLM evaluation...")
tqdm.pandas()
df['llm_relevance_response'] = df['text'].progress_apply(get_full_llm_response)

df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
print(f"Done! Full LLM responses saved to {OUTPUT_CSV}")
