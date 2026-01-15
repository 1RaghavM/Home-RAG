from fastapi import FastAPI
from ollama import chat
import chromadb
import os
import logging
from newsapi import NewsApiClient
from datetime import datetime as dt
from duckduckgo_search import DDGS

SYSTEM_INSTRUCTIONS = """
You are a high and proficient news reporter. You will report daily news to the user and answer questions based on the users questions.
You will provide summaries, headlines of trending topics for the current day, and if the user requests, you will use tools like web search to dig deeper.
You will use the date() tool to know the current date. You must append the date to the query for any web search.
You will use websearch if you do not have enough context or information for that specific article, or news source.
"""
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

MODEL_NAME = os.getenv("MODEL_NAME", "tinyllama")
logging.info(f"Using model: {MODEL_NAME}")

newsapi = NewsApiClient(api_key=os.getenv("NEWS_API"))

app = FastAPI()
chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection("docs")


def scrape_headlines(topic, date):
    top_headlines = newsapi.get_top_headlines(
        q=topic,
        sources='cnn.com,foxnews.com',
        language='en'
    )
    return top_headlines

def scrape_news_summ(topic, date):
        all_articles = newsapi.get_everything(
                q=topic,
                domains='foxnews.com,cnn.com',
                from_param=date,
                language='en',
                sort_by='relevancy'
        )
        return all_articles

def add(response: dict):
    try:
            import uuid
            doc_id = str(uuid.uuid4())
            logging.info(f"Received new news to add.")

            collection.add(documents=response, ids=[doc_id])

            return {
            "status":"success",
            "message": "Content added to knowledge base",
            "id": doc_id
        }
    except Exception as e:
        return {
            "status":"error",
            "message":str(e)
        }

def web_search(query, max_results=10):
    results = []
    with DDGS() as ddgs:
        for result in ddgs.text(query, region='wt-wt', safesearch='off', max_results=max_results):
            results.append(result)
    return results
    
def query(q):
    results = collection.query(query_texts=[q], n_results=2)
    context = results['documents'][0][0] if results['documents'] else ""

    logging.info(f"/query asked: {q}")

def date():
    x = dt.now()
    date = x.date()
    return date

answer = chat(
        model=MODEL_NAME,
        prompt=f"INSTRUCTIONS: {SYSTEM_INSTRUCTIONS}",
        tools=[web_search, date]
    )