import schedule
import time
import asyncio
import edge_tts
import pygame
import speech_recognition as sr
import ollama
import chromadb
import os
from dotenv import load_dotenv
from newsapi import NewsApiClient
from datetime import datetime
from duckduckgo_search import DDGS

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API")  
DB_PATH = "./my_news_db"           
WAKE_UP_TIME = "08:00"             

if not NEWS_API_KEY:
    raise ValueError(
        "NEWS_API environment variable is not set. "
        "Please create a .env file with NEWS_API=your_api_key_here "
        "or set it as an environment variable. "
        "Get your API key from https://newsapi.org/"
    )

newsapi = NewsApiClient(api_key=NEWS_API_KEY)
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name="daily_news")

async def speak(text):
    print(f"Assistant: {text}")
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    await communicate.save("response.mp3")

    pygame.mixer.init()
    pygame.mixer.music.load("response.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()

def fetch_and_store_news():
    print(f"Fetching news for {datetime.now().date()}...")
    
    top_headlines = newsapi.get_top_headlines(language='en', country='us', page_size=10)
    if not top_headlines['articles']:
        return "I couldn't find any news right now."

    documents = []
    ids = []
    metadatas = []
    summary_text = "Here are today's top stories: "
    for i, article in enumerate(top_headlines['articles']):
        content = f"Title: {article['title']}. Description: {article['description']}. Content: {article['content']}"
        
        documents.append(content)
        ids.append(f"news_{datetime.now().strftime('%Y%m%d')}_{i}")
        metadatas.append({"source": article['source']['name'], "date": str(datetime.now().date())})
        
        if i < 3: 
            summary_text += f"{i+1}. {article['title']}. "

    collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
    print("News saved to vector database.")

    return summary_text + " Would you like to know more about any of these?"

def web_search(query):
    today = datetime.now().strftime("%B %d, %Y")
    search_query = f"{query} news {today}"
    print(f"Searching web for: {search_query}")
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=5))
            if results:
                context = ""
                for r in results:
                    context += f"Title: {r['title']}. Body: {r['body']}. Source: {r['source']}. "
                return context
            
            results = list(ddgs.text(search_query, max_results=5))
            if results:
                context = ""
                for r in results:
                    context += f"Title: {r['title']}. Content: {r['body']}. "
                return context
    except Exception as e:
        print(f"Web search error: {e}")
    
    return None

def query_rag(question):
    results = collection.query(query_texts=[question], n_results=3)
    context = " ".join(results['documents'][0]) if results['documents'][0] else ""
    distances = results.get('distances', [[]])[0]
    
    rag_is_relevant = len(distances) > 0 and min(distances) < 1.5 and context.strip()
    
    if rag_is_relevant:
        check_prompt = f"""Based on this news context, can you answer the question? 
If the context contains relevant information, answer it. 
If the context is NOT relevant or doesn't contain information to answer the question, respond ONLY with: "NEED_WEB_SEARCH"

Context: {context}

Question: {question}"""
        
        check_response = ollama.chat(model='qwen2:7b', messages=[{'role': 'user', 'content': check_prompt}])
        answer = check_response['message']['content']
        
        if "NEED_WEB_SEARCH" not in answer:
            return answer
    
    print("RAG context insufficient, searching the web...")
    web_context = web_search(question)
    
    if web_context:
        today = datetime.now().strftime("%B %d, %Y")
        prompt = f"""Using the following web search results from {today}, answer the user's question clearly and concisely.

Web Search Results: {web_context}

Question: {question}"""
        response = ollama.chat(model='gemma2:2b', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']
    
    if context.strip():
        prompt = f"Using the following news context, answer the user's question as best as you can. If you don't know, say so.\n\nContext: {context}\n\nQuestion: {question}"
        response = ollama.chat(model='qwen2:7b', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']
    
    return "I couldn't find relevant information about that topic. Could you try rephrasing your question?"

def listen_for_questions():
    r = sr.Recognizer()
    mic = sr.Microphone()
    
    while True:
        with mic as source:
            print("Listening for questions... (Say 'stop' to end)")
            r.adjust_for_ambient_noise(source)
            try:
                audio = r.listen(source, timeout=5, phrase_time_limit=10)
                text = r.recognize_google(audio).lower()
                print(f"You: {text}")

                if "stop" in text or "that is all" in text:
                    asyncio.run(speak("Alright, have a great day."))
                    break
                
                answer = query_rag(text)
                asyncio.run(speak(answer))
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(e)

def morning_routine():
    summary = fetch_and_store_news()
    asyncio.run(speak(f"Good morning Raghav. {summary}"))
    listen_for_questions()

if __name__ == "__main__":
    print(f"System armed. Waiting for {WAKE_UP_TIME}...")
    morning_routine()
    schedule.every().day.at(WAKE_UP_TIME).do(morning_routine)

    while True:
        schedule.run_pending()
        time.sleep(60)
