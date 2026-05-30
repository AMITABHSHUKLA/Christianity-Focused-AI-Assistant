import os
import requests
import numpy as np
import faiss
from openai import OpenAI
from rank_bm25 import BM25Okapi
from duckduckgo_search import DDGS
from dotenv import load_dotenv

# loading env variables securely so I don't leak keys on github
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_and_index_bible():
    url = "https://raw.githubusercontent.com/thiagobodruk/bible/master/json/en_kjv.json"
    bible_data = requests.get(url).json()

    demo_chunks = []
    for book in bible_data:
        # keeping it to the Book of John for the demo to bypass memory/time limits
        if book['name'] != 'John': continue
        for chapter_idx, chapter_verses in enumerate(book['chapters']):
            # sliding window overlapping chunk strategy 
            for i in range(0, len(chapter_verses), 2):
                verse_block = chapter_verses[i : i + 3]
                if not verse_block: break
                demo_chunks.append({
                    "text": " ".join(verse_block),
                    "metadata": {"chunk_id": f"John_{chapter_idx+1}_{i+1}-{i+len(verse_block)}"}
                })

    demo_texts = [c['text'] for c in demo_chunks]
    
    # 1536 dims for text-embedding-3-small
    embeddings = [d.embedding for d in client.embeddings.create(input=demo_texts, model="text-embedding-3-small").data]
    vector_index = faiss.IndexFlatL2(len(embeddings[0]))
    vector_index.add(np.array(embeddings).astype('float32'))
    
    bm25_index = BM25Okapi([t.lower().split() for t in demo_texts])
    
    return demo_chunks, vector_index, bm25_index

def hybrid_search(query, demo_chunks, vector_index, bm25_index, top_k=5):
    bm25_top = np.argsort(bm25_index.get_scores(query.lower().split()))[::-1][:top_k]
    query_vec = np.array([client.embeddings.create(input=[query], model="text-embedding-3-small").data[0].embedding]).astype('float32')
    _, faiss_top = vector_index.search(query_vec, top_k)
    # merged into a set to scrub duplicate hits
    return [demo_chunks[i] for i in set(bm25_top.tolist() + faiss_top[0].tolist())]

def evaluate_retrieval(query, chunks):
    # my circuit breaker LLM judge to kill hallucinations
    context = "\n".join([c['text'] for c in chunks])
    prompt = f"Does this context answer the query fully? Reply YES or NO.\nQuery:{query}\nContext:{context}"
    res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0)
    return "YES" if "YES" in res.choices[0].message.content.upper() else "NO"

def fact_check(query):
    # dynamic fallback if the circuit breaker trips
    try:
        with DDGS() as ddgs:
            results = [r['body'] for r in ddgs.text(f"{query} Christianity Bible fact check fake", max_results=2)]
        return "\n".join(results) if results else "No external facts found."
    except: 
        return "Fact check unavailable."

def check_moderation(prompt):
    return client.moderations.create(input=prompt).results[0].flagged

def process_image_request(prompt):
    # dual pipeline: rewrite for safety, then pass to actual DALL-E
    refiner = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Refine the user's image prompt to be highly reverent and Biblical. If the request is offensive, violent, or heretical, reply ONLY with the word: REJECTED."},
                  {"role": "user", "content": prompt}]
    ).choices[0].message.content
    
    if "REJECTED" in refiner:
        return None, "Image request rejected. It violates our safety and reverence policies."
    
    try:
        # switched back to active API instead of local placeholder
        img_response = client.images.generate(model="dall-e-2", prompt=refiner, n=1)
        return img_response.data[0].url, f"Generated via Refined Prompt: {refiner[:75]}..."
    except Exception as e:
        return None, f"Image API Error: {e}"

def process_text_request(prompt, chat_memory, demo_chunks, vector_index, bm25_index):
    chunks = hybrid_search(prompt, demo_chunks, vector_index, bm25_index)
        
    if evaluate_retrieval(prompt, chunks) == "YES":
        context = "BIBLICAL CONTEXT:\n" + "\n".join([f"[{c['metadata']['chunk_id']}] {c['text']}" for c in chunks])
    else:
        context = f"EXTERNAL FACT-CHECK:\n{fact_check(prompt)}"
    
    sys_prompt = "You are a respectful Christian AI assistant. Answer ONLY using the provided context. If citing the Bible, use the Book, Chapter, and Verse. Be denominationally neutral on debated topics. Refuse malicious or extreme prompts gracefully."
    
    api_messages = [{"role": "system", "content": sys_prompt}]
    # prune history to last 4 messages to prevent context dilution
    api_messages.extend([{"role": m["role"], "content": m["content"]} for m in chat_memory[-4:]])
    api_messages.append({"role": "user", "content": f"CONTEXT:\n{context}\n\nQUERY:{prompt}"})
    
    response = client.chat.completions.create(model="gpt-4o", messages=api_messages, temperature=0.3)
    return response.choices[0].message.content