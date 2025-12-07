import os
import argparse
import json
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tqdm import tqdm
import time

from routerag.RouteRAG import RouteRAG

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Launch the RouteRAG FastAPI retrieval server.")
parser.add_argument("--save_dir", type=str, default="outputs", help="Directory where RouteRAG objects are stored.")
parser.add_argument('--dataset', type=str, default='hotpotqa_10k', help='Dataset name')
parser.add_argument("--llm_model_name", type=str, default="/models/Llama-3.1-8B-Instruct", help="The name of the LLM model to use.")
parser.add_argument('--llm_base_url', type=str, default='http://localhost:6000/v1', help='LLM base URL')
parser.add_argument("--embedding_model_name", type=str, default="/models/contriever", help="The name of the embedding model to use.")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for the API server.")
parser.add_argument("--port", type=int, default=8001, help="Port for the API server.")
parser.add_argument("--corpus_path", type=str, required=True, help="Path to a JSON corpus file for indexing.")
parser.add_argument("--force_reindex", action="store_true", help="Force re-indexing even if an index already exists.")
args = parser.parse_args()

# --- Globals ---
routerag: Optional[RouteRAG] = None

# --- FastAPI App ---
app = FastAPI(
    title="RouteRAG Retrieval Service",
    description="A FastAPI server to handle retrieval using the RouteRAG framework.",
    version="1.0.0"
)

# --- Pydantic Models ---
class SearchRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False
    return_times: bool = False

class SearchResponse(BaseModel):
    result: List
    time: List

# --- FastAPI Events ---
@app.on_event("startup")
async def startup_event():
    """
    Initializes the RouteRAG instance when the server starts.
    If no index is found, it will automatically index a default corpus.
    """
    global routerag
    if '/' not in args.save_dir:
        args.save_dir = args.save_dir + '/' + args.dataset

    print("--- Server Configuration ---")
    print(f"Save Directory: {args.save_dir}")
    print(f"Extractor Model: {args.llm_model_name}")
    print(f"Embedding Model: {args.embedding_model_name}")
    print(f"Corpus Path: {args.corpus_path}")
    print(f"Force Re-index: {args.force_reindex}")
    print("--------------------------")

    try:
        print("Initializing RouteRAG...")
        routerag = RouteRAG(
            save_dir=args.save_dir,
            llm_model_name=args.llm_model_name,
            llm_base_url=args.llm_base_url,
            embedding_model_name=args.embedding_model_name
        )
        
        graph_pickle_file = routerag._graph_pickle_filename
        if args.force_reindex or not os.path.exists(graph_pickle_file):
            print(f"Index not found at {graph_pickle_file} or re-indexing is forced.")
            print("Starting indexing process...")
            
            print(f"Loading corpus from {args.corpus_path}...")
            try:
                with open(args.corpus_path, 'r', encoding='utf-8') as f:
                    corpus = json.load(f)
                if not isinstance(corpus, list) or not all(isinstance(item, dict) for item in corpus):
                    raise ValueError("Corpus file must contain a JSON list of dict.")
                docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
            except Exception as e:
                raise RuntimeError(f"Failed to load or parse corpus file: {e}") from e
            
            if docs:
                routerag.index(docs=docs)
                print("Indexing complete.")
            else:
                print("Warning: Corpus file is empty. No documents found to index.")

        print("Preparing retrieval objects...")
        routerag.prepare_retrieval_objects()
        print("RouteRAG is ready to retrieve.")

    except Exception as e:
        print(f"FATAL: Failed to initialize RouteRAG. Error: {e}")
        raise RuntimeError("Could not initialize RouteRAG model.") from e

# --- API Endpoints ---
@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Batch retrieval endpoint that supports multiple queries and returns retrieval results for each query (with optional scores).
    - **queries**: List of queries
    - **topk**: Number of documents to return for each query
    - **return_scores**: Whether to return scores
    """
    if not routerag:
        raise HTTPException(status_code=503, detail="RouteRAG model is not initialized. The server might be starting up or encountered an error.")

    try:
        topk = request.topk if request.topk is not None else routerag.global_config.retrieval_top_k
        resp, search_times = [], []
        
        for query in tqdm(request.queries, desc="Retrieving"):
            if request.return_scores:
                start_time = time.time()
                docs, scores = routerag.search(query=query, num_to_retrieve=topk, return_scores=True)
                end_time = time.time()
                elapsed_time = end_time - start_time
                combined = []
                for doc, score in zip(docs, scores):
                    combined.append({"document": doc, "score": float(score)})
                resp.append(combined)
                if request.return_times:
                    search_times.append(elapsed_time)
            else:
                start_time = time.time()
                docs = routerag.search(query=query, num_to_retrieve=topk, return_scores=False)
                end_time = time.time()
                elapsed_time = end_time - start_time
                resp.append(docs)
                if request.return_times:
                    search_times.append(elapsed_time)
                
        return SearchResponse(result=resp, time=search_times)
    except Exception as e:
        print(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during the retrieval process.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the RouteRAG Retrieval API."}

# --- Main Block ---
if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port) 
