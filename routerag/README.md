# RouteRAG

RouteRAG is a sophisticated multi-turn hybrid Retrieval-Augmented Generation (RAG) framework built upon the [HippoRAG 2](https://github.com/OSU-NLP-Group/HippoRAG) foundation. It introduces several key architectural improvements and enhancements to enable more efficient and flexible information retrieval and question answering.

## Key Modifications

### 1. Multi-Turn Hybrid Retrieval

RouteRAG implements an advanced multi-turn hybrid retrieval system that combines:

- **Dense Retrieval**: Traditional semantic search using dense embeddings
- **Graph-based Retrieval**: Leverages structured knowledge graphs for more precise information retrieval
- **Reciprocal Rank Fusion**: Intelligently merges results from different retrieval methods
- **Iterative Refinement**: Supports multiple rounds of retrieval to progressively refine and expand the search space

The hybrid approach is controlled through special tokens in queries:

- `[passage]`: Triggers dense retrieval from document corpus
- `[graph]`: Activates graph-based retrieval from knowledge structures
- Both tokens can be used together for comprehensive hybrid retrieval

### 2. Separation of LLM Models: IE Model and Reader Model

RouteRAG introduces a clear architectural separation between two distinct LLM roles:

- **IE Model (Information Extraction Model)**: Handles the extraction of structured information from documents, including Named Entity Recognition (NER) and triple extraction for knowledge graph construction
- **Reader Model**: Dedicated to reading and comprehending retrieved documents to generate accurate answers

This separation allows for:

- **Specialized optimization**: Each model can be fine-tuned for its specific task
- **Independent scaling**: Different models can be deployed on different hardware configurations
- **Flexible deployment**: IE and reader models can be hosted separately or use different model architectures

### 3. vLLM Support for Reader Model

RouteRAG integrates vLLM (vLLM) support for the reader model, providing:

- **High-performance inference**: Leverages vLLM's optimized inference engine for faster text generation
- **Memory efficiency**: Better GPU memory utilization through vLLM's advanced memory management
- **Scalability**: Support for distributed inference across multiple GPUs

The framework automatically detects vLLM availability and falls back to standard LLM implementations when vLLM is not available.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐     ┌───────────────────┐
│   Documents     │───▶│  OpenIE Model   │───▶│  Knowledge Graph │
│   (Corpus)      │    │   (Information   │     │    Construction   │
└─────────────────┘    │   Extraction)    │     └───────────────────┘
         |             └──────────────────┘             |
         |______________________________________________|
                                |
                                ▼
┌─────────────────┐    ┌────────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Hybrid Retrieval │───▶│  Reader Model │
│                 │    │  (Multi-Turn)      │    │ (vLLM Support)  │
└─────────────────┘    └────────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Retrieved Docs   │    │ Final Answer    │
                       └──────────────────┘    └─────────────────┘
```
