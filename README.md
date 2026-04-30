# AI Research Copilot

AI Research Copilot is an early-stage Python project for building a Retrieval-Augmented Generation (RAG) application from notebook experiments into a maintainable product.

This repository is currently centered around the prototype in `notebooks/copilot_v2.ipynb`. The goal is to evolve that notebook code into a clean application architecture that is scalable, easy to extend, and aligned with DRY and SOLID principles.

## Suggested Repository Description

Notebook-driven starter project for building a scalable, maintainable RAG research copilot in Python.

## Project Status

This project is still in its initial setup phase.

What exists today:

- notebook-based experimentation in `notebooks/copilot_v1.ipynb` and `notebooks/copilot_v2.ipynb`
- supporting markdown source material in `notebooks/`
- basic project packaging with `pyproject.toml`
- a minimal `main.py` entrypoint placeholder

What the project is aiming for:

- document ingestion
- chunking and preprocessing
- retrieval over indexed knowledge
- answer generation with an LLM
- modular architecture for future vector store and model changes
- production-friendly structure for testing, maintenance, and scaling

## Repository Layout

```text
.
├── main.py
├── pyproject.toml
├── README.md
└── notebooks/
    ├── copilot_v1.ipynb
    ├── copilot_v2.ipynb
    ├── mastering_gradient_descent:_a_comprehensive_guide_for_beginners_and_beyond.md
    ├── understanding_self-attention:_the_heart_of_modern_deep_learning.md
    └── Mastering Gradient Descent A Practical Guide for Developers
```

## Purpose

The main purpose of this repository is to take notebook logic and turn it into a real RAG application with better separation of concerns.

The long-term architecture should separate:

- data loading
- chunking
- embedding and indexing
- retrieval
- answer generation
- orchestration
- configuration
- testing

This keeps the system easier to debug, easier to swap components in and out, and safer to scale over time.

## Current Prototype

The prototype in `notebooks/copilot_v2.ipynb` is the main reference implementation right now.

It appears to focus on:

- graph-based orchestration
- LLM-driven content generation
- structured output patterns
- early experimentation with LangGraph and Mistral-based workflows

That notebook is a good exploration artifact, but notebook code alone is not enough for a scalable application. A production-ready version should move that logic into reusable Python modules and testable services.

## Getting Started

### Requirements

- Python 3.11 or newer

### Install dependencies

If you are using `uv`:

```bash
uv sync
```

If you are using `pip`:

```bash
pip install -e .
```

### Run the current entrypoint

```bash
python3 main.py
```

At the moment, `main.py` is only a placeholder and does not yet run a full RAG workflow.

## Recommended Next Steps

To turn this into a real RAG application, the next implementation steps should be:

1. Create a proper package structure under `src/`.
2. Move notebook logic into Python modules.
3. Add configuration management for models, API keys, and paths.
4. Build document loaders for `.md`, `.txt`, `.pdf`, and `.ipynb`.
5. Add chunking and metadata preservation.
6. Add embeddings and a retriever or vector database.
7. Implement answer generation using retrieved context.
8. Add tests for chunking, retrieval, and orchestration behavior.
9. Add logging, error handling, and output persistence.

## Design Principles

This project should be built with these principles in mind:

- DRY: avoid duplicated retrieval, prompting, and preprocessing logic
- SOLID: keep responsibilities small and dependencies replaceable
- maintainability: organize code by domain responsibility, not by experiment
- scalability: make it easy to swap in better retrievers, vector stores, and models
- observability: keep the system easy to trace and debug as workflows grow

## Notes

- The notebooks are valuable source material and should be treated as prototypes, not final architecture.
- The markdown files in `notebooks/` can become the initial knowledge base for testing retrieval.
- The current repository metadata has been updated, but the implementation itself still needs the actual RAG modules to be added.

## License

This project is licensed under the terms of the `LICENSE` file in this repository.
