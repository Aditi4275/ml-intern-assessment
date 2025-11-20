# Trigram Language Model

This project implements a Trigram Language Model that learns word sequences from a text corpus and generates new text based on trigram probabilities. It stores unigram, bigram, and trigram counts with smoothing and backoff for improved text generation.

## How to Run 
**1. Initialize the uv environment**
```bash
uv init 
```
**2. Sync dependencies and create a virtual environment**
```bash
uv sync
```
**3. Run scripts**
```bash
uv run src/generate.py
```
**4. Run tests**
```bash
uv run pytest tests/test_ngram.py
```

# Scaled Dot-Product Attention   
This project implements the core scaled dot-product attention mechanism—the fundamental building block of Transformers using only numpy for numerical computing. The function computes attention scores, applies scaling, masking, softmax, and produces the attended output and attention weights.

## How to Run 

**Run scripts**
```bash
uv run task2/demo.py
```