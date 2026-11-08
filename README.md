# LangChain Translation API

A small translation API built with LangChain and the Groq LLM `llama-3.1-8b-instant`. It is served with FastAPI and LangServe.

## What it does

`simplemessage.py` builds a chain (prompt template, ChatGroq model, string output parser) with the system prompt "Translate the following into {language}". LangServe exposes it at `/chain` on port 8000.

## How to run

```
pip install -r requirements.txt
set GROQ_API_KEY=your_key_here      (Windows; use export on Linux/Mac)
python simplemessage.py
```

You need your own Groq API key. Set it as the `GROQ_API_KEY` environment variable (or in a local `.env` file, which is git-ignored). Do not commit it. The docs page is at http://localhost:8000/docs.

## Benchmark script

`src/benchmark.py` does not call the Groq API and does not need a key. It scores a few hard-coded example translations (6 sentences, English to Turkish/German/French/Spanish) with simple BLEU and chrF implementations, and the latencies in it are simulated numbers. So the table below is only a demo of the scoring code, not a measurement of the real API.

```
pip install numpy
python src/benchmark.py
```

| Pipeline | bleu_4 | chrf_6 | terminology_fidelity |
|---|---|---|---|
| Glossary prompt chain | 0.7952 | 0.9389 | 1.0 |
| Zero-shot chain | 0.1993 | 0.5782 | 0.35 |
| Lexical phrase baseline | 0.052 | 0.3796 | 0.075 |

## Files

- `simplemessage.py`: the API
- `requirements.txt`: dependencies
- `src/benchmark.py`: scoring demo
- `benchmarks/benchmark_results.json`: its output
- `assets/benchmark_chart.svg`: chart of the output
