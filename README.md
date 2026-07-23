# LangChain Translation API

A simple translation API built with LangChain, Groq, FastAPI, and LangServe.

## Features

- Translates text into a selected language
- Uses a Groq-hosted language model
- Provides a REST API with FastAPI
- Includes an interactive LangServe playground

## Requirements

- Python 3.12 or later
- A Groq API key
- A LangChain API key if LangSmith tracing is enabled

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/langchain-translation-api.git
cd langchain-translation-api
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
```

Never commit your real `.env` file or API keys to GitHub.

## Running the application

```bash
python simplemessage.py
```

After the server starts, open:

- API documentation: http://127.0.0.1:8000/docs
- LangServe playground: http://127.0.0.1:8000/chain/playground/

## Technologies

- Python
- LangChain
- Groq
- FastAPI
- LangServe
