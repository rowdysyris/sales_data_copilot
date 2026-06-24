# OpenAI API Setup

The app works without an API key. OpenAI is optional and is used only to rewrite deterministic Pandas answers into cleaner manager-ready wording.

## Local `.env` setup

1. Copy `.env.example` to `.env`.
2. Put this inside `.env`:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
```

3. Restart Streamlit:

```powershell
streamlit run app.py
```

## Windows PowerShell alternative

```powershell
setx OPENAI_API_KEY "sk-your-key-here"
setx LLM_PROVIDER "openai"
setx OPENAI_MODEL "gpt-4o-mini"
```

Close and reopen VS Code/terminal after `setx`.

## Safety rules

- Never paste your API key into `app.py` or any Python file.
- Never commit `.env` to GitHub.
- The app only sends already-computed summaries to OpenAI, not the full raw dataset.
- If the OpenAI call fails, the app falls back to deterministic answers.
