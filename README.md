# Agent Analityka Finansowego (LLM Project)

Projekt jest inteligentnym agentem AI zdolnym do przeprowadzania analiz finansowych przy użyciu narzędzi (Ceny akcji, Analiza techniczna, RSI) oraz systemu RAG (Retrieval Augmented Generation) dostarczającego kontekst z wiadomości finansowych.

## Funkcjonalności
- **Tryb API i Lokalny**: Obsługa OpenAI (GPT-3.5/4) oraz Google Gemini.
- **Function Calling**: Zaawansowany dispatcher narzędzi ze ścisłą walidacją argumentów (Pydantic) i typowaniem.
- **System Mini-RAG**: Wyszukiwanie semantyczne oparte na embeddingach (`sentence-transformers` + `faiss`) z rerankingiem (Cross-Encoder) dla zwiększenia precyzji.
- **Guardrails (Bezpieczeństwo)**:
  - Ochrona przed Prompt Injection (heurystyki).
  - Walidacja wyjścia (blokada słów zakazanych).
  - Sanitacja ścieżek (ochrona przed Path Traversal).
- **REST API**: Endpoint `/ask` w technologii FastAPI.
- **Ewaluacja**: Zautomatyzowany zestaw testów z raportem skuteczności.

## Instrukcja Uruchomienia

### 1. Instalacja zależności
Upewnij się, że masz zainstalowanego Pythona (3.9+).
```bash
pip install -r requirements.txt
```

### 2. Konfiguracja środowiska
Skopiuj plik `.env.template` do `.env` i uzupełnij swój klucz API:
```bash
# Windows (PowerShell)
copy .env.template .env
```
Edytuj plik `.env` wpisując klucz `OPENAI_API_KEY` (działa też z kluczami Google `AIza...`).

### 3. Uruchomienie API
Serwer wystartuje pod adresem `http://localhost:8000`.
```bash
uvicorn app.main:app --reload
```

### 4. Uruchomienie testów i ewaluacji
Aby wygenerować raport z testów (`evaluation_report.md`):
```bash
python evaluate.py
```

## Użycie API

**Endpoint**: `POST /ask`

**Przykładowe żądanie (JSON)**:
```json
{
  "query": "Jaka jest cena akcji AAPL i ich prognoza na przyszły tydzień?"
}
```

**Przykładowa odpowiedź**:
```json
{
  "answer": "Cena akcji AAPL wynosi 145.20 USD. Biorąc pod uwagę RSI na poziomie 35, sugeruje to trend wzrostowy...",
  "tool_calls": [
    {
       "tool": "get_stock_price",
       "args": { "ticker": "AAPL" }
    }
  ]
}
```

## Architektura Systemu

System składa się z 4 głównych modułów, połączonych w następujący sposób:

```mermaid
graph TD
    User[Użytkownik / Klient] -->|POST /ask| API[FastAPI Endpoint]
    API --> Agent[Agent Główny]
    
    subgraph "Bezpieczeństwo"
        Agent -->|1. Check| Guardrails[Guardrails & Walidacja]
    end
    
    subgraph "Retrieval Augmented Generation"
        Agent -->|2. Retrieve| RAG[RAG Engine]
        RAG -->|Szukaj| DB[(Baza Wiedzy news.txt)]
        RAG -->|Rerank| CrossEncoder[Reranker Model]
    end
    
    subgraph "Function Calling"
        Agent -->|3. LLM Call| LLM[LLM (OpenAI/Gemini)]
        LLM -->|Tool Call?| Dispatcher[Tool Registry]
        Dispatcher -->|Execute| Tools[Narzędzia (Cena, RSI)]
        Tools -->|Wynik| LLM
    end
    
    LLM -->|Final Response| Agent
    Agent -->|JSON| API
    API --> User
```

1.  **Dispatcher (`app/registry.py`)**: Centralny rejestr z walidacją Pydantic.
2.  **Agent (`app/agent.py`)**: Logika sterująca, RAG i komunikacja z LLM.
3.  **RAG Engine (`app/rag.py`)**: Wyszukiwanie hybrydowe (Embeddingi + Reranking).
4.  **Bezpieczeństwo (`app/guardrails.py`)**: Filtrowanie wejścia i wyjścia.

## Demo i Certyfikaty

**Certyfikat IBM**:
![Certyfikat IBM](ibm_certificate.png)
*(Umieść plik `ibm_certificate.png` w folderze głównym)*

**Demo - Przykład użycia**:
![Demo Screen](demo_screen.png)
*(Umieść zrzut ekranu np. z Postmana jako `demo_screen.png`)*


## Struktura Plików
*   `app/` - Kod źródłowy aplikacji (agent, narzędzia, API).
*   `tests/` - Testy jednostkowe i integracyjne.
*   `data/` - Pliki danych dla RAG.
*   `evaluate.py` - Skrypt uruchamiający ewaluację.
*   `requirements.txt` - Lista bibliotek.

## Autor
Projekt wykonany w ramach zaliczenia przedmiotu LLM.
