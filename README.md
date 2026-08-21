# AI Agent z Własnym Modelem, Web Search i Reasoning

## 🎯 Opis Projektu

Zaawansowany agent AI wyposażony w:
- **Własny model AI** - sieć neuronowa do analizy intencji
- **Web Search** - wyszukiwanie informacji w internecie (DuckDuckGo)
- **Model Reasoningowy** - wieloetapowe wnioskowanie logiczne

## 🚀 Funkcjonalności

### 1. Model AI (`src/model.py`)
- Sieć neuronowa z warstwą ukrytą (100 → 64 → 10)
- Analiza intencji zapytań użytkownika
- Trening od podstaw z funkcjami aktywacji ReLU i Softmax
- Zapis/odczyt modelu do pliku

### 2. Wyszukiwanie Internetowe (`src/web_search.py`)
- Integracja z DuckDuckGo API
- Ekstrakcja treści ze stron WWW
- Weryfikacja informacji z wielu źródeł
- Parsowanie i czyszczenie danych HTML

### 3. Model Reasoningowy (`src/reasoning_model.py`)
- Wielostopniowe wnioskowanie:
  - 🔍 Analiza problemu
  - ➡️ Dedukcja logiczna
  - 💡 Generowanie hipotez (abdukcja)
  - 🧩 Synteza wniosków
  - ✅ Weryfikacja poprawności
- Baza wiedzy rozszerzalna
- Łańcuchy rozumowania z poziomami pewności

### 4. Główny Agent (`src/agent.py`)
- Łączy wszystkie komponenty w spójną całość
- Przetwarzanie zapytań z użyciem:
  - Analizy intencji (AI Model)
  - Wyszukiwania online (Web Search)
  - Głębokiego rozumowania (Reasoning Model)
- Historia konwersacji i statystyki
- Konfigurowalne parametry działania

## 📁 Struktura Projektu

```
/workspace
├── README.md              # Dokumentacja projektu
├── requirements.txt       # Zależności Python
├── src/
│   ├── __init__.py       # Inicjalizacja pakietu
│   ├── model.py          # Model AI (sieć neuronowa)
│   ├── web_search.py     # Moduł wyszukiwania internetowego
│   ├── reasoning_model.py # Model rozumowania logicznego
│   └── agent.py          # Główny agent AI
├── data/                  # Dane treningowe
└── models/                # Zapisane modele
```

## 🛠️ Instalacja

```bash
# Instalacja zależności
pip install -r requirements.txt

# Uruchomienie agenta
python -m src.agent
```

## 📦 Zależności

- `numpy>=1.24.0` - Obliczenia numeryczne
- `pandas>=2.0.0` - Przetwarzanie danych
- `scikit-learn>=1.3.0` - Narzędzia ML
- `requests>=2.31.0` - Żądania HTTP
- `beautifulsoup4>=4.12.0` - Parsowanie HTML
- `duckduckgo-search>=3.9.0` - API wyszukiwania DuckDuckGo

## 💡 Przykłady Użycia

```python
from src.agent import AIAgent

# Inicjalizacja
agent = AIAgent()

# Proste zapytanie
response = agent.chat("Jak AI zmienia świat w 2024?")
print(response)

# Statystyki
stats = agent.get_stats()
print(f"Interakcje: {stats['total_interactions']}")
print(f"Średnia pewność: {stats['avg_confidence']:.1%}")
```

## 🎯 Architektura

```
Zapytanie → Model AI → Web Search → Reasoning → Synteza → Odpowiedź
```

## 🧪 Testowanie

```bash
python -m src.agent
```

## 📝 Licencja

MIT License
