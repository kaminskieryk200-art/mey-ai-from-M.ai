"""
Moduł wyszukiwania w internecie dla agenta AI.
Obsługuje wyszukiwanie przez DuckDuckGo i parsowanie wyników.
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from duckduckgo_search import DDGS
import re


class WebSearchEngine:
    """
    Silnik wyszukiwania w internecie dla agenta AI.
    Pozwala na przeszukiwanie sieci i ekstrakcję informacji.
    """

    def __init__(self, max_results: int = 5, timeout: int = 10):
        """
        Inicjalizacja silnika wyszukiwania.

        Args:
            max_results: Maksymalna liczba wyników do zwrócenia
            timeout: Timeout dla żądań HTTP w sekundach
        """
        self.max_results = max_results
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def search(self, query: str, num_results: Optional[int] = None) -> List[Dict]:
        """
        Wyszukaj informacje w internecie.

        Args:
            query: Zapytanie wyszukiwania
            num_results: Liczba wyników (domyślnie max_results)

        Returns:
            Lista słowników z wynikami wyszukiwania
        """
        if num_results is None:
            num_results = self.max_results

        results = []
        
        try:
            # Użyj DuckDuckGo do wyszukiwania
            with DDGS() as ddgs:
                ddg_results = list(ddgs.text(query, max_results=num_results))
                
                for result in ddg_results:
                    results.append({
                        'title': result.get('title', ''),
                        'url': result.get('href', ''),
                        'snippet': result.get('body', ''),
                        'source': 'duckduckgo'
                    })
        except Exception as e:
            print(f"Błąd wyszukiwania: {e}")
            # Fallback do prostego wyszukiwania
            results = self._fallback_search(query, num_results)

        return results

    def _fallback_search(self, query: str, num_results: int) -> List[Dict]:
        """
        Zapasowa metoda wyszukiwania w przypadku awarii głównej.
        """
        # Prosta implementacja fallback - w produkcji można dodać inne API
        return [{
            'title': f'Wynik dla: {query}',
            'url': f'https://duckduckgo.com/?q={query}',
            'snippet': f'Brak szczegółowych wyników dla zapytania: {query}',
            'source': 'fallback'
        }]

    def fetch_page_content(self, url: str, max_length: int = 5000) -> str:
        """
        Pobierz i przeanalizuj zawartość strony internetowej.

        Args:
            url: Adres URL strony do pobrania
            max_length: Maksymalna długość zwracanego tekstu

        Returns:
            Przetworzona zawartość tekstowa strony
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Parsowanie HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Usuń skrypty i style
            for script in soup(['script', 'style', 'nav', 'footer', 'header']):
                script.decompose()
            
            # Ekstrahuj tekst
            text = soup.get_text(separator=' ', strip=True)
            
            # Wyczyść tekst
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\w\s.,!?-]', '', text)
            
            return text[:max_length]
            
        except Exception as e:
            return f"Błąd pobierania strony: {str(e)}"

    def search_and_extract(self, query: str, extract_depth: int = 2) -> Dict:
        """
        Wyszukaj i wyekstrahuj szczegółowe informacje.

        Args:
            query: Zapytanie wyszukiwania
            extract_depth: Liczba stron do głębszej analizy

        Returns:
            Słownik z podsumowaniem i szczegółowymi informacjami
        """
        # Wstępne wyszukiwanie
        search_results = self.search(query)
        
        if not search_results:
            return {
                'query': query,
                'summary': 'Brak wyników wyszukiwania.',
                'sources': [],
                'detailed_content': []
            }

        # Ekstrakcja szczegółów z top wyników
        detailed_content = []
        for i, result in enumerate(search_results[:extract_depth]):
            content = self.fetch_page_content(result['url'])
            detailed_content.append({
                'url': result['url'],
                'title': result['title'],
                'content': content[:2000]  # Ogranicz długość
            })

        # Stwórz podsumowanie
        summary_parts = [result['snippet'] for result in search_results[:3]]
        summary = ' '.join(summary_parts)

        return {
            'query': query,
            'summary': summary,
            'sources': [{'title': r['title'], 'url': r['url']} for r in search_results],
            'detailed_content': detailed_content
        }

    def verify_information(self, claim: str, sources: Optional[List[str]] = None) -> Dict:
        """
        Zweryfikuj informacje poprzez cross-referencing wielu źródeł.

        Args:
            claim: Twierdzenie do weryfikacji
            sources: Opcjonalna lista źródeł do sprawdzenia

        Returns:
            Wynik weryfikacji z poziomem pewności
        """
        # Wyszukaj potwierdzenia i zaprzeczeń
        confirm_search = self.search(f"{claim} true fact")
        deny_search = self.search(f"{claim} false hoax myth")
        
        confidence_score = 0.5  # Neutralny punkt startowy
        
        # Analiza wyników
        if len(confirm_search) > len(deny_search):
            confidence_score += 0.2
        elif len(deny_search) > len(confirm_search):
            confidence_score -= 0.2
            
        # Dodatkowa analiza snippetów
        positive_keywords = ['confirmed', 'true', 'fact', 'verified', 'official']
        negative_keywords = ['false', 'hoax', 'myth', 'debunked', 'fake']
        
        for result in confirm_search + deny_search:
            snippet_lower = result['snippet'].lower()
            for keyword in positive_keywords:
                if keyword in snippet_lower:
                    confidence_score += 0.05
            for keyword in negative_keywords:
                if keyword in snippet_lower:
                    confidence_score -= 0.05

        confidence_score = max(0.0, min(1.0, confidence_score))
        
        verification_result = "uncertain"
        if confidence_score > 0.7:
            verification_result = "likely_true"
        elif confidence_score < 0.3:
            verification_result = "likely_false"
        elif confidence_score > 0.55:
            verification_result = "probably_true"
        elif confidence_score < 0.45:
            verification_result = "probably_false"

        return {
            'claim': claim,
            'verification': verification_result,
            'confidence': confidence_score,
            'supporting_sources': confirm_search[:3],
            'contradicting_sources': deny_search[:3]
        }


# Przykładowe użycie
if __name__ == "__main__":
    engine = WebSearchEngine(max_results=3)
    
    # Test wyszukiwania
    print("Test wyszukiwania:")
    results = engine.search("sztuczna inteligencja 2024")
    for r in results:
        print(f"- {r['title']}: {r['snippet'][:100]}...")
    
    print("\nTest ekstrakcji:")
    extracted = engine.search_and_extract("najnowsze odkrycia naukowe 2024")
    print(f"Podsumowanie: {extracted['summary'][:200]}...")
    print(f"Liczba źródeł: {len(extracted['sources'])}")
