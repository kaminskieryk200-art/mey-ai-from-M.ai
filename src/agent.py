"""
Główny agent AI łączący model AI, wyszukiwanie internetowe i rozumowanie.
Inteligentny agent zdolny do analizy, wyszukiwania informacji i logicznego wnioskowania.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from .model import SimpleAIModel
from .web_search import WebSearchEngine
from .reasoning_model import ReasoningModel, ReasoningChain


class AIAgent:
    """
    Zaawansowany agent AI z własnym modelem, wyszukiwaniem internetowym 
    i zaawansowanym systemem rozumowania.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Inicjalizacja agenta AI.

        Args:
            config: Konfiguracja agenta (opcjonalna)
        """
        self.config = config or {}
        
        # Inicjalizacja komponentów
        self.model = SimpleAIModel(
            input_size=self.config.get('input_size', 100),
            hidden_size=self.config.get('hidden_size', 64),
            output_size=self.config.get('output_size', 10)
        )
        
        self.web_search = WebSearchEngine(
            max_results=self.config.get('max_search_results', 5),
            timeout=self.config.get('search_timeout', 10)
        )
        
        self.reasoning_model = ReasoningModel(
            max_depth=self.config.get('max_reasoning_depth', 10),
            confidence_threshold=self.config.get('confidence_threshold', 0.6)
        )
        
        # Stan agenta
        self.conversation_history: List[Dict] = []
        self.decision_log: List[Dict] = []
        self.is_initialized = False
        
        # Dodaj podstawową wiedzę
        self._initialize_knowledge()

    def _initialize_knowledge(self):
        """Inicjalizacja podstawowej wiedzy agenta"""
        base_knowledge = {
            'ai': 'Sztuczna inteligencja to symulacja ludzkiej inteligencji przez maszyny',
            'ml': 'Machine Learning to podzbiór AI uczący się z danych',
            'dl': 'Deep Learning używa wielowarstwowych sieci neuronowych',
            'nlp': 'Przetwarzanie języka naturalnego umożliwia komunikację z ludźmi',
            'agent': 'Autonomiczny system AI wykonujący zadania'
        }
        
        for key, value in base_knowledge.items():
            self.reasoning_model.add_knowledge(key, value)
        
        self.is_initialized = True

    def process_query(self, query: str, use_web_search: bool = True, 
                     use_reasoning: bool = True) -> Dict:
        """
        Przetwórz zapytanie użytkownika używając wszystkich dostępnych narzędzi.

        Args:
            query: Zapytanie od użytkownika
            use_web_search: Czy użyć wyszukiwania internetowego
            use_reasoning: Czy użyć zaawansowanego rozumowania

        Returns:
            Słownik z odpowiedzią i metadanymi
        """
        timestamp = datetime.now().isoformat()
        
        result = {
            'query': query,
            'timestamp': timestamp,
            'components_used': [],
            'response': '',
            'confidence': 0.0,
            'sources': [],
            'reasoning_chain': None
        }
        
        # Krok 1: Analiza intencji za pomocą modelu AI
        intent_analysis = self._analyze_intent(query)
        result['components_used'].append('ai_model')
        result['intent'] = intent_analysis
        
        # Krok 2: Wyszukiwanie internetowe (jeśli włączono)
        web_data = None
        if use_web_search:
            web_data = self._perform_web_search(query)
            result['components_used'].append('web_search')
            result['sources'] = web_data.get('sources', [])
            result['web_summary'] = web_data.get('summary', '')
        
        # Krok 3: Zaawansowane rozumowanie (jeśli włączono)
        reasoning_result = None
        if use_reasoning:
            context = {
                'intent': intent_analysis,
                'web_data': web_data,
                'facts': web_data.get('summary', '').split('.')[:3] if web_data else []
            }
            reasoning_result = self._perform_reasoning(query, context)
            result['components_used'].append('reasoning_model')
            result['reasoning_chain'] = reasoning_result.to_dict()
            result['reasoning_summary'] = self.reasoning_model.get_reasoning_summary(reasoning_result)
        
        # Krok 4: Synteza końcowej odpowiedzi
        final_response = self._synthesize_response(
            query, 
            intent_analysis, 
            web_data, 
            reasoning_result
        )
        
        result['response'] = final_response['text']
        result['confidence'] = final_response['confidence']
        
        # Zapisz do historii
        self._log_interaction(query, result)
        
        return result

    def _analyze_intent(self, query: str) -> Dict:
        """Analiza intencji zapytania za pomocą modelu AI"""
        # Przygotuj dane wejściowe (prosty embedding)
        input_vector = self._text_to_vector(query)
        
        # Predykcja modelu
        prediction = self.model.predict([input_vector])[0]
        
        # Mapowanie na typy intencji
        intent_types = [
            'information_request',
            'problem_solving',
            'analysis_request',
            'comparison',
            'definition',
            'how_to',
            'why_question',
            'what_question',
            'opinion_request',
            'other'
        ]
        
        predicted_intent = intent_types[prediction]
        confidence = float(np.max(prediction))
        
        return {
            'type': predicted_intent,
            'confidence': confidence,
            'query_length': len(query),
            'keywords': self.reasoning_model._extract_keywords(query)
        }

    def _text_to_vector(self, text: str) -> np.ndarray:
        """Konwertuj tekst na wektor wejściowy dla modelu"""
        # Prosta implementacja - w produkcji użyj prawdziwych embeddingów
        words = text.lower().split()
        vector = np.zeros(self.model.input_size)
        
        # Hashowanie słów do indeksów wektora
        for i, word in enumerate(words[:50]):  # Ogranicz do 50 słów
            idx = hash(word) % self.model.input_size
            vector[idx] += 1
        
        # Normalizacja
        if np.sum(vector) > 0:
            vector = vector / np.sum(vector)
        
        return vector

    def _perform_web_search(self, query: str) -> Dict:
        """Przeprowadź wyszukiwanie internetowe"""
        try:
            search_data = self.web_search.search_and_extract(
                query, 
                extract_depth=self.config.get('search_depth', 2)
            )
            return search_data
        except Exception as e:
            return {
                'query': query,
                'summary': f'Błąd wyszukiwania: {str(e)}',
                'sources': [],
                'detailed_content': []
            }

    def _perform_reasoning(self, problem: str, context: Dict) -> ReasoningChain:
        """Przeprowadź zaawansowane rozumowanie"""
        return self.reasoning_model.analyze_problem(problem, context)

    def _synthesize_response(self, query: str, intent: Dict, 
                            web_data: Optional[Dict], 
                            reasoning: Optional[ReasoningChain]) -> Dict:
        """Syntezuj końcową odpowiedź ze wszystkich komponentów"""
        
        response_parts = []
        confidence_scores = []
        
        # Część z rozumowania
        if reasoning and reasoning.conclusion:
            response_parts.append(f"🧠 **Analiza i wnioski:**\n{reasoning.conclusion}")
            confidence_scores.append(reasoning.confidence)
        
        # Część z wyszukiwania
        if web_data and web_data.get('summary'):
            summary = web_data['summary']
            if len(summary) > 300:
                summary = summary[:300] + "..."
            response_parts.append(f"\n📊 **Informacje z sieci:**\n{summary}")
            confidence_scores.append(0.7)  # Ufanie danym z sieci
            
            # Dodaj źródła
            if web_data.get('sources'):
                sources_list = "\n".join([
                    f"  - {s['title']}" for s in web_data['sources'][:3]
                ])
                response_parts.append(f"\n📚 **Źródła:**\n{sources_list}")
        
        # Jeśli brak danych, użyj samego modelu
        if not response_parts:
            response_parts.append(
                f"Na podstawie analizy Twojego zapytania o '{query[:50]}...', "
                f"mogę stwierdzić, że wymaga ono {'głębszej analizy' if intent['confidence'] < 0.5 else 'standardowej odpowiedzi'}."
            )
            confidence_scores.append(intent['confidence'])
        
        # Oblicz całkowite zaufanie
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.5
        
        # Dodaj stopkę
        response_parts.append(
            f"\n\n💡 **Poziom pewności:** {overall_confidence:.1%} | "
            f"Użyte komponenty: {len([intent, web_data, reasoning]) - [intent, web_data, reasoning].count(None)}/3"
        )
        
        return {
            'text': '\n'.join(response_parts),
            'confidence': overall_confidence
        }

    def _log_interaction(self, query: str, result: Dict):
        """Zapisz interakcję do historii"""
        self.conversation_history.append({
            'timestamp': result['timestamp'],
            'query': query,
            'response_preview': result['response'][:100],
            'confidence': result['confidence'],
            'components_used': result['components_used']
        })
        
        # Ogranicz historię do ostatnich 100 interakcji
        if len(self.conversation_history) > 100:
            self.conversation_history = self.conversation_history[-100:]

    def chat(self, message: str, enable_web: bool = True, 
             enable_reasoning: bool = True) -> str:
        """
        Prosty interfejs czatu.

        Args:
            message: Wiadomość od użytkownika
            enable_web: Włącz wyszukiwanie internetowe
            enable_reasoning: Włącz zaawansowane rozumowanie

        Returns:
            Odpowiedź agenta jako tekst
        """
        result = self.process_query(
            message,
            use_web_search=enable_web,
            use_reasoning=enable_reasoning
        )
        return result['response']

    def get_stats(self) -> Dict:
        """Pobierz statystyki działania agenta"""
        return {
            'is_initialized': self.is_initialized,
            'total_interactions': len(self.conversation_history),
            'avg_confidence': np.mean([h['confidence'] for h in self.conversation_history]) if self.conversation_history else 0.0,
            'most_used_component': self._get_most_used_component(),
            'knowledge_base_size': len(self.reasoning_model.knowledge_base)
        }

    def _get_most_used_component(self) -> str:
        """Znajdź najczęściej używany komponent"""
        if not self.conversation_history:
            return 'none'
        
        component_counts = {}
        for interaction in self.conversation_history:
            for component in interaction.get('components_used', []):
                component_counts[component] = component_counts.get(component, 0) + 1
        
        if not component_counts:
            return 'none'
        
        return max(component_counts, key=component_counts.get)

    def export_conversation(self, filename: str = 'conversation_log.json'):
        """Eksportuj historię konwersacji do pliku JSON"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
        return filename


# Import numpy na poziomie modułu
import numpy as np


# Przykładowe użycie
if __name__ == "__main__":
    print("🤖 Inicjalizacja Agent AI...")
    agent = AIAgent()
    
    print(f"✅ Agent gotowy! Statystyki: {agent.get_stats()}")
    print("\n" + "="*60 + "\n")
    
    # Przykładowe zapytania
    queries = [
        "Jak sztuczna inteligencja zmienia świat w 2024 roku?",
        "Czy warto uczyć się programowania w erze AI?",
        "Jakie są najnowsze odkrycia w dziedzinie machine learning?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n📝 Zapytanie {i}: {query}")
        print("-" * 60)
        response = agent.chat(query, enable_web=True, enable_reasoning=True)
        print(response)
        print("\n")
    
    # Podsumowanie
    print("\n" + "="*60)
    print("📊 PODSUMOWANIE SESJI")
    print("="*60)
    stats = agent.get_stats()
    print(f"Liczba interakcji: {stats['total_interactions']}")
    print(f"Średnie zaufanie: {stats['avg_confidence']:.1%}")
    print(f"Najczęściej używany komponent: {stats['most_used_component']}")
    print(f"Rozmiar bazy wiedzy: {stats['knowledge_base_size']}")
