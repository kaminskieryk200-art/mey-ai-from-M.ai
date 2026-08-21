"""
Zaawansowany model rozumowania (Reasoning Model) dla agenta AI.
Implementuje wieloetapowe wnioskowanie, analizę logiczną i rozwiązywanie problemów.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import json


class ReasoningStepType(Enum):
    """Typy kroków rozumowania"""
    ANALYSIS = "analysis"
    DEDUCTION = "deduction"
    INDUCTION = "induction"
    ABDUCTION = "abduction"
    EVALUATION = "evaluation"
    SYNTHESIS = "synthesis"
    VERIFICATION = "verification"


class ReasoningChain:
    """
    Reprezentuje łańcuch rozumowania z wieloma krokami.
    """

    def __init__(self, problem: str):
        self.problem = problem
        self.steps: List[Dict] = []
        self.conclusion: Optional[str] = None
        self.confidence: float = 0.0
        self.alternatives: List[Dict] = []

    def add_step(self, step_type: ReasoningStepType, content: str, 
                 confidence: float, dependencies: Optional[List[int]] = None):
        """Dodaj krok do łańcucha rozumowania"""
        step = {
            'id': len(self.steps),
            'type': step_type.value,
            'content': content,
            'confidence': confidence,
            'dependencies': dependencies or [],
            'timestamp': len(self.steps)
        }
        self.steps.append(step)
        return step['id']

    def get_chain_confidence(self) -> float:
        """Oblicz całkowite zaufanie do łańcucha rozumowania"""
        if not self.steps:
            return 0.0
        
        # Średnia ważona zaufania, z karą za długie łańcuchy
        base_confidence = np.mean([s['confidence'] for s in self.steps])
        length_penalty = 0.95 ** len(self.steps)  # Kara za długość
        return base_confidence * length_penalty

    def to_dict(self) -> Dict:
        """Konwertuj do słownika"""
        return {
            'problem': self.problem,
            'steps': self.steps,
            'conclusion': self.conclusion,
            'confidence': self.get_chain_confidence(),
            'alternatives': self.alternatives
        }


class ReasoningModel:
    """
    Zaawansowany model rozumowania dla agenta AI.
    Implementuje różne strategie wnioskowania i ocenę logiczną.
    """

    def __init__(self, max_depth: int = 10, confidence_threshold: float = 0.6):
        """
        Inicjalizacja modelu rozumowania.

        Args:
            max_depth: Maksymalna głębokość łańcucha rozumowania
            confidence_threshold: Minimalny próg zaufania dla wniosków
        """
        self.max_depth = max_depth
        self.confidence_threshold = confidence_threshold
        self.knowledge_base: Dict[str, Any] = {}
        self.reasoning_history: List[ReasoningChain] = []

    def analyze_problem(self, problem: str, context: Optional[Dict] = None) -> ReasoningChain:
        """
        Przeanalizuj problem i zbuduj łańcuch rozumowania.

        Args:
            problem: Opis problemu do rozwiązania
            context: Dodatkowy kontekst (dane, fakty, ograniczenia)

        Returns:
            Obiekt ReasoningChain z pełnym rozumowaniem
        """
        chain = ReasoningChain(problem)
        
        # Krok 1: Analiza problemu
        analysis = self._perform_analysis(problem, context)
        chain.add_step(
            ReasoningStepType.ANALYSIS,
            analysis['breakdown'],
            analysis['confidence']
        )

        # Krok 2: Identyfikacja znanych faktów
        facts = self._identify_facts(problem, context)
        if facts:
            chain.add_step(
                ReasoningStepType.DEDUCTION,
                f"Zidentyfikowane fakty: {', '.join(facts)}",
                0.9
            )

        # Krok 3: Generowanie hipotez
        hypotheses = self._generate_hypotheses(problem, facts, context)
        for i, hyp in enumerate(hypotheses[:3]):  # Top 3 hipotezy
            chain.add_step(
                ReasoningStepType.ABDUCTION,
                f"Hipoteza {i+1}: {hyp['hypothesis']}",
                hyp['confidence']
            )

        # Krok 4: Wnioskowanie dedukcyjne
        deductions = self._perform_deduction(hypotheses, facts)
        for deduction in deductions:
            chain.add_step(
                ReasoningStepType.DEDUCTION,
                deduction,
                0.85,
                dependencies=[len(chain.steps) - 1]
            )

        # Krok 5: Synteza i wniosek
        conclusion, confidence = self._synthesize_conclusion(chain)
        chain.conclusion = conclusion
        chain.confidence = confidence

        # Krok 6: Weryfikacja
        verification = self._verify_reasoning(chain)
        chain.add_step(
            ReasoningStepType.VERIFICATION,
            verification['status'],
            verification['confidence']
        )

        self.reasoning_history.append(chain)
        return chain

    def _perform_analysis(self, problem: str, context: Optional[Dict]) -> Dict:
        """Przeprowadź analizę problemu"""
        # Symulacja analizy - w produkcji można użyć LLM
        keywords = self._extract_keywords(problem)
        complexity = min(1.0, len(problem) / 200)  # Prosta metryka złożoności
        
        breakdown = (
            f"Problem zawiera {len(keywords)} kluczowych elementów. "
            f"Złożoność szacowana na {complexity:.2f}. "
            f"Kluczowe koncepcje: {', '.join(keywords[:5])}"
        )
        
        return {
            'breakdown': breakdown,
            'confidence': 0.85,
            'keywords': keywords,
            'complexity': complexity
        }

    def _extract_keywords(self, text: str) -> List[str]:
        """Ekstrahuj kluczowe słowa z tekstu"""
        # Prosta implementacja - w produkcji użyj NLP
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                      'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
                      'who', 'whom', 'this', 'that', 'these', 'those', 'am'}
        
        words = text.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        return list(set(keywords))[:10]

    def _identify_facts(self, problem: str, context: Optional[Dict]) -> List[str]:
        """Zidentyfikuj fakty w problemie i kontekście"""
        facts = []
        
        # Sprawdź kontekst
        if context:
            if 'facts' in context:
                facts.extend(context['facts'])
            if 'data' in context:
                facts.append(f"Dostępne dane: {len(context['data'])} elementów")
        
        # Sprawdź bazę wiedzy
        keywords = self._extract_keywords(problem)
        for kw in keywords:
            if kw in self.knowledge_base:
                facts.append(f"Znany fakt o '{kw}': {self.knowledge_base[kw]}")
        
        return facts

    def _generate_hypotheses(self, problem: str, facts: List[str], 
                            context: Optional[Dict]) -> List[Dict]:
        """Generuj możliwe hipotezy wyjaśniające"""
        hypotheses = []
        
        # Hipoteza bazowa
        hypotheses.append({
            'hypothesis': "Najprostsze wyjaśnienie oparte na dostępnych danych",
            'confidence': 0.7,
            'supporting_facts': facts[:2] if facts else []
        })
        
        # Hipoteza alternatywna
        hypotheses.append({
            'hypothesis': "Alternatywne wyjaśnienie uwzględniające czynniki zewnętrzne",
            'confidence': 0.5,
            'supporting_facts': facts[1:3] if len(facts) > 1 else []
        })
        
        # Hipoteza konserwatywna
        hypotheses.append({
            'hypothesis': "Konserwatywne podejście wymagające dodatkowej weryfikacji",
            'confidence': 0.6,
            'supporting_facts': []
        })
        
        # Sortuj po zaufaniu
        hypotheses.sort(key=lambda x: x['confidence'], reverse=True)
        return hypotheses

    def _perform_deduction(self, hypotheses: List[Dict], 
                          facts: List[str]) -> List[str]:
        """Przeprowadź wnioskowanie dedukcyjne"""
        deductions = []
        
        if not hypotheses:
            return ["Brak wystarczających danych do dedukcji"]
        
        top_hypothesis = hypotheses[0]
        
        # Dedukcja z głównej hipotezy
        if facts:
            deductions.append(
                f"Biorąc pod uwagę fakty ({len(facts)}) i główną hipotezę, "
                f"można wywnioskować spójne rozwiązanie"
            )
        else:
            deductions.append(
                "Na podstawie samej hipotezy wnioskujemy o prawdopodobnym scenariuszu"
            )
        
        # Sprawdzenie spójności
        if len(hypotheses) > 1:
            confidence_diff = abs(hypotheses[0]['confidence'] - hypotheses[1]['confidence'])
            if confidence_diff < 0.2:
                deductions.append(
                    "Alternatywne hipotezy mają podobne zaufanie - wymagana dodatkowa analiza"
                )
        
        return deductions

    def _synthesize_conclusion(self, chain: ReasoningChain) -> Tuple[str, float]:
        """Syntezuj końcowy wniosek z łańcucha rozumowania"""
        if not chain.steps:
            return "Brak podstaw do wyciągnięcia wniosków", 0.0
        
        # Agregacja wniosków z poszczególnych kroków
        step_contents = [s['content'] for s in chain.steps[-3:]]  # Ostatnie 3 kroki
        
        # Prosta synteza
        conclusion = (
            f"Na podstawie analizy problemu: '{chain.problem[:50]}...' "
            f"oraz {len(chain.steps)} kroków rozumowania, "
            f"główny wniosek jest następujący: Rozwiązanie istnieje i może być osiągnięte "
            f"poprzez systematyczne podejście przedstawione w krokach analizy."
        )
        
        # Oblicz zaufanie
        confidence = chain.get_chain_confidence()
        
        return conclusion, confidence

    def _verify_reasoning(self, chain: ReasoningChain) -> Dict:
        """Zweryfikuj poprawność łańcucha rozumowania"""
        issues = []
        
        # Sprawdź spójność
        if len(chain.steps) < 2:
            issues.append("Zbyt mało kroków rozumowania")
        
        # Sprawdź zaufanie
        avg_confidence = np.mean([s['confidence'] for s in chain.steps])
        if avg_confidence < self.confidence_threshold:
            issues.append(f"Średnie zaufanie ({avg_confidence:.2f}) poniżej progu")
        
        # Sprawdź zależności
        for step in chain.steps:
            for dep in step.get('dependencies', []):
                if dep >= step['id']:
                    issues.append(f"Niespójna zależność w kroku {step['id']}")
        
        status = "VALID" if not issues else f"ISSUES: {'; '.join(issues)}"
        verification_confidence = 1.0 - (len(issues) * 0.2)
        
        return {
            'status': status,
            'confidence': max(0.0, verification_confidence),
            'issues': issues
        }

    def add_knowledge(self, key: str, value: Any):
        """Dodaj informację do bazy wiedzy"""
        self.knowledge_base[key] = value

    def clear_knowledge(self):
        """Wyczyść bazę wiedzy"""
        self.knowledge_base.clear()

    def get_reasoning_summary(self, chain: ReasoningChain) -> str:
        """Generuj podsumowanie rozumowania w formacie czytelnym dla człowieka"""
        summary_lines = [
            "=" * 60,
            "PODSUMOWANIE ROZUMOWANIA",
            "=" * 60,
            f"Problem: {chain.problem}",
            f"Liczba kroków: {len(chain.steps)}",
            f"Końcowe zaufanie: {chain.get_chain_confidence():.2%}",
            "",
            "KROKI ROZUMOWANIA:"
        ]
        
        for step in chain.steps:
            icon = {
                'analysis': '🔍',
                'deduction': '➡️',
                'induction': '📊',
                'abduction': '💡',
                'evaluation': '⚖️',
                'synthesis': '🧩',
                'verification': '✅'
            }.get(step['type'], '📌')
            
            summary_lines.append(
                f"{icon} [{step['type'].upper()}] {step['content'][:100]}..."
                if len(step['content']) > 100
                else f"{icon} [{step['type'].upper()}] {step['content']}"
            )
        
        summary_lines.extend([
            "",
            "WNIOSEK:",
            f"{chain.conclusion}",
            "",
            f"Poziom pewności: {chain.confidence:.2%}",
            "=" * 60
        ])
        
        return "\n".join(summary_lines)


# Przykładowe użycie
if __name__ == "__main__":
    model = ReasoningModel()
    
    # Dodaj trochę wiedzy
    model.add_knowledge("ai", "Sztuczna inteligencja to dziedzina informatyki")
    model.add_knowledge("ml", "Machine Learning to podzbiór AI")
    
    # Test rozumowania
    problem = "Jak sztuczna inteligencja może pomóc w diagnostyce medycznej?"
    context = {
        'facts': ['AI potrafi analizować obrazy', 'Lekarze mają ograniczony czas'],
        'data': [{'type': 'x-ray', 'count': 1000}]
    }
    
    chain = model.analyze_problem(problem, context)
    
    print(model.get_reasoning_summary(chain))
    
    print("\n\nPełny JSON:")
    print(json.dumps(chain.to_dict(), indent=2, ensure_ascii=False))
