"""
Prosta aplikacja webowa dla Agent AI z własnym modelem.
Uruchomienie: python app.py
Dostęp pod: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import sys
import os

# Dodaj ścieżkę src do path, żeby zaimportować agenta
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from agent import AIAgent
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Ostrzeżenie: Nie można zaimportować agenta: {e}")
    AGENT_AVAILABLE = False

app = Flask(__name__)

# Inicjalizacja agenta
agent = None
if AGENT_AVAILABLE:
    try:
        agent = AIAgent()
        print("✅ Agent AI został pomyślnie załadowany.")
    except Exception as e:
        print(f"❌ Błąd inicjalizacji agenta: {e}")
        agent = None
else:
    print("⚠️  Tryb demonstracyjny (brak pełnego agenta).")

@app.route('/')
def index():
    """Strona główna z interfejsem czatu."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Endpoint API do obsługi wiadomości od użytkownika."""
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'Wiadomość nie może być pusta'}), 400
    
    if not agent:
        return jsonify({
            'response': "Tryb demonstracyjny: Agent nie jest w pełni skonfigurowany. Sprawdź logi.",
            'sources': [],
            'reasoning': "Brak połączenia z modułem AI."
        }), 200
    
    try:
        # Wywołanie agenta
        response_data = agent.run(user_message)
        
        # Formatowanie odpowiedzi (dostosuj do struktury zwracanej przez agenta)
        if isinstance(response_data, dict):
            reply = response_data.get('answer', response_data.get('response', str(response_data)))
            sources = response_data.get('sources', [])
            reasoning = response_data.get('reasoning', "Przetwarzanie...")
        else:
            reply = str(response_data)
            sources = []
            reasoning = "Analiza wykonana."

        return jsonify({
            'response': reply,
            'sources': sources,
            'reasoning': reasoning
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Błąd przetwarzania: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Uruchamianie serwera aplikacji AI Agent...")
    print("🌐 Dostęp pod adresem: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
