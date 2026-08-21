"""
Prosty model AI dla agenta - implementacja od podstaw.
Ten model to mała sieć neuronowa do klasyfikacji tekstu/intencji.
"""

import numpy as np
from typing import List, Dict, Tuple
import pickle
import os

class SimpleAIModel:
    """
    Prosty model AI oparty na sieci neuronowej z warstwą ukrytą.
    Przeznaczony do rozpoznawania intencji w prostym agencie.
    """
    
    def __init__(self, input_size: int = 100, hidden_size: int = 64, output_size: int = 10):
        """
        Inicjalizacja modelu z losowymi wagami.
        
        Args:
            input_size: Rozmiar wektora wejściowego (np. embedding tekstu)
            hidden_size: Liczba neuronów w warstwie ukrytej
            output_size: Liczba klas wyjściowych (intencji)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Inicjalizacja wag (Xavier initialization)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Historia treningu
        self.training_history = []
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """Funkcja aktywacji ReLU"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Pochodna funkcji ReLU"""
        return (x > 0).astype(float)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Funkcja aktywacji Softmax dla warstwy wyjściowej"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Propagacja w przód.
        
        Args:
            X: Dane wejściowe o kształcie (batch_size, input_size)
            
        Returns:
            Wyjście modelu (prawdopodobieństwa klas)
        """
        # Warstwa ukryta
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Warstwa wyjściowa
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        
        return self.a2
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Obliczanie straty cross-entropy.
        
        Args:
            y_pred: Przewidywania modelu
            y_true: Prawdziwe etykiety (one-hot encoded)
            
        Returns:
            Wartość straty
        """
        batch_size = y_true.shape[0]
        # Dodajemy małą wartość dla stabilności numerycznej
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -np.sum(y_true * np.log(y_pred)) / batch_size
        return loss
    
    def backward(self, X: np.ndarray, y_true: np.ndarray, learning_rate: float = 0.01):
        """
        Propagacja w tył i aktualizacja wag.
        
        Args:
            X: Dane wejściowe
            y_true: Prawdziwe etykiety (one-hot encoded)
            learning_rate: Współczynnik uczenia
        """
        batch_size = X.shape[0]
        
        # Gradient dla warstwy wyjściowej
        dz2 = self.a2 - y_true
        dW2 = np.dot(self.a1.T, dz2) / batch_size
        db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size
        
        # Gradient dla warstwy ukrytej
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / batch_size
        db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size
        
        # Aktualizacja wag
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              epochs: int = 100, learning_rate: float = 0.01, 
              verbose: bool = True) -> List[float]:
        """
        Trening modelu.
        
        Args:
            X_train: Dane treningowe
            y_train: Etykiety treningowe (one-hot encoded)
            epochs: Liczba epok
            learning_rate: Współczynnik uczenia
            verbose: Czy wyświetlać postęp
            
        Returns:
            Historia strat
        """
        self.training_history = []
        
        for epoch in range(epochs):
            # Propagacja w przód
            y_pred = self.forward(X_train)
            
            # Obliczanie straty
            loss = self.compute_loss(y_pred, y_train)
            self.training_history.append(loss)
            
            # Propagacja w tył
            self.backward(X_train, y_train, learning_rate)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoka {epoch + 1}/{epochs}, Strata: {loss:.4f}")
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predykcja klasy.
        
        Args:
            X: Dane wejściowe
            
        Returns:
            Indeksy przewidzianych klas
        """
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predykcja prawdopodobieństw klas.
        
        Args:
            X: Dane wejściowe
            
        Returns:
            Prawdopodobieństwa klas
        """
        return self.forward(X)
    
    def save(self, filepath: str):
        """Zapisz model do pliku"""
        model_data = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'training_history': self.training_history
        }
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model zapisany do {filepath}")
    
    def load(self, filepath: str):
        """Wczytaj model z pliku"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.input_size = model_data['input_size']
        self.hidden_size = model_data['hidden_size']
        self.output_size = model_data['output_size']
        self.W1 = model_data['W1']
        self.b1 = model_data['b1']
        self.W2 = model_data['W2']
        self.b2 = model_data['b2']
        self.training_history = model_data.get('training_history', [])
        
        print(f"Model wczytany z {filepath}")


def create_sample_data(num_samples: int = 1000, num_classes: int = 5, 
                       input_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tworzenie przykładowych danych treningowych.
    
    Args:
        num_samples: Liczba próbek
        num_classes: Liczba klas
        input_size: Rozmiar wektora wejściowego
        
    Returns:
        X_train, y_train (one-hot encoded)
    """
    # Generowanie losowych danych
    X = np.random.randn(num_samples, input_size)
    
    # Tworzenie etykiet (każda klasa ma trochę inną charakterystykę)
    y = np.zeros((num_samples, num_classes))
    for i in range(num_samples):
        # Dodajemy sygnał zależny od klasy
        true_class = i % num_classes
        X[i, true_class*20:(true_class+1)*20] += 2  # Wzmocnienie cech dla danej klasy
        y[i, true_class] = 1
    
    return X, y


if __name__ == "__main__":
    print("=== Trening prostego modelu AI ===\n")
    
    # Parametry
    INPUT_SIZE = 100
    HIDDEN_SIZE = 64
    OUTPUT_SIZE = 5
    NUM_SAMPLES = 1000
    EPOCHS = 100
    LEARNING_RATE = 0.1
    
    # Tworzenie danych
    print("Generowanie danych treningowych...")
    X_train, y_train = create_sample_data(NUM_SAMPLES, OUTPUT_SIZE, INPUT_SIZE)
    print(f"Dane: {X_train.shape}, Etykiety: {y_train.shape}\n")
    
    # Inicjalizacja modelu
    model = SimpleAIModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE
    )
    
    # Trening
    print("Rozpoczynanie treningu...")
    history = model.train(
        X_train, y_train,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        verbose=True
    )
    
    # Testowanie
    print("\n=== Testowanie modelu ===")
    X_test = np.random.randn(10, INPUT_SIZE)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    print(f"\nPrzykładowe predykcje:")
    for i in range(5):
        print(f"Próbka {i+1}: Klasa={predictions[i]}, Pewność={probabilities[i][predictions[i]]:.3f}")
    
    # Zapis modelu
    model_path = "models/simple_ai_model.pkl"
    model.save(model_path)
    
    print("\n=== Gotowe! ===")
    print(f"Model wytrenowany przez {EPOCHS} epok")
    print(f"Końcowa strata: {history[-1]:.4f}")
    print(f"Model zapisany w: {model_path}")
