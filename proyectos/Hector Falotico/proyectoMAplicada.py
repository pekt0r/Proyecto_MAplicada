# Importación de bibliotecas necesarias
from dataclasses import dataclass  
import pandas as pd  # Para manipulación de datos
import nltk  # Para procesamiento de lenguaje natural
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # Para análisis de sentimientos
from nltk.corpus import stopwords  # Para eliminar palabras comunes sin significado
from nltk.tokenize import word_tokenize  # Para dividir texto en palabras
from nltk.stem import WordNetLemmatizer  # Para obtener la raíz de las palabras
import numpy as np  
import skfuzzy as fuzz  
import time  # Para medir tiempos de ejecución
import re  # Para expresiones regulares
from tqdm import tqdm  # Para barras de progreso
from typing import Tuple, List  # Para tipado de datos

@dataclass
class SentimentScores:
    """Clase para almacenar los puntajes de sentimiento positivo y negativo"""
    positive: float
    negative: float

class TextPreprocessor:
    """Clase para preprocesar texto: limpieza, tokenización y lematización"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text: str) -> str:
        """
        Proceso principal de limpieza de texto
        1. Convierte a minúsculas
        2. Aplica limpieza con regex
        3. Tokeniza
        4. Remueve stopwords
        5. Lematiza
        """
        text = text.lower()
        text = self._apply_regex_cleaning(text)
        tokens = word_tokenize(text)
        tokens = self._remove_stopwords(tokens)
        return ' '.join(self._lemmatize_tokens(tokens))
    
    def _apply_regex_cleaning(self, text: str) -> str:
        """
        Aplica patrones de regex para limpiar el texto:
        - Elimina URLs
        - Elimina menciones (@usuario)
        - Elimina puntuación y números
        - Normaliza espacios
        """
        patterns = [
            (r'http[s]?://\S+', ''),  # URLs
            (r'www\S+', ''),          # Enlaces web
            (r'@\S+', ''),            # Menciones
            (r'[^\w\s]|[\d]', ' '),   # Puntuación y números
            (r'\s\s+', ' '),          # Espacios múltiples
            (r'_\S+', ''),            # Palabras con guión bajo
            (r'^[a-zA-Z]$', '')       # Caracteres sueltos
        ]
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)
        return text
    
    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Elimina palabras comunes sin significado relevante"""
        return [token for token in tokens if token not in stopwords.words('english')]
    
    def _lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Convierte las palabras a su forma base (sin conjugar)"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]

class SentimentAnalyzer:
    """Clase para realizar análisis de sentimientos usando VADER"""
    
    def __init__(self):
        """Inicializa el analizador de sentimientos VADER"""
        self.analyzer = SentimentIntensityAnalyzer()
        
    def get_scores(self, text: str) -> SentimentScores:
        """Obtiene los puntajes de sentimiento positivo y negativo"""
        scores = self.analyzer.polarity_scores(text)
        return SentimentScores(
            positive=round(scores['pos'], 2),
            negative=round(scores['neg'], 2)
        )

class FuzzyLogicProcessor:
    """Clase para procesar lógica difusa en el análisis de sentimientos"""
    
    def __init__(self, score_ranges: dict):
        """
        Inicializa los rangos para las funciones de membresía
        Crea los universos de discurso para positivo, negativo y salida
        """
        self.x_pos = np.arange(0, 1, 0.001)
        self.x_neg = np.arange(0, 1, 0.001)
        self.x_output = np.arange(0, 10, 0.001)
        self._initialize_membership_functions(score_ranges)
        
    def _initialize_membership_functions(self, ranges: dict):
        """
        Inicializa las funciones de membresía triangulares para:
        - Sentimiento positivo (bajo, medio, alto)
        - Sentimiento negativo (bajo, medio, alto)
        - Salida (negativo, neutro, positivo)
        """
        # Funciones de membresía para sentimiento positivo
        self.low_pos = fuzz.trimf(self.x_pos, [ranges['min_pos'], ranges['min_pos'], ranges['mid_pos']])
        self.medium_pos = fuzz.trimf(self.x_pos, [ranges['min_pos'], ranges['mid_pos'], ranges['max_pos']])
        self.high_pos = fuzz.trimf(self.x_pos, [ranges['mid_pos'], ranges['max_pos'], ranges['max_pos']])
        
        # Funciones de membresía para sentimiento negativo
        self.low_neg = fuzz.trimf(self.x_neg, [ranges['min_neg'], ranges['min_neg'], ranges['mid_neg']])
        self.medium_neg = fuzz.trimf(self.x_neg, [ranges['min_neg'], ranges['mid_neg'], ranges['max_neg']])
        self.high_neg = fuzz.trimf(self.x_neg, [ranges['mid_neg'], ranges['max_neg'], ranges['max_neg']])
        
        # Funciones de membresía para la salida
        self.op_neg = fuzz.trimf(self.x_output, [0, 0, 5])
        self.op_neu = fuzz.trimf(self.x_output, [0, 5, 10])
        self.op_pos = fuzz.trimf(self.x_output, [5, 10, 10])

    def process(self, scores: SentimentScores) -> Tuple[float, str]:
        """
        Proceso principal de lógica difusa:
        1. Calcula membresías
        2. Aplica reglas de Mamdani
        3. Agrega resultados
        4. Desfuzzifica
        """
        fuzz_start = time.time()
        
        memberships = self._calculate_memberships(scores)
        rules = self._apply_mandani_rules(memberships)
        aggregated = self._aggregate_rules(rules)
        
        fuzz_time = time.time() - fuzz_start
        defuzz_start = time.time()
        
        output = round(fuzz.centroid(self.x_output, aggregated), 2)
        sentiment = self._determine_sentiment(output)
        
        defuzz_time = time.time() - defuzz_start
        
        return output, sentiment, fuzz_time, defuzz_time
    
    def _calculate_memberships(self, scores: SentimentScores) -> dict:
        """Calcula los grados de membresía para cada función"""
        return {
            'low_pos': fuzz.interp_membership(self.x_pos, self.low_pos, scores.positive),
            'mid_pos': fuzz.interp_membership(self.x_pos, self.medium_pos, scores.positive),
            'high_pos': fuzz.interp_membership(self.x_pos, self.high_pos, scores.positive),
            'low_neg': fuzz.interp_membership(self.x_neg, self.low_neg, scores.negative),
            'mid_neg': fuzz.interp_membership(self.x_neg, self.medium_neg, scores.negative),
            'high_neg': fuzz.interp_membership(self.x_neg, self.high_neg, scores.negative)
        }
    
    def _apply_mandani_rules(self, m: dict) -> dict:
        """Aplica las reglas de inferencia de Mamdani"""
        return {
            'R1': np.fmin(m['low_pos'], m['low_neg']),    # Si pos bajo y neg bajo -> neutro
            'R2': np.fmin(m['mid_pos'], m['low_neg']),    # Si pos medio y neg bajo -> positivo
            'R3': np.fmin(m['high_pos'], m['low_neg']),   # Si pos alto y neg bajo -> positivo
            'R4': np.fmin(m['mid_neg'], m['low_pos']),    # Si neg medio y pos bajo -> negativo
            'R5': np.fmin(m['mid_neg'], m['mid_pos']),    # Si neg medio y pos medio -> neutro
            'R6': np.fmin(m['mid_neg'], m['high_pos']),   # Si neg medio y pos alto -> positivo
            'R7': np.fmin(m['high_neg'], m['low_pos']),   # Si neg alto y pos bajo -> negativo
            'R8': np.fmin(m['high_neg'], m['mid_pos']),   # Si neg alto y pos medio -> negativo
            'R9': np.fmin(m['high_neg'], m['high_pos'])   # Si neg alto y pos alto -> neutro
        }
    
    def _aggregate_rules(self, rules: dict) -> np.ndarray:
        """Agrega los resultados de todas las reglas"""
        # Agrupa reglas por tipo de salida
        neg = np.fmax.reduce([rules['R4'], rules['R7'], rules['R8']])
        neu = np.fmax.reduce([rules['R1'], rules['R5'], rules['R9']])
        pos = np.fmax.reduce([rules['R2'], rules['R3'], rules['R6']])
        
        # Combina todas las salidas
        return np.fmax.reduce([
            np.fmin(neg, self.op_neg),
            np.fmin(neu, self.op_neu),
            np.fmin(pos, self.op_pos)
        ])
    
    @staticmethod
    def _determine_sentiment(output: float) -> str:
        """Determina el sentimiento final basado en el valor desfuzzificado"""
        if output <= 3.33:
            return 'Negativo'
        elif output <= 6.66:
            return 'Neutro'
        return 'Positivo'

def main():

    start_time = time.time()
    
    # Lectura de datos
    print("Realizando lectura de datos")
    df = pd.read_csv('test_data.csv')
    original_sentences = df['sentence'].copy()
    
    # Preprocesamiento de texto
    print("Preprocesando textos")
    preprocessor = TextPreprocessor()
    df['sentence'] = df['sentence'].progress_apply(preprocessor.clean_text)
    
    # Análisis de sentimientos
    print("Calculando puntajes de sentimiento")
    analyzer = SentimentAnalyzer()
    scores = df['sentence'].progress_apply(analyzer.get_scores)
    df['puntaje_positivo'] = scores.apply(lambda x: x.positive)
    df['puntaje_negativo'] = scores.apply(lambda x: x.negative)
    
    # Cálculo de rangos para lógica difusa
    score_ranges = {
        'min_pos': df['puntaje_positivo'].min(),
        'max_pos': df['puntaje_positivo'].max(),
        'min_neg': df['puntaje_negativo'].min(),
        'max_neg': df['puntaje_negativo'].max(),
        'mid_pos': (df['puntaje_positivo'].min() + df['puntaje_positivo'].max()) / 2,
        'mid_neg': (df['puntaje_negativo'].min() + df['puntaje_negativo'].max()) / 2
    }
    
    # Inicialización del procesador de lógica difusa
    fuzzy_processor = FuzzyLogicProcessor(score_ranges)
    
    # Preparación de estructura para resultados
    results = {
        'tweet_original': original_sentences,
        'tweet_preprocesado': df['sentence'],
        'puntaje_positivo': df['puntaje_positivo'],
        'puntaje_negativo': df['puntaje_negativo'],
        'sentimiento_calculado': [],
        'sentimiento': [],
        'tiempo_fuzz': [],
        'tiempo_defuzz': [],
        'tiempo_total': []
    }
    
    # Procesamiento de lógica difusa
    print("Realizando cálculo de lógica difusa")
    sentiment_counts = {'Positivo': 0, 'Neutro': 0, 'Negativo': 0}
    
    # Procesa cada tweet
    for _, row in tqdm(df.iterrows(), total=len(df)):
        output, sentiment, fuzz_time, defuzz_time = fuzzy_processor.process(
            SentimentScores(row['puntaje_positivo'], row['puntaje_negativo'])
        )
        
        total_time = fuzz_time + defuzz_time
        sentiment_counts[sentiment] += 1
        
        # Almacena resultados
        results['sentimiento_calculado'].append(output)
        results['sentimiento'].append(sentiment)
        results['tiempo_fuzz'].append(fuzz_time)
        results['tiempo_defuzz'].append(defuzz_time)
        results['tiempo_total'].append(total_time)
    
    # Guarda resultados en CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('datosfinales.csv', sep=';', index=False)
    
    # Cálculo y muestra de estadísticas finales
    total_time = time.time() - start_time
    total_records = len(df)
    
    print(f"Tweets positivos: {sentiment_counts['Positivo']} ({sentiment_counts['Positivo']/total_records*100:.2f}%)")
    print(f"Tweets neutros: {sentiment_counts['Neutro']} ({sentiment_counts['Neutro']/total_records*100:.2f}%)")
    print(f"Tweets negativos: {sentiment_counts['Negativo']} ({sentiment_counts['Negativo']/total_records*100:.2f}%)")
    print(f"Tiempo total de ejecución: {total_time:.4f} segundos")
    print(f"Tiempo promedio de ejecución por cada registro: {total_time/total_records:.4f} segundos")

if __name__ == "__main__":
    tqdm.pandas()  # Habilita las barras de progreso para pandas
    main()