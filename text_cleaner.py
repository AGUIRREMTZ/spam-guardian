"""
Módulo para limpieza y procesamiento de texto
"""

import re
import html
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


class TextCleaner:
    """
    Clase para limpiar y procesar texto de correos electrónicos
    """
    
    def __init__(self):
        self.stemmer = PorterStemmer()
    
    def clean_html(self, text):
        """
        Elimina etiquetas HTML y decodifica entidades HTML
        """
        # Decodificar entidades HTML
        text = html.unescape(text)
        # Eliminar etiquetas HTML
        text = re.sub(r'<[^>]+>', ' ', text)
        return text
    
    def clean_text(self, text):
        """
        Limpia el texto eliminando caracteres especiales, números, etc.
        """
        # Convertir a minúsculas
        text = text.lower()
        
        # Eliminar URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\$$\$$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        
        # Eliminar correos electrónicos
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Eliminar números
        text = re.sub(r'\d+', ' ', text)
        
        # Eliminar caracteres especiales y puntuación, mantener solo letras y espacios
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Eliminar espacios múltiples
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def stem_text(self, text):
        """
        Aplica stemming a las palabras del texto
        """
        try:
            words = word_tokenize(text)
            stemmed_words = [self.stemmer.stem(word) for word in words]
            return ' '.join(stemmed_words)
        except Exception:
            # Si falla el tokenizer, usar split simple
            words = text.split()
            stemmed_words = [self.stemmer.stem(word) for word in words]
            return ' '.join(stemmed_words)
    
    def process(self, text):
        """
        Procesa el texto aplicando todas las limpiezas
        """
        # Limpiar HTML
        text = self.clean_html(text)
        
        # Limpiar texto
        text = self.clean_text(text)
        
        # Aplicar stemming
        text = self.stem_text(text)
        
        return text
