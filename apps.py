from django.apps import AppConfig
import joblib
import os
from pathlib import Path
import nltk


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'
    
    # Variables globales para los modelos
    modelo = None
    vectorizador = None
    
    def ready(self):
        """
        Cargar los modelos de ML al iniciar la aplicación
        """
        if os.environ.get('RUN_MAIN') != 'true':
            return
            
        try:
            # Descargar recursos de nltk necesarios
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            # Ruta a la carpeta static
            base_dir = Path(__file__).resolve().parent.parent
            static_dir = base_dir / 'static'
            
            # Cargar modelo y vectorizador
            modelo_path = static_dir / 'modelo_spam.joblib'
            vectorizador_path = static_dir / 'vectorizador.joblib'
            
            if modelo_path.exists() and vectorizador_path.exists():
                ApiConfig.modelo = joblib.load(modelo_path)
                ApiConfig.vectorizador = joblib.load(vectorizador_path)
                print("✓ Modelos cargados exitosamente")
            else:
                print("⚠ Advertencia: No se encontraron los archivos de modelo en static/")
                print(f"  Esperando: {modelo_path}")
                print(f"  Esperando: {vectorizador_path}")
        except Exception as e:
            print(f"✗ Error cargando modelos: {str(e)}")
