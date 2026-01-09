"""
Vistas de la API para análisis de spam
"""

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import numpy as np
from .apps import ApiConfig
from .text_cleaner import TextCleaner


@csrf_exempt
@require_http_methods(["POST"])
def analizar_correo(request):
    """
    Endpoint para analizar si un correo es spam o no
    
    POST /api/analizar/
    Body: { "texto": "contenido del correo" }
    
    Response: {
        "resultado": "Spam" | "Ham",
        "confianza": 85.5,
        "top_palabras": ["palabra1", "palabra2", ...]
    }
    """
    
    try:
        # Validar que los modelos estén cargados
        if ApiConfig.modelo is None or ApiConfig.vectorizador is None:
            return JsonResponse({
                'error': 'Los modelos no están disponibles. Asegúrate de cargar modelo_spam.joblib y vectorizador.joblib en la carpeta static/'
            }, status=500)
        
        # Parsear el body JSON
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({
                'error': 'JSON inválido en el body de la petición'
            }, status=400)
        
        # Validar que venga el campo texto
        texto = data.get('texto', '').strip()
        if not texto:
            return JsonResponse({
                'error': 'El campo "texto" es requerido y no puede estar vacío'
            }, status=400)
        
        # Procesar el texto
        cleaner = TextCleaner()
        texto_procesado = cleaner.process(texto)
        
        # Vectorizar el texto
        texto_vectorizado = ApiConfig.vectorizador.transform([texto_procesado])
        
        # Predecir
        prediccion = ApiConfig.modelo.predict(texto_vectorizado)[0]
        probabilidades = ApiConfig.modelo.predict_proba(texto_vectorizado)[0]
        
        # Determinar resultado y confianza
        resultado = "Spam" if prediccion == 1 else "Ham"
        confianza = float(max(probabilidades) * 100)
        
        # Obtener las palabras más influyentes
        top_palabras = obtener_top_palabras(
            texto_vectorizado,
            ApiConfig.vectorizador,
            ApiConfig.modelo,
            n=5
        )
        
        return JsonResponse({
            'resultado': resultado,
            'confianza': round(confianza, 2),
            'top_palabras': top_palabras,
            'texto_procesado': texto_procesado[:100] + '...' if len(texto_procesado) > 100 else texto_procesado
        })
        
    except Exception as e:
        return JsonResponse({
            'error': f'Error procesando la solicitud: {str(e)}'
        }, status=500)


def obtener_top_palabras(texto_vectorizado, vectorizador, modelo, n=5):
    """
    Obtiene las palabras que más influyeron en la predicción
    """
    try:
        # Obtener coeficientes del modelo
        if hasattr(modelo, 'coef_'):
            coeficientes = modelo.coef_[0]
        else:
            # Si el modelo no tiene coeficientes (ej: Random Forest),
            # usar feature_importances_ si está disponible
            if hasattr(modelo, 'feature_importances_'):
                coeficientes = modelo.feature_importances_
            else:
                return []
        
        # Obtener índices de features no ceros en el texto
        indices_no_ceros = texto_vectorizado.nonzero()[1]
        
        if len(indices_no_ceros) == 0:
            return []
        
        # Calcular scores para cada palabra presente
        feature_names = vectorizador.get_feature_names_out()
        scores = []
        
        for idx in indices_no_ceros:
            palabra = feature_names[idx]
            score = abs(coeficientes[idx]) * texto_vectorizado[0, idx]
            scores.append((palabra, score))
        
        # Ordenar por score y tomar las top N
        scores.sort(key=lambda x: x[1], reverse=True)
        top_palabras = [palabra for palabra, _ in scores[:n]]
        
        return top_palabras
        
    except Exception as e:
        print(f"Error obteniendo top palabras: {str(e)}")
        return []


@require_http_methods(["GET"])
def health_check(request):
    """
    Endpoint para verificar el estado del servicio
    """
    modelos_cargados = ApiConfig.modelo is not None and ApiConfig.vectorizador is not None
    
    return JsonResponse({
        'status': 'ok' if modelos_cargados else 'warning',
        'modelos_cargados': modelos_cargados,
        'mensaje': 'API funcionando correctamente' if modelos_cargados else 'API activa pero modelos no cargados'
    })
