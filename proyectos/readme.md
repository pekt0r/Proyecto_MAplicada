# Proyecto Matemática Aplicada
# Análisis de Sentimientos con Lógica Difusa
# Elaborado por: Hector Raul Falotico Diaz

## Descripción
Este proyecto implementa un sistema de análisis de sentimientos que combina un preprocesado de datos con lógica difusa para analizar los tweets. El sistema realiza las siguientes operaciones:

1. Preprocesamiento de texto
   - Limpieza de URLs, menciones y caracteres especiales
   - Tokenización
   - Eliminación de stopwords
   - Lematización

2. Análisis de sentimientos usando VADER
   - Cálculo de puntajes positivos y negativos

3. Clasificación mediante lógica difusa
   - Funciones de membresía triangulares
   - Reglas de inferencia de Mamdani
   - Desfuzzificación por método del centroide

## Requisitos Previos
- Python 3.8 o superior
- Pip (gestor de paquetes de Python)

## Instalación

bash
pip install -r requirements.txt


## Estructura de Datos
El programa espera un archivo CSV llamado `test_data.csv` con la siguiente estructura:
- Una columna llamada 'sentence' que contenga los tweets a analizar

## Uso

1. Asegúrese de que su archivo `test_data.csv` está en el directorio del proyecto

2. Ejecute el programa

3. El programa generará un archivo `datos.csv` con los resultados, que incluirá:
   - Tweet original
   - Tweet preprocesado
   - Puntajes positivos y negativos
   - Sentimiento calculado
   - Tiempos de procesamiento

## Salida
El programa mostrará en consola:
- Progreso del procesamiento
- Estadísticas finales:
  - Porcentaje de tweets positivos, neutros y negativos
  - Tiempo total de ejecución
  - Tiempo promedio por tweet

## Archivos del Proyecto
- `proyectoMAplicada.py`: Código principal
- `requirements.txt`: Lista de dependencias
- `test_data.csv`: Archivo de entrada (debe ser proporcionado por el usuario)
- `datosfinales.csv`: Archivo de salida con resultados

## Dependencias Principales
- pandas: Manipulación de datos
- nltk: Procesamiento de lenguaje natural
- scikit-fuzzy: Implementación de lógica difusa
- numpy: Operaciones numéricas
- tqdm: Barras de progreso
