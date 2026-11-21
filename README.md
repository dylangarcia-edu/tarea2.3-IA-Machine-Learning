# Machine Learning (Clasificación)

**Modelo:** Árbol de decisión (`DecisionTreeClassifier` de scikit-learn)

**Dataset:** Iris, con 150 flores de 3 especies diferentes y 4 medidas en cm.

**Video:** https://youtu.be/YGd3cTgK2pE

## Cómo ejecutar

Instalar las librerías y ejecutar el programa:

```bash
pip install -r requirements.txt
python arbol_decision.py
```

El programa muestra las reglas del árbol y cómo realiza una predicción. También guarda varias imágenes en la carpeta `figuras/`.

 Archivo y su descripción:

 `1_eleccion_max_depth.png`  Accuracy según la profundidad del árbol. 
 `2_arbol.png`  Árbol de decisión entrenado. 
 `3_matriz_confusion.png`  Aciertos y errores del modelo. 
 `4_importancia.png`  Importancia de las medidas. 
 `5_fronteras.png`  Separación de las especies. 

## Justificación

**Dataset:** Elegí Iris porque es pequeño, tiene datos completos y sus medidas la verdad son más fáciles de entender.

**Algoritmo:** El árbol de decisión es fácil de entender porque utiliza preguntas sobre las medidas de las flores. Además, no necesita normalizar los datos.

**Hiperparámetros:**

- `max_depth=3`: Se eligió porque obtuvo el mejor promedio en la validación cruzada.
- `criterion="gini"`: Se utiliza para separar las especies.
- `random_state=42`: Para obtener los mismos resultados al ejecutar el programa.
- División 70/30: Se usan 105 flores para entrenar y 45 para probar.

## Resultados

- Accuracy: **97.8 %** (44 de 45 flores clasificadas correctamente).
- Las medidas más importantes fueron el largo y el ancho del pétalo.
- La regla principal es que si el largo del pétalo es menor o igual a 2.45 cm, la flor es setosa.
- Las otras dos especies se separan usando las medidas del pétalo.
