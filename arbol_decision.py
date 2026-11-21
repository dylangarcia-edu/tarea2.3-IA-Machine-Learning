"""
Tarea: Machine Learning (Clasificación)
Modelo: Árbol de decisión usando el dataset Iris.

Para ejecutar: python arbol_decision.py
Guarda las gráficas en la carpeta figuras/ y muestra en consola
las reglas del árbol y un ejemplo de cómo hace una predicción.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

CARPETA = "figuras"
os.makedirs(CARPETA, exist_ok=True)
SEMILLA = 42

# ------------------------------------------------------------
# 1. Cargar los datos
# ------------------------------------------------------------
iris = load_iris()
X, y = iris.data, iris.target
caracteristicas = ["largo_sepalo", "ancho_sepalo", "largo_petalo", "ancho_petalo"]
clases = [str(c) for c in iris.target_names]  # setosa, versicolor, virginica

print("=" * 60)
print("DATASET IRIS")
print("=" * 60)
print(f"Flores: {X.shape[0]}  |  Características: {X.shape[1]}  |  Clases: {clases}")

# Separar los datos: 70% para entrenar y 30% para probar
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEMILLA, stratify=y
)
print(f"Entrenamiento: {len(X_train)} flores  |  Prueba: {len(X_test)} flores")

# ------------------------------------------------------------
# 2. Probar diferentes profundidades del árbol
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PRUEBA DE max_depth (validación cruzada de 5 partes)")
print("=" * 60)
profundidades = range(1, 9)
medias = []
for d in profundidades:
    m = DecisionTreeClassifier(max_depth=d, random_state=SEMILLA)
    puntajes = cross_val_score(m, X_train, y_train, cv=5)
    medias.append(puntajes.mean())
    print(f"max_depth={d}:  accuracy promedio = {puntajes.mean():.3f}")

plt.figure(figsize=(7, 4))
plt.plot(list(profundidades), medias, marker="o")
plt.axvline(3, color="red", linestyle="--", label="elegido: max_depth=3")
plt.xlabel("max_depth (profundidad máxima del árbol)")
plt.ylabel("Accuracy promedio (validación cruzada)")
plt.title("¿Qué tan profundo debe ser el árbol?")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{CARPETA}/1_eleccion_max_depth.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# 3. Entrenar el árbol
# ------------------------------------------------------------
modelo = DecisionTreeClassifier(
    criterion="gini", max_depth=3, random_state=SEMILLA
)
modelo.fit(X_train, y_train)

print("\n" + "=" * 60)
print("REGLAS DEL ÁRBOL")
print("=" * 60)
print(export_text(modelo, feature_names=caracteristicas, class_names=clases))

plt.figure(figsize=(16, 9))
plot_tree(
    modelo,
    feature_names=caracteristicas,
    class_names=clases,
    filled=True,
    rounded=True,
    fontsize=11,
)
plt.title("Árbol de decisión entrenado (max_depth=3)", fontsize=16)
plt.tight_layout()
plt.savefig(f"{CARPETA}/2_arbol.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# 4. Ver cómo hace una predicción
# ------------------------------------------------------------
def explicar_prediccion(modelo, muestra, nombre="flor"):
    arbol = modelo.tree_
    camino = modelo.decision_path(muestra.reshape(1, -1)).indices
    print(f"\nValores de la {nombre}:")
    for n, v in zip(caracteristicas, muestra):
        print(f"   {n:13s} = {v:.1f} cm")
    print("\nRecorrido por el árbol:")
    for paso, nodo in enumerate(camino, start=1):
        if arbol.children_left[nodo] == arbol.children_right[nodo]:  # hoja
            conteo = arbol.value[nodo][0]
            pred = clases[int(np.argmax(conteo))]
            print(f"   Paso {paso}: llegué a una HOJA -> predicción: {pred.upper()}")
            break
        f = arbol.feature[nodo]
        umbral = arbol.threshold[nodo]
        valor = muestra[f]
        va_izq = valor <= umbral
        print(
            f"   Paso {paso}: ¿{caracteristicas[f]} <= {umbral:.2f}?  "
            f"valor = {valor:.1f}  ->  {'SÍ, voy a la izquierda' if va_izq else 'NO, voy a la derecha'}"
        )


print("=" * 60)
print("EJEMPLO DE PREDICCIÓN PARA UNA FLOR")
print("=" * 60)
# Tomar una flor versicolor de los datos de prueba
idx = int(np.where(y_test == 1)[0][0])
explicar_prediccion(modelo, X_test[idx], "flor de prueba")
print(f"   Clase real: {clases[y_test[idx]].upper()}")

# ------------------------------------------------------------
# 5. Revisar qué tan bien funciona el modelo
# ------------------------------------------------------------
pred = modelo.predict(X_test)
acc = accuracy_score(y_test, pred)
print("\n" + "=" * 60)
print("RESULTADOS DE LA PRUEBA")
print("=" * 60)
print(f"Accuracy: {acc:.3f}  ({int(acc * len(y_test))} de {len(y_test)} correctas)")

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, pred, display_labels=clases, cmap="Blues", ax=ax
)
ax.set_title(f"Matriz de confusión (accuracy = {acc:.2%})")
ax.set_xlabel("Predicción del modelo")
ax.set_ylabel("Clase real")
plt.tight_layout()
plt.savefig(f"{CARPETA}/3_matriz_confusion.png", dpi=150)
plt.close()

# Ver la importancia de cada medida
importancias = modelo.feature_importances_
print("\nImportancia de cada característica:")
for n, imp in sorted(zip(caracteristicas, importancias), key=lambda t: -t[1]):
    print(f"   {n:13s}: {imp:.2f}")

plt.figure(figsize=(7, 4))
orden = np.argsort(importancias)
plt.barh(np.array(caracteristicas)[orden], importancias[orden], color="seagreen")
plt.xlabel("Importancia (cuánto ayuda a separar las clases)")
plt.title("¿Qué medidas usa más el árbol?")
plt.tight_layout()
plt.savefig(f"{CARPETA}/4_importancia.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# 6. Mostrar cómo se separan las especies con las medidas del pétalo
# ------------------------------------------------------------
X2 = X[:, 2:4]
m2 = DecisionTreeClassifier(max_depth=3, random_state=SEMILLA).fit(X2, y)
xx, yy = np.meshgrid(
    np.linspace(X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5, 400),
    np.linspace(X2[:, 1].min() - 0.3, X2[:, 1].max() + 0.3, 400),
)
zz = m2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, zz, alpha=0.25, cmap="viridis")
colores = ["#440154", "#21918c", "#fde725"]
for i, c in enumerate(clases):
    plt.scatter(X2[y == i, 0], X2[y == i, 1], label=c, edgecolor="k", color=colores[i])
plt.xlabel("largo_petalo (cm)")
plt.ylabel("ancho_petalo (cm)")
plt.title("Separación de las especies según las medidas del pétalo")
plt.legend()
plt.tight_layout()
plt.savefig(f"{CARPETA}/5_fronteras.png", dpi=150)
plt.close()

print(f"\nFiguras guardadas en la carpeta '{CARPETA}/'")
