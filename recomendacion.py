# recomendacion.py
# Sistema de Recomendación utilizando GitHub Copilot - INACAP - Isabel Suazo

# Importar las librerías necesarias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# 1. Cargar los datos (con respaldo de datos de prueba)
try:
    data = pd.read_csv('producto.csv')
except FileNotFoundError:
    # Datos de demostración en caso de que no exista 'producto.csv'
    raw_data = {
        'feature1': [1.0, 2.0, 1.5, 8.0, 9.0, 8.5],
        'feature2': [2.0, 1.0, 1.8, 8.0, 9.5, 9.0],
        'feature3': [3.0, 2.5, 2.9, 9.0, 8.0, 8.7],
        'label': ['Categoría A', 'Categoría A', 'Categoría A', 'Categoría B', 'Categoría B', 'Categoría B']
    }
    data = pd.DataFrame(raw_data)

# 2. Preprocesamiento de datos
features = data[['feature1', 'feature2', 'feature3']]
labels = data['label']

# 3. Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# 4. Crear y entrenar el modelo (KNN)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 5. Realizar predicciones y evaluar el modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# 6. Función de recomendación (evitando la advertencia de feature names)
def recommend(product_features):
    features_df = pd.DataFrame([product_features], columns=['feature1', 'feature2', 'feature3'])
    prediction = model.predict(features_df)
    return prediction

# Ejemplo de uso de la función de recomendación
example_product = [1.0, 2.0, 3.0]
recommended_product = recommend(example_product)
print(f'Recommended Product: {recommended_product[0]}')

print("Datos leídos por el programa:\n", data)
