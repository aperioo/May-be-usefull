from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Pobranie danych MNIST
mnist = fetch_openml('mnist_784', version=1)

# Przygotowanie danych
X, y = mnist['data'], mnist['target']
X = X / 255.0  # Normalizacja
y = y.astype(int)

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trenowanie modelu KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Zapisanie modelu
joblib.dump(knn, 'knn_digit_recognition_model.pkl')
