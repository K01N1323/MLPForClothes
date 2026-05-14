import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from preprocessing import to_one_hot, normalize_data
from layers import Dense, ReLU
from network import MLP
from visuals import plot_loss, plot_roc_curve, plot_decision_boundary, plot_mistakes
from evaluation import calculate_metrics

def main():
    print("Загрузка данных Fashion-MNIST...")
    fashion_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
    X = fashion_mnist.data.astype('float32') 
    y = fashion_mnist.target.astype('int')

    # Делим выборки
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, train_size=0.7, random_state=42, stratify=y
    )

    # Нормализация
    X_train, X_test, mean, std = normalize_data(X_train_raw, X_test_raw)
    y_train = to_one_hot(y_train_raw)

    # Архитектура
    input_size = X_train.shape[1] 
    layers = [
        Dense(input_size, 128),
        ReLU(),
        Dense(128, 10)
    ]

    model = MLP(layers)

    print("Начало обучения...")
    history = model.fit(X_train, y_train, epochs=20, batch_size=64, learning_rate=0.01)

    # Визуализация
    plot_loss(history)
    
    print("Оценка модели...")
    predictions = model.predict(X_test)
    calculate_metrics(y_test_raw, predictions)
    
    print("Построение ROC-кривой...")
    plot_roc_curve(model, X_test, y_test_raw)
    
    print("Построение разделяющих поверхностей (это может занять время)...")
    # Для ускорения отрисовки используем подмножество теста
    plot_decision_boundary(model, X_test[:500], y_test_raw[:500], mean, std)
    
    print("Примеры ошибок:")
    plot_mistakes(X_test_raw, y_test_raw, predictions)

if __name__ == "__main__":
    main()
