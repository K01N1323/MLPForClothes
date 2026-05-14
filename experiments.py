import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from preprocessing import to_one_hot, normalize_data
from layers import Dense, ReLU
from network import MLP

def run_experiments():
    print("Загрузка данных для экспериментов...")
    fashion_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
    X = fashion_mnist.data.astype('float32') 
    y = fashion_mnist.target.astype('int')
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, train_size=0.7, random_state=42)
    X_train, X_test, _, _ = normalize_data(X_train_raw, X_test_raw)
    y_train = to_one_hot(y_train_raw)
    
    input_size = X_train.shape[1]
    EPOCHS = 10

    # Эксперимент 1: Learning Rate
    lrs = [0.001, 0.01, 0.1]
    plt.figure(figsize=(10, 5))
    for lr in lrs:
        print(f"Тест LR={lr}")
        model = MLP([Dense(input_size, 128), ReLU(), Dense(128, 10)])
        hist = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=64, learning_rate=lr)
        plt.plot(range(1, EPOCHS + 1), hist, label=f'LR = {lr}')
    plt.title('Влияние Learning Rate')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_experiments()
