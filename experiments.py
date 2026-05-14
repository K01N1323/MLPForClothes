import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from preprocessing import to_one_hot, normalize_data
from layers import Dense, ReLU
from network import MLP

def run_experiments():
    print("\n>>> Запуск сравнительных экспериментов (с графиками на экран) <<<")
    fashion_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
    X = fashion_mnist.data.astype('float32') 
    y = fashion_mnist.target.astype('int')
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, train_size=0.7, random_state=42)
    X_train, X_test, _, _ = normalize_data(X_train_raw, X_test_raw)
    y_train = to_one_hot(y_train_raw)
    
    input_size = X_train.shape[1]
    EPOCHS = 50 

    # 1. Learning Rate
    print("Тестирование Learning Rate...")
    plt.figure(figsize=(10, 6))
    for lr in [0.001, 0.01, 0.1]:
        model = MLP([Dense(input_size, 128), ReLU(), Dense(128, 10)])
        hist = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=64, learning_rate=lr)
        plt.plot(hist, label=f'LR = {lr}')
    plt.title('Влияние Learning Rate (50 эпох)')
    plt.xlabel('Эпоха'), plt.ylabel('Loss'), plt.legend(), plt.grid(True)
    plt.show() # ВЫСВЕТИТСЯ НА ЭКРАНЕ

    # 2. Batch Size
    print("Тестирование Batch Size...")
    plt.figure(figsize=(10, 6))
    for b in [16, 64, 256]:
        model = MLP([Dense(input_size, 128), ReLU(), Dense(128, 10)])
        hist = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=b, learning_rate=0.01)
        plt.plot(hist, label=f'Batch Size = {b}')
    plt.title('Влияние Batch Size (50 эпох)')
    plt.xlabel('Эпоха'), plt.ylabel('Loss'), plt.legend(), plt.grid(True)
    plt.show()

    # 3. Инициализация
    print("Тестирование Инициализации...")
    plt.figure(figsize=(10, 6))
    for init in ['zero', 'he', 'large']:
        model = MLP([Dense(input_size, 128, init_type=init), ReLU(), Dense(128, 10, init_type=init)])
        hist = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=64, learning_rate=0.01)
        plt.plot(hist, label=f'Init = {init}')
    plt.title('Влияние типа инициализации')
    plt.xlabel('Эпоха'), plt.ylabel('Loss'), plt.legend(), plt.grid(True)
    plt.show()

    # 4. Momentum vs Nesterov
    print("Тестирование Momentum vs Nesterov...")
    plt.figure(figsize=(10, 6))
    for nest in [False, True]:
        label = 'Nesterov' if nest else 'Standard'
        model = MLP([Dense(input_size, 128), ReLU(), Dense(128, 10)])
        hist = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=64, learning_rate=0.01, use_nesterov=nest)
        plt.plot(hist, label=label)
    plt.title('Standard vs Nesterov Momentum')
    plt.xlabel('Эпоха'), plt.ylabel('Loss'), plt.legend(), plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_experiments()
