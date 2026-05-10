#импорт нужных библиотек
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
#фиксированный рандом для воспроизводимости жксперимента 
#загрузка датасета 
fashion_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
# делим признаки и примеры
X = fashion_mnist.data.astype('float32') 
y = fashion_mnist.target.astype('int')
# приводим к виду 0 0 0 1 0 0 ....
def to_one_hot(y, num_classes = 10):
    oh = np.zeros((y.size, num_classes))
    oh[np.arange(y.size), y] = 1
    return oh
#делим выборки
X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y, train_size=0.7, random_state=42, stratify=y)
# Z-score
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis = 0) + 1e-7

X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

y_train = to_one_hot(y_train_raw)
y_test = to_one_hot(y_test_raw)
#базовый интефейс
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    def forward(self, input_data):
        raise NotImplementedError("Функция должна быть переопределена в наследнике")
    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError("Функция должна быть переопределена в наследнике")
# слой 
class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()

        he_scale = np.sqrt(2.0 / input_size)
        self.W = np.random.normal(loc=0.0, scale=he_scale, size=(input_size, output_size))
        self.b = np.zeros((1, output_size))
        
        self.v_W = np.zeros_like(self.W)
        self.v_b = np.zeros_like(self.b)

    def forward(self, input_data):
        self.input = input_data
        return input_data @ self.W + self.b

    #  beta (для Momentum) и lambda_l2 (для регуляризации)
    def backward(self, output_gradient, learning_rate, beta=0.9, lambda_l2=0.001):
        self.dW = self.input.T @ output_gradient
        self.db = np.sum(output_gradient, axis=0, keepdims=True)

        # L2 РЕГУЛЯРИЗАЦИЯ
        # Штрафуем большие веса (добавляем их к градиенту)
        self.dW += lambda_l2 * self.W

        # MOMENTUM SGD
        # Обновляем скорости 
        self.v_W = beta * self.v_W + (1 - beta) * self.dW
        self.v_b = beta * self.v_b + (1 - beta) * self.db

        # Шаг оптимизатора делается с учетом скорости, а не чистого градиента
        self.W -= learning_rate * self.v_W
        self.b -= learning_rate * self.v_b

        self.dX = output_gradient @ self.W.T

        return self.dX
# релу
class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        self.input = input_data

        return np.maximum(0, input_data)

    def backward(self, output_gradient, learning_rate):
        self.dX = output_gradient * (self.input > 0)
        
        return  self.dX

class SoftmaxCrossEntropy(Layer):
    def __init__(self):
        super().__init__()

        self.y_true = None

    def forward(self, logits, y_true):
        self.y_true = y_true
        # выитаем большее из примера чтобы не было e**100000
        shifted_logits = logits - np.max(logits, axis = 1, keepdims = True)
        # считаем экспоненты
        exps = np.exp(shifted_logits)
        # считаем вероятности
        self.probs = exps / np.sum(exps, axis = 1, keepdims = True)
        #считаем функцию потерь
        loss = -np.sum(y_true * np.log(self.probs + 1e-9)) / logits.shape[0]

        return loss

    def backward(self):
        self.dX = (self.probs - self.y_true) / self.y_true.shape[0]

        return self.dX

class MLP:
    def __init__(self, layers):
        self.layers = layers
        self.loss_layer = SoftmaxCrossEntropy()

    def forward(self, X, y_true):
        current_input = X
        # пропускаем через все слои 
        for layer in self.layers:
            current_input = layer.forward(current_input)
        # пропускаем черещ softmax
        loss = self.loss_layer.forward(current_input, y_true)

        return loss
    def backward(self, learning_rate):
        grad = self.loss_layer.backward()

        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)


    def fit(self, X_train, y_train, epochs = 100, batch_size = 64, learning_rate = 0.01, rgen = 10):
        # Сохраняем историю ошибки для построения графиков
        history = []
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)

            epoch_loss = 0
        
            # 3. Цикл по мини-батчам
            for i in range(0, X_train.shape[0], batch_size):
                batch_idx = indices[i : i + batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]
                
                epoch_loss += self.forward(X_batch, y_batch)
                self.backward(learning_rate)
                
            
            # Считаем средний loss за эпоху и сохраняем
            epoch_loss /= (X_train.shape[0] / batch_size)
            history.append(epoch_loss)
            print(f"Эпоха {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}")
            
        return history

    def predict(self, X):
        current_input = X
        # пропускаем через все слои 
        for layer in self.layers:
            current_input = layer.forward(current_input)
        
        predictions = np.argmax(current_input, axis = 1)

        return predictions

import matplotlib.pyplot as plt

input_size = X_train.shape[1] 

layers = [
    Dense(input_size, 128),
    ReLU(),
    Dense(128, 10)
]


model = MLP(layers)

print(" обучение...")
history = model.fit(X_train, y_train, epochs=100, batch_size=64, learning_rate=0.01)

# 4. Строим график функции потерь
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(history) + 1), history, marker='o', color='b', label='Training Loss')
plt.title('График падения ошибки (Loss) по эпохам')
plt.xlabel('Эпоха')
plt.ylabel('Loss (Cross-Entropy)')
plt.grid(True)
plt.legend()
plt.show()

# 5. Проверяем точность на тестовой выборке
predictions = model.predict(X_test)

accuracy = np.mean(predictions == y_test_raw) 
print(f"Точность на тестовой выборке (Accuracy): {accuracy * 100:.2f}%")


import matplotlib.pyplot as plt


# АВТОМАТИЗИРОВАННЫЕ ТЕСТЫ (ЭКСПЕРИМЕНТЫ ДЛЯ ЛАБОРАТОРНОЙ)


input_size = X_train.shape[1]
EPOCHS = 50  

print("\n" + "="*50)
print("ЭКСПЕРИМЕНТ 1: Влияние скорости обучения (Learning Rate)")
print("="*50)

lrs_to_test = [0.001, 0.01, 0.1] 
history_lr = {}

for lr in lrs_to_test:
    print(f"\n---> Тест LR = {lr}")
    # Важно: каждый раз создаем новые слои, чтобы сбросить веса к начальным значениям!
    layers = [Dense(input_size, 128), ReLU(), Dense(128, 10)]
    model = MLP(layers)
    history_lr[lr] = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=64, learning_rate=lr)

# График 1
plt.figure(figsize=(10, 5))
for lr, hist in history_lr.items():
    plt.plot(range(1, EPOCHS + 1), hist, marker='o', label=f'LR = {lr}')
plt.title('Влияние скорости обучения (Learning Rate)')
plt.xlabel('Эпоха')
plt.ylabel('Loss (Cross-Entropy)')
plt.legend()
plt.grid(True)
plt.show()  


print("\n" + "="*50)
print("ЭКСПЕРИМЕНТ 2: Влияние размера мини-батча")
print("="*50)

# Батч 1 (Stochastic) будет считаться около 15 минут, поэтому берем 16
batches_to_test = [16, 64, 256] 
history_batch = {}

for bs in batches_to_test:
    print(f"\n---> Тест Batch Size = {bs}")
    layers = [Dense(input_size, 128), ReLU(), Dense(128, 10)]
    model = MLP(layers)
    history_batch[bs] = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=bs, learning_rate=0.01)

# График 2
plt.figure(figsize=(10, 5))
for bs, hist in history_batch.items():
    plt.plot(range(1, EPOCHS + 1), hist, marker='s', label=f'Batch Size = {bs}')
plt.title('Влияние размера батча (Batch Size)')
plt.xlabel('Эпоха')
plt.ylabel('Loss (Cross-Entropy)')
plt.legend()
plt.grid(True)
plt.show()


print("\n" + "="*50)
print("ЭКСПЕРИМЕНТ 3: Влияние инициализации весов")
print("="*50)

init_types = ['Zeros (Нули)', 'Normal (Стандартная 0.01)', 'Large (Большие 10.0)']
history_init = {}

for init_type in init_types:
    print(f"\n---> Тест инициализации: {init_type}")
    layers = [Dense(input_size, 128), ReLU(), Dense(128, 10)]
    
    # Искусственно ломаем веса для тестов (без изменения кода самого класса)
    if init_type == 'Zeros (Нули)':
        layers[0].W = np.zeros((input_size, 128))
        layers[2].W = np.zeros((128, 10))
    elif init_type == 'Large (Большие 10.0)':
        layers[0].W = np.random.normal(loc=0.0, scale=10.0, size=(input_size, 128))
        layers[2].W = np.random.normal(loc=0.0, scale=10.0, size=(128, 10))
    
    model = MLP(layers)
    history_init[init_type] = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=64, learning_rate=0.01)

# График 3
plt.figure(figsize=(10, 5))
for init_type, hist in history_init.items():
    plt.plot(range(1, EPOCHS + 1), hist, marker='^', label=f'Init: {init_type}')
plt.title('Влияние инициализации весов')
plt.xlabel('Эпоха')
plt.ylabel('Loss (Cross-Entropy)')
plt.legend()
plt.grid(True)
plt.show()


def calculate_metrics(y_true, y_pred, num_classes=10):
    f1_scores = []
    
    print("\n--- Отчет по классам ---")
    print(f"{'Класс':<4} | {'Precision':<9} | {'Recall':<9} | {'F1-score':<9}")
    print("-" * 40)
    
    for c in range(num_classes):
        # Истинные положительные (Угадали класс c)
        TP = np.sum((y_pred == c) & (y_true == c))
        # Ложные положительные (Назвали классом c, а это другое)
        FP = np.sum((y_pred == c) & (y_true != c))
        # Ложные отрицательные (Это класс c, но мы назвали его иначе)
        FN = np.sum((y_pred != c) & (y_true == c))

        # Считаем Precision и Recall с защитой от деления на ноль
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        # Считаем F1
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
            
        f1_scores.append(f1)
        print(f"{c:<4} | {precision:<9.4f} | {recall:<9.4f} | {f1:<9.4f}")
        
    macro_f1 = np.mean(f1_scores)
    print("-" * 40)
    print(f"Macro F1-score: {macro_f1:.4f}")
    return macro_f1



def plot_mistakes(X_test_raw, y_true, y_pred):
    # Названия классов для Fashion-MNIST (чтобы выводить слова, а не цифры)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Находим индексы всех ошибок
    errors = np.where(y_pred != y_true)[0]
    
    print(f"\nВсего ошибок сети: {len(errors)} из {len(y_true)}")
    
    if len(errors) == 0:
        print("Ошибок нет! Сеть идеальна.")
        return

    # Выбираем 5 случайных ошибок для отображения
    num_images = min(5, len(errors))
    random_errors = np.random.choice(errors, num_images, replace=False)

    plt.figure(figsize=(15, 4))
    for i, err_idx in enumerate(random_errors):
        plt.subplot(1, num_images, i + 1)
        # Возвращаем картинку в размер 28x28 (берем ДО Z-score нормализации)
        img = X_test_raw[err_idx].reshape(28, 28) 
        
        plt.imshow(img, cmap='gray')
        plt.title(f"Предсказали:\n{class_names[y_pred[err_idx]]}\n\nНа самом деле:\n{class_names[y_true[err_idx]]}", 
                  color='red', fontsize=10)
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()


calculate_metrics(y_test_raw, predictions)

plot_mistakes(X[len(X_train):], y_test_raw, predictions)