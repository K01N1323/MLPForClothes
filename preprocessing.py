import numpy as np

def to_one_hot(y, num_classes=10):
    oh = np.zeros((y.size, num_classes))
    oh[np.arange(y.size), y] = 1
    return oh

def normalize_data(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # Добавляем epsilon и защищаем от деления на слишком малые числа (константные признаки)
    eps = 1e-8
    std[std < eps] = 1.0
    
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    
    return X_train_norm, X_test_norm, mean, std
