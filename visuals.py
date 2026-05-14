import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def plot_loss(history):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history) + 1), history, marker='o', color='b', label='Training Loss')
    plt.title('График падения ошибки (Loss)')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_roc_curve(model, X_test, y_test_raw, num_classes=10):
    y_score = model.predict_probs(X_test)
    y_test_bin = label_binarize(y_test_raw, classes=range(num_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая для многоклассовой классификации')
    plt.legend(loc="lower right")
    plt.show()

def plot_decision_boundary(model, X_test, y_test, mean, std):
    # Уменьшаем размерность до 2D через PCA для визуализации
    pca = PCA(n_components=2)
    X_test_2d = pca.fit_transform(X_test)
    
    h = .05  # шаг сетки
    x_min, x_max = X_test_2d[:, 0].min() - 1, X_test_2d[:, 0].max() + 1
    y_min, y_max = X_test_2d[:, 1].min() - 1, X_test_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Восстанавливаем из 2D в 784D для предсказания
    grid_points_2d = np.c_[xx.ravel(), yy.ravel()]
    grid_points_784d = pca.inverse_transform(grid_points_2d)
    
    # Предсказание
    Z = model.predict(grid_points_784d)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='tab10')
    plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, s=20, edgecolor='k', cmap='tab10', alpha=0.6)
    plt.title("Разделяющие поверхности (PCA 2D проекция)")
    plt.xlabel("Главная компонента 1")
    plt.ylabel("Главная компонента 2")
    plt.show()

def plot_mistakes(X_test_raw, y_true, y_pred):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    errors = np.where(y_pred != y_true)[0]
    num_images = min(5, len(errors))
    random_errors = np.random.choice(errors, num_images, replace=False)

    plt.figure(figsize=(15, 4))
    for i, err_idx in enumerate(random_errors):
        plt.subplot(1, num_images, i + 1)
        img = X_test_raw[err_idx].reshape(28, 28) 
        plt.imshow(img, cmap='gray')
        plt.title(f"Pred: {class_names[y_pred[err_idx]]}\nTrue: {class_names[y_true[err_idx]]}", color='red')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
