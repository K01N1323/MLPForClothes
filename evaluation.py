import numpy as np

def calculate_metrics(y_true, y_pred, num_classes=10):
    f1_scores = []
    print("\n--- Отчет по классам ---")
    print(f"{'Класс':<4} | {'Precision':<9} | {'Recall':<9} | {'F1-score':<9}")
    print("-" * 40)
    
    for c in range(num_classes):
        TP = np.sum((y_pred == c) & (y_true == c))
        FP = np.sum((y_pred == c) & (y_true != c))
        FN = np.sum((y_pred != c) & (y_true == c))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
        f1_scores.append(f1)
        print(f"{c:<4} | {precision:<9.4f} | {recall:<9.4f} | {f1:<9.4f}")
        
    macro_f1 = np.mean(f1_scores)
    accuracy = np.mean(y_pred == y_true)
    print("-" * 40)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    return macro_f1
