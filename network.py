import numpy as np
from layers import Dense, SoftmaxCrossEntropy

class MLP:
    def __init__(self, layers):
        self.layers = layers
        self.loss_layer = SoftmaxCrossEntropy()

    def forward(self, X, y_true=None):
        current_input = X
        for layer in self.layers:
            current_input = layer.forward(current_input)
        
        if y_true is not None:
            loss = self.loss_layer.forward(current_input, y_true)
            return loss, current_input
        return current_input

    def predict_probs(self, X):
        logits = self.forward(X)
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shifted_logits)
        return exps / (np.sum(exps, axis=1, keepdims=True) + 1e-9)

    def predict(self, X):
        probs = self.predict_probs(X)
        return np.argmax(probs, axis=1)

    def fit(self, X_train, y_train, epochs=100, batch_size=64, learning_rate=0.01, mu_max=0.99, use_nesterov=False):
        history = []
        n_samples = X_train.shape[0]
        global_step = 0 

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0
            for i in range(0, n_samples, batch_size):
                global_step += 1
                
                floor_val = np.floor(global_step / 250.0) + 1
                current_beta = min(1.0 - (2.0 ** (-1.0 - np.log2(floor_val))), mu_max)

                batch_idx = indices[i : i + batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]
                
                loss, _ = self.forward(X_batch, y_batch)
                epoch_loss += loss
                
                grad = self.loss_layer.backward()
                
                # Клиппинг градиентов для предотвращения взрыва
                grad = np.clip(grad, -1.0, 1.0)
                
                for layer in reversed(self.layers):
                    if isinstance(layer, Dense):
                        grad = layer.backward(grad, learning_rate, beta=current_beta, use_nesterov=use_nesterov)
                    else:
                        grad = layer.backward(grad, learning_rate)
                
            epoch_loss /= (n_samples / batch_size)
            history.append(epoch_loss)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} - Momentum: {current_beta:.4f}")
        return history
