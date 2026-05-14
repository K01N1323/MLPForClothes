import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    def forward(self, input_data):
        raise NotImplementedError
    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size, output_size, init_type='he'):
        super().__init__()
        if init_type == 'he':
            he_scale = np.sqrt(2.0 / input_size)
            self.W = np.random.normal(loc=0.0, scale=he_scale, size=(input_size, output_size))
        elif init_type == 'zero':
            self.W = np.zeros((input_size, output_size))
        elif init_type == 'large':
            self.W = np.random.normal(loc=0.0, scale=10.0, size=(input_size, output_size))
        else: # normal
            self.W = np.random.normal(loc=0.0, scale=0.01, size=(input_size, output_size))
            
        self.b = np.zeros((1, output_size))
        self.v_W = np.zeros_like(self.W)
        self.v_b = np.zeros_like(self.b)

    def forward(self, input_data):
        self.input = input_data
        return input_data @ self.W + self.b

    def backward(self, output_gradient, learning_rate, beta=0.9, lambda_l2=0.001, use_nesterov=False):
        self.dW = self.input.T @ output_gradient
        self.db = np.sum(output_gradient, axis=0, keepdims=True)
        
        self.dW += lambda_l2 * self.W
        
        if use_nesterov:
            v_W_prev = self.v_W.copy()
            v_b_prev = self.v_b.copy()
            
            # v = beta * v - lr * grad
            self.v_W = beta * self.v_W - learning_rate * self.dW
            self.v_b = beta * self.v_b - learning_rate * self.db
            
            # w = w - beta * v_prev + (1 + beta) * v
            self.W += -beta * v_W_prev + (1 + beta) * self.v_W
            self.b += -beta * v_b_prev + (1 + beta) * self.v_b
        else:
            self.v_W = beta * self.v_W + (1 - beta) * self.dW
            self.v_b = beta * self.v_b + (1 - beta) * self.db

            self.W -= learning_rate * self.v_W
            self.b -= learning_rate * self.v_b

        return output_gradient @ self.W.T

class ReLU(Layer):
    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * (self.input > 0)

class SoftmaxCrossEntropy(Layer):
    def __init__(self):
        super().__init__()
        self.y_true = None
        self.probs = None

    def forward(self, logits, y_true):
        self.y_true = y_true
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shifted_logits)
        self.probs = exps / (np.sum(exps, axis=1, keepdims=True) + 1e-9)
        return -np.sum(y_true * np.log(self.probs + 1e-9)) / logits.shape[0]

    def backward(self):
        return (self.probs - self.y_true) / self.y_true.shape[0]
