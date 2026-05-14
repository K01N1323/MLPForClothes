import unittest
import numpy as np
from layers import Dense, ReLU
from network import MLP

class TestMLP(unittest.TestCase):
    def test_initializations(self):
        """Проверка различных стратегий инициализации весов."""
        input_size, output_size = 784, 128
        layer_he = Dense(input_size, output_size, init_type='he')
        expected_std = np.sqrt(2.0 / input_size)
        self.assertAlmostEqual(np.std(layer_he.W), expected_std, delta=0.05)
        
        layer_zero = Dense(input_size, output_size, init_type='zero')
        self.assertTrue(np.all(layer_zero.W == 0))

    def test_learning_rate_impact(self):
        """Проверка, что lr влияет на шаг."""
        input_size, output_size = 10, 5
        X = np.random.randn(5, input_size)
        y = np.eye(5)
        
        def get_weight_diff(lr):
            np.random.seed(42)
            layer = Dense(input_size, output_size)
            model = MLP([layer])
            W_before = layer.W.copy()
            model.fit(X, y, epochs=10, batch_size=5, learning_rate=lr)
            return np.abs(layer.W - W_before).sum()

        diff_small = get_weight_diff(0.001)
        diff_large = get_weight_diff(0.1)
        self.assertGreater(diff_large, diff_small)

    def test_convergence(self):
        """Тест на реальную сходимость: лосс должен упасть."""
        np.random.seed(42)
        X = np.random.randn(10, 4)
        y = np.eye(10)
        
        model = MLP([Dense(4, 16), ReLU(), Dense(16, 10)])
        history = model.fit(X, y, epochs=50, batch_size=10, learning_rate=0.1)
        
        self.assertLess(history[-1], history[0], "Loss did not decrease after 50 epochs")
        print(f"\n[OK] Сходимость подтверждена: Loss {history[0]:.4f} -> {history[-1]:.4f}")

if __name__ == '__main__':
    unittest.main()
