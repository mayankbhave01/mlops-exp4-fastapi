# src/test_model.py

"""
Unit tests for ML model
"""

import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import train_model


class TestModel(unittest.TestCase):

    def test_model_training(self):
        accuracy = train_model()
        self.assertIsNotNone(accuracy)
        self.assertGreater(accuracy, 0.0)

    def test_model_accuracy_threshold(self):
        accuracy = train_model()
        self.assertGreater(accuracy, 0.85, "Model accuracy below threshold")

    def test_model_saved(self):
        train_model()
        self.assertTrue(os.path.exists('models/model.pkl'))


if __name__ == '__main__':
    unittest.main()
    