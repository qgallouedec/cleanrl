import unittest
import numpy as np
import torch
from collections import deque
from types import SimpleNamespace
from cleanrl.rainbow_atari import SumTree, PrioritizedReplayBuffer

class TestSumTree(unittest.TestCase):
    def test_initialization(self):
        tree = SumTree(capacity=4)
        self.assertEqual(len(tree.tree), 8)
        self.assertEqual(tree.max_priority, 1.0)

    def test_update_and_total(self):
        tree = SumTree(capacity=4)
        tree.update(0, 5.0)
        tree.update(1, 3.0)
        self.assertEqual(tree.total(), 8.0)

    def test_get(self):
        tree = SumTree(capacity=4)
        tree.update(0, 1.0)
        tree.update(1, 3.0)
        tree.update(2, 4.0)
        tree.update(3, 2.0)
        # Check if get method returns the right index based on the segment value
        self.assertEqual(tree.get(0.5), 0)
        self.assertEqual(tree.get(3.5), 1)
        self.assertEqual(tree.get(7.5), 2)
        self.assertEqual(tree.get(9.5), 3)

class TestPrioritizedReplayBuffer(unittest.TestCase):

    def setUp(self):
        self.buffer = PrioritizedReplayBuffer(size=4, n_step=2, device="cpu")

    def test_add_and_sample(self):
        for i in range(4):
            self.buffer.add(np.zeros((1, 4, 84, 84), dtype=np.uint8),
                            np.zeros((1, 4, 84, 84), dtype=np.uint8),
                            np.zeros((1, 1), dtype=np.int64),
                            np.zeros((1, 1), dtype=np.float32),
                            np.zeros((1, 1), dtype=bool), {})
        sample, idxs, weights = self.buffer.sample(2)
        self.assertEqual(len(sample.observations), 2)
        self.assertEqual(len(idxs), 2)
        self.assertEqual(len(weights), 2)

    def test_update_priorities(self):
        for i in range(2):
            self.buffer.add(np.zeros((1, 4, 84, 84), dtype=np.uint8),
                            np.zeros((1, 4, 84, 84), dtype=np.uint8),
                            np.zeros((1, 1), dtype=np.int64),
                            np.zeros((1, 1), dtype=np.float32),
                            np.zeros((1, 1), dtype=bool), {})
        idxs = [0]
        projected_dists = torch.zeros(1, 51)
        target_dists = torch.zeros(1, 51)
        self.buffer.update_priorities(idxs, projected_dists, target_dists)
        _, _, weights = self.buffer.sample(1)
        self.assertEqual(weights[0], 1.0)

    def test_update_beta(self):
        self.buffer.update_beta(0.5)
        self.assertAlmostEqual(self.buffer.beta, 0.7)  # 0.7 = (1.0 - 0.4) * 0.5 + 0.4

if __name__ == '__main__':
    # unittest.main()
    
    # Just test update_priorities
    buffer = PrioritizedReplayBuffer(size=4, n_step=2, device="cpu")
    for i in range(2):
        buffer.add(np.zeros((1, 4, 84, 84), dtype=np.uint8),
                    np.zeros((1, 4, 84, 84), dtype=np.uint8),
                    np.zeros((1, 1), dtype=np.int64),
                    np.zeros((1, 1), dtype=np.float32),
                    np.zeros((1, 1), dtype=bool), {})
    idxs = [0]
    projected_dists = torch.zeros(1, 51)
    target_dists = torch.zeros(1, 51)
    buffer.update_priorities(idxs, projected_dists, target_dists)
    _, _, weights = buffer.sample(1)
    print(weights[0])
    
