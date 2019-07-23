import unittest
import torch

from rl_pysc2.agents.a2c import A2C


class QueueTest(unittest.TestCase):

    def test_filling(self):
        transition = tuple([torch.ones(3, 1) for i in range(5)])
        n = 2
        queue = A2C.TransitionQueue(n)

        self.assertRaises(RuntimeError, lambda: next(iter(queue)))
        queue.put(*transition)
        self.assertEqual(len(queue), 1)
        queue.put(*transition)
        self.assertEqual(len(queue), 2)
        queue.put(*transition)
        self.assertEqual(len(queue), 2)
        self.assertEqual(queue.cycle, 1)


class A2CTest(unittest.TestCase):

    class Network(torch.nn.Module):

        def __init__(self, in_s, out_s):
            super().__init__()
            self.policy = torch.nn.Linear(in_s, out_s)
            self.value = torch.nn.Linear(in_s, 1)

        def forward(self, state):
            return self.policy(state), self.value(state)

    def test_forward(self):
        network = self.Network(3, 4)
        optim = torch.optim.Adam(network.parameters())
        agent = A2C(network, 2, optim)

        state = torch.ones(12, 3)

        self.assertEqual(len(agent(state)), 4)
        agent.eval()
        self.assertFalse(isinstance(agent(state), tuple))
        agent.train()
        for returned_tensor in agent(state):
            self.assertIsInstance(returned_tensor, torch.Tensor)
            self.assertEqual(returned_tensor.shape, torch.Size([12, 1]))

    def test_gae_and_return(self):

        test_1 = dict(
            done=[0, 0, 0, 1, 0, 0],
            reward=[1, 1, 1, 10, 1, 1],
            value=[-100, -100, -100, -100, -200, -100],
            expected_r=13,
            expected_gae=113,
            n=4)
        test_2 = dict(
            done=[0, 0, 0, 0, 0],
            reward=[1, 1, 1, 1, 1],
            value=[-100, -100, -100, -100, -100],
            expected_r=-97,
            expected_gae=3,
            n=3
        )
        test_3 = dict(
            done=[1, 0, 0, 0],
            reward=[1, 1, 1, 1],
            value=[-100, -90, -80, -70],
            expected_r=1,
            expected_gae=101,
            n=3
        )
        
        def unit_test_fn(done, reward, value, expected_r, expected_gae, n):
            log_prob = torch.ones(1, 1)
            entropy = torch.ones(1, 1)

            network = self.Network(3, 4)
            optim = torch.optim.Adam(network.parameters())
            agent = A2C(network, n, optim)

            self.assertRaises(ValueError,
                              lambda: agent.add_transition(value, done, reward,
                                                           log_prob, entropy))
            self.assertRaises(ValueError,
                              lambda: agent.add_transition(torch.tensor(value),
                                                           torch.tensor(done),
                                                           torch.tensor(
                                                               reward),
                                                           log_prob, entropy))

            for d, r, v in zip(done, reward, value):
                v = torch.tensor(v).reshape(1, 1).float()
                d = torch.tensor(d).reshape(1, 1).float()
                r = torch.tensor(r).reshape(1, 1).float()
                agent.add_transition(v, r, d, log_prob, entropy)

            n_return, gae = agent._gae_and_return(1, 1)
            self.assertEqual(n_return, expected_r)
            self.assertEqual(gae, expected_gae)
        
        for test_ in (test_1, test_2, test_3):
            unit_test_fn(**test_)

    def test_update(self):
        


if __name__ == '__main__':
    unittest.main()
