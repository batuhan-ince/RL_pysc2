""" AC2 implementation with n step generelized advantage estimation and
parallel environments in mind.
"""
import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical


class A2C(nn.Module):
    """ N step Actor critic with Generalized advantage estimation. Designed to
    be used with vectorized(parallel) environments. Parameter update is done
    implicitly. This means all the n-step td and gae calculations are perfomed
    inside. This requires an aditional queue in order to hold transitions. It
    is expected to push transition elements(value, reward, done, log_prob and
    entropy) into the queue at every iteration.
        Unlike some other A2C implementations we used step-by-step updating
    scheme. For example, one way to calculate losses is to collect
    transitions for n steps to perform single update and skipping
    all the n step transition as shown below.

        - - - - < - - - - - < - - -    - - - - < - - - - - < - - -
        _____ ___ _____ _____ _____    ______      l_0
          l_0 l_1  l_2   l_3   l_4       _____     l_1
                                           _____   l_2
                                             ...

        First example uses each n-step transitions only one time while the
    example at the left which we used in this implemntation updates at every
    time step while calculating n-step return and gae using queues.
        Example usage is to append transitions to the agent via
    <add_transition> method and calling update after function after the first
    n transitions are fed.
        Arguments:
            - network: Torch module that has two heads one for policy
                distribution logits and the other for state value.
            - nstep: number of steps to lookforward for td
            - optimizer: Torch optimizer that is fed by the parameters of the
                given network
    """

    class TransitionQueue:
        """ List implementation of Queue for holding batch of transition
        tensors to be used in n-step td and gae calculations. When iterating
        queue returns one less transitions that it holds since it also returns
        the value of a next state.
            Arguments:
                - capacity: Capacity of the queue. Expected to be n+1.
            Raise:
                - RuntimeError: If tried to be iterated when the size of the
                queue is less than 2.
        """

        def __init__(self, capacity):
            self.cycle = 0
            self._queue = []
            self.capacity = capacity

        @property
        def size(self):
            return len(self._queue)

        def __len__(self):
            return self.size

        def put(self, value, reward, done, log_prob, entropy):
            element = (value, reward, done, log_prob, entropy)
            if self.size != self.capacity:
                self._queue.append(element)
            else:
                self._queue[self.cycle] = element
                self.cycle = (self.cycle + 1) % self.capacity

        def __reversed__(self):
            if self.size < 1:
                raise RuntimeError("Queue has less than 2 element!")
            for i in range(self.size - 1):
                i_0 = (self.cycle - i - 2) % self.size
                i_1 = (self.cycle - i - 1) % self.size
                yield (*self._queue[i_0][:3], self._queue[i_1][0])

        def __iter__(self):
            if self.size < 1:
                raise RuntimeError("Queue has less than 2 element!")
            for i in range(self.size - 1):
                i = (self.cycle + i) % self.size
                yield (self._queue[i][0], *self._queue[i][3:])

        @property
        def last_value(self):
            return self._queue[self.cycle - 1][0]

    def __init__(self, network, nstep, optimizer):
        super(A2C, self).__init__()
        self.network = network
        self.nstep = nstep
        self.optimizer = optimizer
        # Transitions are gather inside A2C for updating. There are n_step +1
        # transitions in the queue at any time. This is because we need the
        # last next_state for TD calculations
        self.queue = self.TransitionQueue(nstep + 1)
        self.update_cycle = self.nstep

    def forward(self, state):
        """ Generate distribution of the policy. Log probability, entropy and
        action is taken from the distribution. In the training mode agent
        stores trainstions to be used in the gradient calculations. Note that
        network is expected to return logits and not the softmax output of
        logits and the returning action is an array integers.
            Arguments:
                - state: Torch tensor of observations. Expected dim (B, *F)
                    where B is the batch size and F is the feature size or
                    tuple of dims. For example; (32, 100) or (16, 3, 84, 84)
                    depending on to the network.
            Return(train mode):
                - action -> torch-int-tensor: Discrete action. Sample from
                    the distribution generated with the given state.
                    Dim: (?, 1)
                - log_prob -> torch-float-tensor: Log probability of the
                    action. Dim: (?, 1)
                - entropy -> torch-float-tensor: Entropy of the distribution.
                    Dim: (?, 1)
                - value -> torch-float-tensor: Value of the state. Dim: (?, 1)
            Return(eval mode):
                - action -> torch-int-tensor: Discrete action. Most likely
                    action. Dim: (?, 1)

        """
        if self.training:
            logit_act, value = self.network(state)
        else:
            with torch.no_grad():
                logit_act, value = self.network(state)
            return torch.argmax(logit_act, dim=-1)

        dist = Categorical(logits=logit_act)
        action = dist.sample().detach()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return (action.reshape(-1, 1), log_prob.reshape(-1, 1),
                entropy.reshape(-1, 1), value)

    def update(self, gamma, tau, beta):
        """
        Calculate actor and critic losses. Generalized advantage estimation is
        used in the actor loss. Loss is calculated for a batch of transitions.
        Necessary batch is gathered from agent's queue. Update function should
        only be called once before addition of new transition into the queue.
        Arguments:
            - gamma: Discount rate
            - tau: Generalized advantage estimation coefficient
            - beta: Entropy regularization coefficient
        Return:
            - loss
        """
        loss = 0.0
        returns, gaes = self._gae_and_return(gamma, tau)

        value_loss = 0
        policy_gain = 0
        entropy_gain = 0

        for n_return, gae, (value, log_prob, entropy) in zip(
                returns, gaes, self.queue):

            value_loss += torch.nn.functional.smooth_l1_loss(
                value.flatten(), n_return)
            policy_gain += -log_prob.flatten() * gae
            entropy_gain += -entropy.flatten() * beta

        loss += (value_loss.mean() + policy_gain.mean() + entropy_gain.mean())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return value_loss.detach(), policy_gain.detach(), entropy_gain.detach()

    def _gae_and_return(self, gamma, tau):
        """ Calculate n-step generalized advantage estimation and n step
        return over the model's queue. Transitions are kept in the queue. Each
        transition is a tensor with batch dimension. Both the td error and gae
        are calculated for n steps even if one or more episodes are terminated.
        This is because we use tensor operations instead of looping for each
        episode.
            Masks are used so that we dont include rewards or values where we
        shoudn't. Reward mask is simply masking rewards after the termination
        therefore it starts with 1.0 and after the first termination is
        observerd it vanishes. Value mask is a bit different, we want to mask
        the value until the end of the episode but if a termination occurs we
        mask it completly.
            Arguments:
                - gamma: Discount rate
                - tau: GAE constant
            Return:
                - n_return: N step return
                - gae: N step Generalized advantage estimation
        """

        last_value = self.queue.last_value
        batch_size = last_value.shape[0]
        try:
            self.running_return_mean
        except AttributeError:
            self.running_return_mean = 0
            self.running_return_var = 0

        gae_array = torch.zeros(self.nstep, batch_size, dtype=torch.float32)
        n_return_array = torch.zeros(
            self.nstep, batch_size, dtype=torch.float32)

        n_return = last_value.detach()
        gae = 0

        for i, (value, reward, done, next_value) in enumerate(
                reversed(self.queue)):
            self.running_return_mean = self.running_return_mean*0.95 \
                + 0.05*n_return
            self.running_return_var = self.running_return_var*0.95 \
                + 0.05*torch.pow(n_return - self.running_return_mean, 2)
            n_return = (1 - done)*n_return*gamma + reward
            n_return = (n_return - self.running_return_mean) \
                / (torch.sqrt(self.running_return_var) + 1e-7)

            delta = gamma*next_value*(1-done) + reward - value
            gae = (1 - done)*gae*gamma*tau + delta.detach()
            adv = n_return - value.detach()

            gae_array[-1 - i, :] = adv.flatten()
            n_return_array[-1 - i, :] = n_return.flatten()

        return n_return_array, gae_array

    def add_transition(self, value, reward, done, log_prob, entropy):
        """ Append transitions to the agent's queue. Each element is expected
        to be batch sized tensor objects.
            Raise:
                - ValueError: If the given argument is not a torch tensor
                - ValueError: If the tensor is not 2 dimensional
                - ValueError: If the dtype of a tensor is not float
        """
        for name, element in zip(
            ["value", "reward", "done", "log_prob", "entropy"],
                [value, reward, done, log_prob, entropy]):
            if not isinstance(element, torch.Tensor):
                raise ValueError(
                    "Argument {} is expected to be torch Tensor!".format(name))
            if (len(element.shape)) != 2:
                raise ValueError(
                    "Argument {} must to be 2 dimensional!".format(name))
            if not isinstance(element, (torch.FloatTensor or
                                        torch.cuda.FloatTensor)):
                raise ValueError(
                    "Argument {} must be a float tensor!".format(name))

        self.queue.put(value, reward, done, log_prob, entropy)
