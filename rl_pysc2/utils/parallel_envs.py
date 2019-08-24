""" Run multiple environments in parallel.

    Wrap your environment with ParallelEnv to run in parallel. Wrapped
    parallel environment provides step and reset functions. After an initial
    reset call there is no need for aditional reset calls. If one of the
    environment terminates after step call it automatically resets the
    environment and sends initial observation as next state. Thereby, users
    of this wrapper must be AWARE of the fact that in case of termination
    next state is the initial observation of the new episode. This is a
    dangerous behaviour but makes everything smooth so be aware!
"""
from torch.multiprocessing import Process, Pipe
import torch
from collections import namedtuple
import numpy as np


class ParallelEnv():
    """ Synchronize multi envirnments wrapper.

        Workers are communicated throug pipes where each worker runs a single
        environment. Initiation is started by calling <start> method or using
        "with" statement. After initiating workers, step function can be
        called indefinetly. Each worker restart it's environment in case of
        termination and returns the first state of the restarted environment
        instead of the last state of the terminated one. As a result of this
        stepping is always continuous and homogeneuos.
            Arguments:
                - n_env: Number of environments
                - env_maker_fn: Function that returns environment

        Example:
            >>> p_env = ParallelEnv(n, lambda: gym.make(env_name))
            >>> with p_env as intial_state:
            >>>     actions = policy(initial_state)
            >>>     for i in range(TIMESTEPS):
            >>>         states, rewards, dones = p_env.step(actions)
    """

    EnvProcess = namedtuple("EnvProcess", "process, remote")

    def __init__(self, n_envs, env_maker_fn):
        self.env_maker_fn = env_maker_fn
        self.n_envs = n_envs
        self.started = False

    def start(self):
        """ Initiate worker processes and starts.
            Return:
                - state: First observations in a stacked form
            Raise:
                - RuntimeError: If called twice without close
        """
        if self.started is True:
            raise RuntimeError("cannot restart without closing")

        env_processes = []
        for p_r, w_r in (Pipe() for i in range(self.n_envs)):
            process = Process(target=self.worker,
                              args=(w_r, self.env_maker_fn),
                              daemon=True)
            env_processes.append(self.EnvProcess(process, p_r))
            process.start()
            p_r.send("start")
        self.env_processes = env_processes

        state = np.stack(remote.recv() for _, remote in self.env_processes)
        self.started = True
        return state

    def step(self, actions):
        """ Steps all the workers(environments) and return stacked
        observations, rewards and termination arrays. When a termination
        happens in one of the worker it returns the first observation of the
        restarted environment instead of returning the next-state of the
        terminated episode.
            Arguments:
                - actions: Stacked array of actions. Dim: (#env, #act)
            Return:
                - Stacked state, reward and done arrays. Dimension of state:
                    (#env, #obs), reward: (#env, 1), done: (#env, 1)
            Raise:
                - RuntimeError: If called before start
                - ValueError: If argument <actions> is not a 2D array
                - ValueError: If #actions(0th dimension) is not equal to
                    #environments
        """
        if self.started is False:
            raise RuntimeError("call <start> function first!")
        if len(actions.shape) != 2:
            raise ValueError("<actions> must be 2 dimensional!")
        if actions.shape[0] != self.n_envs:
            raise ValueError("not enough actions!")
        actions = actions.squeeze(-1)
        for act, (_, remote) in zip(actions, self.env_processes):
            remote.send(act)

        state, reward, done = [np.stack(batch) for batch in zip(*(
            remote.recv() for _, remote in self.env_processes))]
        return (state,
                reward.reshape(-1, 1).astype(np.float32),
                done.reshape(-1, 1).astype(np.float32))

    def close(self):
        """ Terminate and join all the workers.
        """
        for process, remote in self.env_processes:
            remote.send("end")
            process.terminate()
            process.join()
        self.started = False

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def worker(remote, env_maker_fn):
        """ Starts when initial start signal is received from <start> function.
        Following the start signal first observation is send through pipe.
        Then in a loop waits for the action from the pipe. If the action is
        "end" string than breaks the loop and terminates. Otherwise worker
        steps the environment and send (state, reward, done) triplet.

        (state, reward, done) -> action -> (state, reward, done) ->
            _____                   ______            ______  ____

            Underlined elements resembles a single transition. Note that when
        the termination happens returned state is the first state of the
        restarted environment.
            Arguments:
                - remote: Child pipe
                - env_maker_fn: Function that returns env object
        """
        env = env_maker_fn()
        state = env.reset()
        # Wait for the start command
        remote.recv()
        remote.send(state)
        while True:
            action = remote.recv()
            if action == "end":
                break
            state, reward, done, info = env.step(action.item())
            if done:
                state = env.reset()
            remote.send((state, reward, done))


if __name__ == "__main__":
    import gym

    vectorized_envs = ParallelEnv(1, lambda: gym.make("CartPole-v0"))

    with vectorized_envs as state:
        for i in range(4000):
            actions = np.ones((1, 1), dtype=np.int)*0
            state, reward, done = vectorized_envs.step(actions)
            # print(state, reward, done)
