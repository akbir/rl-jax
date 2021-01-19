from typing import Optional, NamedTuple

import jax.numpy as jnp
import random
import collections

class EnvOutput(NamedTuple):
  observation: jnp.array
  reward: jnp.array
  discount: jnp.array
  done: bool
  info: Optional[str] = None

class ReplayBuffer(object):
  """A simple Python replay buffer."""
  def __init__(self, capacity):
    self._prev = None
    self._action = None
    self._latest = None
    self.buffer = collections.deque(maxlen=capacity)

  def push(self, env_output: EnvOutput, action):
    self._prev = self._latest
    self._action = action
    self._latest = env_output

    if action is not None:
      self.buffer.append(
          (self._prev.observation, self._action, self._latest.reward,
           self._latest.discount, self._latest.observation))

  def sample(self, batch_size):
    obs_tm1, a_tm1, r_t, discount_t, obs_t = zip(
        *random.sample(self.buffer, batch_size))
    return (jnp.stack(obs_tm1), jnp.asarray(a_tm1), jnp.asarray(r_t),
            jnp.asarray(discount_t) * FLAGS.discount_factor, jnp.stack(obs_t))

  def is_ready(self, batch_size):
    return batch_size <= len(self.buffer)