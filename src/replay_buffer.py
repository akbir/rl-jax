from typing import Optional, NamedTuple, Tuple

import jax.numpy as jnp
import random
import collections
import numpy as np

class EnvOutput(NamedTuple):
  observation: jnp.array
  reward: jnp.array
  discount: jnp.array
  done: bool
  info: Optional[str] = None

class Trajectory(NamedTuple):
  """A trajectory is a sequence of observations, ... etc.
  Note: `observations` should be of length T+1 to make up the final transition.
  """
  observations: jnp.ndarray  # [T + 1, ...]
  actions: jnp.ndarray  # [T]
  pi: jnp.ndarray  # [T]
  gae: jnp.ndarray  # [T]
  rtg: jnp.ndarray  # [T]
  rewards: jnp.ndarray  # [T]
  discounts: jnp.ndarray  # [T]

class SequenceBuffer:
  """A buffer for accumulating trajectories."""
  _observations: jnp.ndarray
  _actions: jnp.ndarray
  _rewards: jnp.ndarray
  _discounts: jnp.ndarray

  _max_sequence_length: int
  _needs_reset: bool = True
  _t: int = 0

  def __init__(
      self,
      obs_spec: jnp.array,
      action_spec: jnp.array,
      max_length: int,
  ):
    """Pre-allocates buffers of numpy arrays to hold the sequences."""
    self._observations = np.zeros(
        shape=(max_length + 1, obs_spec))
    self._values = np.zeros(
        shape=max_length, dtype=jnp.float32)
    self._actions = np.zeros(
        shape=max_length, dtype=jnp.float32)
    self._pi = np.zeros(
        shape=(max_length, action_spec))
    self._rewards = np.zeros(max_length, dtype=jnp.float32)
    self._discounts = np.zeros(max_length, dtype=jnp.float32)
    self._max_sequence_length = max_length

  def append(
      self,
      timestep: EnvOutput,
      value: int,
      action: int,
      pi: int,
  ):
    """Appends an observation, action, reward, and discount to the buffer."""
    if self.full():
      raise ValueError('Cannot append; sequence buffer is full.')

    # Start a new sequence with an initial observation, if required.
    if self._needs_reset:
      self._t = 0
      self._observations[self._t] = timestep.observation
      self._needs_reset = False

    # Append (o, v, a, pi, r, d) to the sequence buffer.
    self._observations[self._t + 1] = timestep.observation
    self._values[self._t] = value
    self._actions[self._t] = action
    self._pi[self._t] = pi
    self._rewards[self._t] = timestep.reward
    self._discounts[self._t] = timestep.discount
    self._t += 1

    # Don't accumulate sequences that cross episode boundaries.
    # It is up to the caller to drain the buffer in this case.
    if timestep.done:
      self._needs_reset = True

  def drain(self, gamma: float, lmbda: float) -> Trajectory:
    """Empties the buffer and returns the (possibly partial) trajectory."""
    if self.empty():
      raise ValueError('Cannot drain; sequence buffer is empty.')

    gae = np.zeros(self._t)
    rtg = np.zeros(self._t)

    for t in reversed(range(self._t - 1)):
      # Calculate rewards-to-go
      rtg[t] = self._rewards[t] + gamma * (rtg[t + 1] if t + 1 < self._t else 0)
      # Calculate TD errors.
      delta = self._rewards[t] + gamma * self._values[t + 1] - self._values[t]
      # Calculate GAE recursively
      gae[t] = delta + gamma * lmbda * (gae[t + 1] if t + 1 < self._t else 0)

    trajectory = Trajectory(
      self._observations[:self._t + 1],
      self._actions[:self._t],
      self._pi[:self._t],
      gae,
      rtg,
      self._rewards[:self._t],
      self._discounts[:self._t],
    )
    self._t = 0  # Mark sequences as consumed.
    self._needs_reset = True
    return trajectory

  def empty(self) -> bool:
    """Returns whether or not the trajectory buffer is empty."""
    return self._t == 0

  def full(self) -> bool:
    """Returns whether or not the trajectory buffer is full."""
    return self._t == self._max_sequence_length

class ReplayBuffer(object):
  """A buffer to store trajectories and apply normalisation"""
  def __init__(self, capacity):
    self._sum = 0
    self._squared_sum = 0
    self.buffer = collections.deque(maxlen=capacity)
    self.capacity = capacity

  def add_trajectory(self, traj: Trajectory) -> None:
    for t in range(1, len(traj.actions)):
      # add to buffer
      self.buffer.append(
        (traj.observations[t - 1], traj.actions[t], traj.pi[t], traj.gae[t],
         traj.rtg[t], traj.discounts[t], traj.observations[t])
      )
      # for normalisation stats
      self._sum += traj.gae[t]
      self._squared_sum += traj.gae[t] * traj.gae[t]

  def sample(self, batch_size: int) -> Tuple:
    obs_tm1, a_tm1, pi_tm1, adv_t, rtg_t, discount_t, obs_t = zip(
        *random.sample(self.buffer, batch_size))

    return (jnp.stack(obs_tm1),
            jnp.asarray(a_tm1),
            jnp.stack(pi_tm1),
            self._normalize(jnp.asarray(adv_t)),
            jnp.asarray(rtg_t),
            jnp.asarray(discount_t),
            jnp.stack(obs_t))

  def _normalize(self, values: jnp.ndarray) -> jnp.ndarray:
    _mean = self._sum/len(self.buffer)
    _std = np.sqrt(self._squared_sum / len(self.buffer) - _mean ** 2)
    return (values - _mean) / (_std + 1e-8)

  def is_ready(self, batch_size) -> bool:
    return batch_size <= len(self.buffer)

  def reset(self):
    self._sum = 0
    self._squared_sum = 0
    self.buffer = collections.deque(maxlen=self.capacity)
