import collections
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from haiku import nets
from rlax import softmax

Params = collections.namedtuple("Params", "actor critic")
ActorState = collections.namedtuple("ActorState", "count")
LearnerState = collections.namedtuple("LearnerState", "count actor_opt_state critic_opt_state clip beta")
ActorOutput = collections.namedtuple("ActorOutput", "action log_prob advantage")

def build_actor_network(num_actions: int) -> hk.Transformed:
  """Factory for a simple MLP network."""
  def network(obs):
    network = hk.Sequential(
        [
         nets.MLP([64, 64, num_actions], activation=jnp.tanh),
         ])
    return network(obs)
  return hk.without_apply_rng(hk.transform(network, apply_rng=True))

def build_critic_network(num_actions: int) -> hk.Transformed:
  """Factory for a simple MLP network."""
  def network(obs):
    network = hk.Sequential(
        [nets.MLP([64, 64, 1], activation=jnp.tanh)])
    return network(obs)
  return hk.without_apply_rng(hk.transform(network, apply_rng=True))


class PPO:
  """A simple PPO agent."""
  def __init__(self, observation_spec, action_spec, learning_rate):
    self._observation_spec = observation_spec
    self._action_spec = action_spec

    # Neural nets and optimisers.
    self._actor = build_actor_network(action_spec)
    self._critic = build_critic_network(action_spec)
    self._actor_optimizer = optax.sgd(0.001)
    self._critic_optimizer = optax.sgd(0.01)

    # Jitting for speed.
    self.actor_step = jax.jit(self.actor_step)
    self.learner_step = jax.jit(self.learner_step)

  def initial_params(self, key) -> Params:
    sample_input = jnp.zeros((0, self._observation_spec))
    actor_params = self._actor.init(key, sample_input)
    critic_params = self._critic.init(key, sample_input)
    return Params(actor_params, critic_params)

  def initial_actor_state(self) -> ActorState:
    actor_count = jnp.zeros((), dtype=jnp.float32)
    return ActorState(actor_count)

  def initial_learner_state(self, params, clip, beta) -> LearnerState:
    learner_count = jnp.zeros((), dtype=jnp.float32)
    actor_opt_state = self._actor_optimizer.init(params.actor)
    critic_opt_state = self._critic_optimizer.init(params.critic)
    return LearnerState(learner_count, actor_opt_state, critic_opt_state, clip, beta)

  def actor_step(self, params, env_output, actor_state, key, evaluation)\
          -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, ActorState]:
    obs = jnp.expand_dims(env_output.observation, 0)  # dummy batch
    logits = self._actor.apply(params.actor, obs)[0]
    value = self._critic.apply(params.critic, obs)[0]
    # sample from policy
    train_a = jax.random.categorical(key, logits).squeeze()
    # else be greedy
    eval_a = jnp.argmax(logits)
    action = jax.lax.select(evaluation, eval_a, train_a)
    return value, action, logits[action], ActorState(actor_state.count + 1)

  def learner_step(self, params, data, learner_state, unused_key) -> Tuple[jnp.ndarray, jnp.ndarray, Params, LearnerState]:
    obs_tm1, a_t, logpi_tm1, advs_tm1, rtgs_t, discount_t, obs_t = data
    advs_tm1 = (advs_tm1 - advs_tm1.mean()) / (advs_tm1.std() + 1e-8)

    # update critic
    critic_loss = self._critic_loss(params.critic, obs_tm1, rtgs_t)
    dl_dc = jax.grad(self._critic_loss)(params.critic, obs_tm1, rtgs_t)
    critic_updates, critic_opt_state = self._critic_optimizer.update(dl_dc, learner_state.critic_opt_state)
    critic_params = optax.apply_updates(params.critic, critic_updates)

    actor_loss, entropy = self._ppo_loss(params.actor, learner_state.clip, obs_tm1, a_t, logpi_tm1, advs_tm1),\
                          self._entropy_loss(params.actor, obs_t)

    # update actor
    dl_da = jax.grad(self._actor_loss)(params.actor, learner_state.clip, learner_state.beta,
                                       obs_tm1, a_t, logpi_tm1, advs_tm1, obs_t)
    actor_updates, actor_opt_state = self._actor_optimizer.update(dl_da, learner_state.actor_opt_state)
    actor_params = optax.apply_updates(params.actor, actor_updates)

    return (
      actor_loss, critic_loss, entropy,
      Params(actor_params, critic_params),
      LearnerState(learner_state.count + 1, actor_opt_state, critic_opt_state, learner_state.clip, learner_state.beta))

  def _actor_loss(self, actor_params, clip, beta, obs_tm1, a_tm1, logpi_tm1, advs, obs_t):
    return self._ppo_loss(actor_params, clip, obs_tm1, a_tm1, logpi_tm1, advs) \
           + beta * self._entropy_loss(actor_params, obs_t)

  def _critic_loss(self, critic_params, obs_tm1, target):
    q_tm1 = self._critic.apply(critic_params, obs_tm1)
    return jnp.power(jax.lax.stop_gradient(target) - q_tm1, 2).mean()

  def _ppo_loss(self, actor_params, eps, obs_tm1, a_tm1, logpi_tm1, advs):
    logits_t = self._actor.apply(actor_params, obs_tm1)
    logpi_t = softmax().logprob(a_tm1, logits_t)

    ratio = jnp.exp(logpi_t - logpi_tm1)
    surr1 = ratio * advs
    surr2 = jnp.clip(ratio, 1.0 - eps, 1.0 + eps) * advs
    return -1 * jnp.minimum(surr1, surr2).mean()

  def _entropy_loss(self, actor_params, obs_t):
    logits_t = self._actor.apply(actor_params, obs_t)
    return -1 * softmax().entropy(logits_t).mean()