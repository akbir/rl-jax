import collections
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from haiku import nets

Params = collections.namedtuple("Params", "actor critic old_actor")
ActorState = collections.namedtuple("ActorState", "count")
LearnerState = collections.namedtuple("LearnerState", "count actor_opt_state critic_opt_state")
ActorOutput = collections.namedtuple("ActorOutput", "action log_prob advantage")

def build_actor_network(num_actions: int) -> hk.Transformed:
  """Factory for a simple MLP network."""
  def network(obs):
    network = hk.Sequential(
        [hk.Flatten(),
         nets.MLP([64, num_actions]),
         jnp.tanh])
    return network(obs)
  return hk.without_apply_rng(hk.transform(network, apply_rng=True))

def build_critic_network(num_actions: int) -> hk.Transformed:
  """Factory for a simple MLP network."""
  def network(obs):
    network = hk.Sequential(
        [hk.Flatten(),
         nets.MLP([64, num_actions])])
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
    self._actor_optimizer = optax.adam(learning_rate)
    self._critic_optimizer = optax.adam(learning_rate)

    # Jitting for speed.
    # self.actor_step = jax.jit(self.actor_step)
    # self.learner_step = jax.jit(self.learner_step)

  def initial_params(self, key) -> Params:
    sample_input = jnp.zeros((0, self._observation_spec))
    actor_params = self._actor.init(key, sample_input)
    critic_params = self._critic.init(key, sample_input)
    return Params(actor_params, critic_params, actor_params)

  def initial_actor_state(self) -> ActorState:
    actor_count = jnp.zeros((), dtype=jnp.float32)
    return ActorState(actor_count)

  def initial_learner_state(self, params) -> LearnerState:
    learner_count = jnp.zeros((), dtype=jnp.float32)
    actor_opt_state = self._actor_optimizer.init(params.actor)
    critic_opt_state = self._critic_optimizer.init(params.critic)
    return LearnerState(learner_count, actor_opt_state, critic_opt_state)

  def actor_step(self, params, env_output, actor_state, key, evaluation) -> Tuple[jnp.ndarray, ActorState]:
    obs = jnp.expand_dims(env_output.observation, 0)  # dummy batch
    q_value = self._critic.apply(params.critic, obs)[0]
    logits = self._actor.apply(params.actor, obs)[0]

    logp = jnp.log(logits)
    train_a = jax.random.categorical(key, logits)

    # else be greedy
    eval_a = jnp.argmax(logits)
    action = jax.lax.select(evaluation, eval_a, train_a)
    return action, ActorState(actor_state.count + 1)

  def learner_step(self, params, data, learner_state, unused_key) -> Tuple[Params, LearnerState]:
    # update critic
    dl_dc = jax.grad(self._critic_loss)(params.critic, data)
    critic_updates, critic_opt_state = self._critic_optimizer.update(dl_dc, learner_state.critic_opt_state)
    critic_params = optax.apply_updates(params.crtic, critic_updates)

    # update actor
    dl_da = jax.grad(self._actor_loss)(params.actor, *data)
    actor_updates, actor_opt_state = self._actor_optimizer.update(dl_da, learner_state.actor_opt_state)
    actor_params = optax.apply_updates(params.actor, actor_updates)

    return (
      Params(actor_params, critic_params, params.crtic_params),
      LearnerState(learner_state.count + 1, actor_opt_state, critic_opt_state))

  def _critic_loss(self, critic_params, *data):
    obs_tm1, a_tm1, r_t, discount_t, obs_t = data
    target = data.r_t + discount_t * self._critic.apply(critic_params, obs_tm1)
    return jnp.mean(jax.lax.stop_gradient(target) - self._critic.apply(critic_params, obs_t))

  def _actor_loss(self, actor_params, *data):
    pass

