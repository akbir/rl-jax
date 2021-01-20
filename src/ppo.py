import collections
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from haiku import nets
from rlax._src import distributions

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
         ])
    return network(obs)
  return hk.without_apply_rng(hk.transform(network, apply_rng=True))

def build_critic_network(num_actions: int) -> hk.Transformed:
  """Factory for a simple MLP network."""
  def network(obs):
    network = hk.Sequential(
        [hk.Flatten(),
         nets.MLP([64, 1])])
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
    self.actor_step = jax.jit(self.actor_step)
    self.learner_step = jax.jit(self.learner_step)

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

  def actor_step(self, params, env_output, actor_state, key, evaluation) -> Tuple[jnp.ndarray, jnp.ndarray, ActorState]:
    obs = jnp.expand_dims(env_output.observation, 0)  # dummy batch
    logits = self._actor.apply(params.actor, obs)[0]
    # sample from policy
    train_a = distributions.softmax().sample(key, logits)
    # else be greedy
    eval_a = jnp.argmax(logits)
    action = jax.lax.select(evaluation, eval_a, train_a)
    return action, logits[action], ActorState(actor_state.count + 1)

  def learner_step(self, params, data, learner_state, unused_key) -> Tuple[Params, LearnerState]:
    obs_tm1, a_tm1, logpi_t, r_t, discount_t, obs_t = data
    # calculate advantage A(s,a) = R + yV(s') - V(s)
    adv = jax.lax.stop_gradient(
      jnp.expand_dims(r_t, 1)
      + jnp.expand_dims(discount_t, 1) * self._critic.apply(params.critic, obs_tm1)
      - self._critic.apply(params.critic, obs_t)
    )
    # update actor
    dl_da = jax.grad(self._actor_loss)(params.actor, adv, 0.2, *data)
    actor_updates, actor_opt_state = self._actor_optimizer.update(dl_da, learner_state.actor_opt_state)
    actor_params = optax.apply_updates(params.actor, actor_updates)

    # update critic
    dl_dc = jax.grad(self._critic_loss)(params.critic, *data)
    critic_updates, critic_opt_state = self._critic_optimizer.update(dl_dc, learner_state.critic_opt_state)
    critic_params = optax.apply_updates(params.critic, critic_updates)

    return (
      Params(actor_params, critic_params, params.critic),
      LearnerState(learner_state.count + 1, actor_opt_state, critic_opt_state))

  def _critic_loss(self, critic_params, obs_tm1, a_t, logp_t, r_t, discount_t, obs_t):
    target_tm1 = r_t + discount_t * self._critic.apply(critic_params, obs_tm1)
    return jnp.mean(jax.lax.stop_gradient(target_tm1) - self._critic.apply(critic_params, obs_t))

  def _actor_loss(self, actor_params, adv, eps, obs_tm1, a_t, logp_t, r_t, discount_t, obs_t):
    logits_t = self._actor.apply(actor_params, obs_t)
    logpi_tm1 = distributions.softmax().logprob(a_t, logits_t)

    ratio = jnp.exp(logpi_tm1 - logp_t)
    clip = jnp.clip(ratio, 1.0 - eps, 1.0 + eps) * adv
    return jnp.mean(jnp.minimum(ratio * adv, clip * adv))

