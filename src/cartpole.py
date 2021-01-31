import os
from datetime import datetime

import bsuite
import haiku as hk
import jax
from tensorboardX import SummaryWriter

from replay_buffer import ReplayBuffer, SequenceBuffer
from src.ppo import PPO, LearnerState


def main(unused_arg):
  seed = 0
  env = bsuite.load_from_id('cartpole/0')
  mini_batch_size = 4096
  iterations = 200
  num_epochs = 5
  accumulator = ReplayBuffer(2 * mini_batch_size * num_epochs)
  eval_episodes = 3
  clip = 0.1
  beta = 0.01
  lmbda = 0.99
  discount_factor = 0.995
  horizon = 400

  # Logging
  now = datetime.now()
  experiment_name = "PPO Run:" + now.strftime("%m/%d/%Y, %H:%M:%S") + 'dm envs'
  logdir = os.path.join("../logs/", "PPO", experiment_name)
  writer = SummaryWriter(logdir=logdir)

  observation_spec = 6
  action_spec = 3
  # Agent
  agent = PPO(observation_spec=observation_spec,
              action_spec=action_spec,
              actor_learning_rate=0.001,
              critic_learning_rate=0.01)

  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  params = agent.initial_params(next(rng))
  learner_state = agent.initial_learner_state(params, clip, beta)

  for episode in range(iterations):
    # Exploring
    while not accumulator.is_ready(2 *num_epochs * mini_batch_size):
      timestep = env.reset()
      actor_state = agent.initial_actor_state()
      seq = SequenceBuffer(obs_spec=observation_spec, action_spec=action_spec, max_length=horizon)
      seq.append(timestep, None, None, None)

      while not timestep.last():
        # agent - environment interaction
        value, action, pi, actor_state = agent.actor_step(params, timestep, actor_state, next(rng), False)
        timestep = env.step(action.item())
        # store in current sequence
        seq.append(timestep, value, action, pi)

        if seq.full() or timestep.last():
          # add to ReplayBuffer
          accumulator.add_trajectory(seq.drain(discount_factor, lmbda))

    # Learning
    for _ in range(num_epochs):
      actor_loss, critic_loss, entropy, params, learner_state = agent.learner_step(
          params, accumulator.sample(mini_batch_size), learner_state, next(rng))

      writer.add_scalar(f'PPO/actor loss', actor_loss, learner_state.count)
      writer.add_scalar(f'PPO/critic loss', critic_loss, learner_state.count)
      writer.add_scalar(f'PPO/entropy loss', entropy, learner_state.count)

    accumulator.reset()
    learner_state = LearnerState(learner_state.count,
                                 learner_state.actor_opt_state,
                                 learner_state.critic_opt_state,
                                 learner_state.beta*0.999,
                                 learner_state.clip*0.999)
    # Evaluation
    returns = 0.
    for eval_num in range(eval_episodes):
      timestep = env.reset()
      actor_state = agent.initial_actor_state()

      while not timestep.last():
        value, action, log_p, actor_state = agent.actor_step(
          params, timestep, actor_state, next(rng), evaluation=True)
        timestep = env.step(int(action))
        returns += timestep.reward
      writer.add_scalar('PPO/returns per episode', returns, episode+eval_num)
      writer.add_scalar('PPO/total episode length', actor_state.count, episode+eval_num)

    avg_returns = returns / eval_episodes
    writer.add_scalar(f'PPO/average returns', avg_returns, episode)
    print(f"Iteration:{episode:4d} Average returns: {avg_returns:.2f}")
  writer.close()
  env.close()
if __name__ == "__main__":
  main(None)