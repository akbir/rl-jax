import os

import gym
import haiku as hk
import jax

from replay_buffer import ReplayBuffer, EnvOutput, SequenceBuffer
from src.ppo import PPO, LearnerState
from tensorboardX import SummaryWriter
from datetime import datetime

def main(unused_arg):
  seed = 0
  env = gym.make('CartPole-v1')
  mini_batch_size = 1024
  iterations = 400
  num_epochs = 5
  accumulator = ReplayBuffer(2 * mini_batch_size * num_epochs)
  eval_episodes = 3
  clip = 0.1
  beta = 0.01
  lmbda = 0.95
  discount_factor = 0.99
  horizon = 50

  # Logging
  now = datetime.now()
  experiment_name = "PPO Run:" + now.strftime("%m/%d/%Y, %H:%M:%S") + 'with reward normalisation refactor'
  logdir = os.path.join("../logs/", "PPO", experiment_name)
  writer = SummaryWriter(logdir=logdir)

  # Agent
  agent = PPO(observation_spec=4,
              action_spec=2,
              actor_learning_rate=0.001,
              critic_learning_rate=0.01)

  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  params = agent.initial_params(next(rng))
  learner_state = agent.initial_learner_state(params, clip, beta)

  for episode in range(iterations):

    # Exploring
    while not accumulator.is_ready(2 *num_epochs * mini_batch_size):
      obs = env.reset()
      env_output = EnvOutput(obs, 0, discount_factor, False)
      actor_state = agent.initial_actor_state()
      seq = SequenceBuffer(obs_spec=4, action_spec=2, max_length=horizon)
      seq.append(env_output, None, None, None)

      while not env_output.done:
        # agent - environment interaction
        value, action, pi, actor_state = agent.actor_step(params, env_output, actor_state, next(rng), False)
        obs, reward, done, info = env.step(action.item())
        env_output = EnvOutput(obs, reward, discount_factor, done, info)

        # store in current sequence
        seq.append(env_output, value, action, pi)

        if seq.full() or env_output.done:
          # add to ReplayBuffer
          accumulator.add_trajectory(seq.drain(discount_factor, lmbda))

    # Learning
    for _ in range(num_epochs):
      actor_loss, critic_loss, entropy, params, learner_state = agent.learner_step(
          params, accumulator.sample(mini_batch_size*num_epochs), learner_state, next(rng))

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
      obs = env.reset()
      env_output = EnvOutput(obs, 0, discount_factor, False)
      actor_state = agent.initial_actor_state()

      while not env_output.done:
        value, action, log_p, actor_state = agent.actor_step(
          params, env_output, actor_state, next(rng), evaluation=True)
        obs, reward, done, info = env.step(int(action))
        env_output = EnvOutput(obs, reward, discount_factor, done, info)
        returns += reward
      writer.add_scalar('PPO/returns per episode', returns, episode+eval_num)
      writer.add_scalar('PPO/total episode length', actor_state.count, episode+eval_num)

    avg_returns = returns / eval_episodes
    writer.add_scalar(f'PPO/average returns', avg_returns, episode)
    print(f"Iteration:{episode:4d} Average returns: {avg_returns:.2f}")

  writer.close()
  env.close()


if __name__ == "__main__":
  main(None)