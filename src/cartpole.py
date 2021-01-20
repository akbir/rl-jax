import gym
import haiku as hk
import jax
from replay_buffer import ReplayBuffer, EnvOutput
from src.ppo import PPO

def main(unused_arg):
  seed = 0
  env = gym.make('CartPole-v1')
  accumulator = ReplayBuffer(50000)
  batch_size = 256
  discount_factor = 0.995
  evaluate_every = 5
  eval_episodes = 5

  agent = PPO(observation_spec=4, action_spec=2, learning_rate=0.001)

  # init agent
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  params = agent.initial_params(next(rng))
  learner_state = agent.initial_learner_state(params)

  for episode in range(100):
    # init agent and environment
    obs = env.reset()
    env_output = EnvOutput(obs, 0, discount_factor, False)
    actor_state = agent.initial_actor_state()
    accumulator.push(env_output, None, None)

    while not env_output.done:
      # env.render()
      # agent - environment interaction
      action, pi, actor_state = agent.actor_step(params, env_output, actor_state, next(rng), False)
      obs, reward, done, info = env.step(action.item())
      env_output = EnvOutput(obs, reward, discount_factor, done, info)

      # store
      accumulator.push(env_output, action, pi)

      # learning.
      if accumulator.is_ready(batch_size):
        params, learner_state = agent.learner_step(
            params, accumulator.sample(batch_size, discount_factor), learner_state, next(rng))
    print(f"Episode finished after {int(actor_state.count)} timesteps")

    # Evaluation.
    if not episode % evaluate_every:
      returns = 0.
      for _ in range(eval_episodes):
        obs = env.reset()
        env_output = EnvOutput(obs, 0, 1.0, False)
        actor_state = agent.initial_actor_state()
        while not env_output.done:
          action, log_p, actor_state = agent.actor_step(
            params, env_output, actor_state, next(rng), evaluation=True)
          obs, reward, done, info = env.step(int(action))
          env_output = EnvOutput(obs, reward, discount_factor, done, info)
          returns += reward
      avg_returns = returns / eval_episodes
      print(f"Episode {episode:4d}: Average returns: {avg_returns:.2f}")

  env.close()


if __name__ == "__main__":
  main(None)