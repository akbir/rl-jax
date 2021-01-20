import gym
import haiku as hk
import jax
from replay_buffer import ReplayBuffer, EnvOutput
from src.ppo import PPO

def main(unused_arg):
  seed = 0
  env = gym.make('CartPole-v0')
  accumulator = ReplayBuffer(500)
  batch_size = 64
  discount_factor = 0.99

  agent = PPO(observation_spec=4, action_spec=2, learning_rate=0.001)

  # init agent
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  params = agent.initial_params(next(rng))
  learner_state = agent.initial_learner_state(params)

  for i_episode in range(20):

    # init agent and environment
    obs = env.reset()
    env_output = EnvOutput(obs, 0, 1.0, False)
    actor_state = agent.initial_actor_state()
    accumulator.push(env_output, None, None)

    while not env_output.done:
      env.render()

      # agent - environment interaction
      action, pi, actor_state = agent.actor_step(params, env_output, actor_state, next(rng), False)
      obs, reward, done, info = env.step(action.item())
      env_output = EnvOutput(obs, reward, 1.0, done, info)

      # store
      accumulator.push(env_output, action, pi)

      # learning.
      if accumulator.is_ready(batch_size):
        params, learner_state = agent.learner_step(
            params, accumulator.sample(batch_size, discount_factor), learner_state, next(rng))
    print(f"Episode finished after {actor_state.count} timesteps")
  env.close()


if __name__ == "__main__":
  main(None)