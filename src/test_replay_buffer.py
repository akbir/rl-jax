from replay_buffer import SequenceBuffer, EnvOutput
import jax.numpy as jnp

def test_sequence_buffer():
    buffer = SequenceBuffer(obs_spec=1, max_length=5)

    timestamp = EnvOutput(observation=jnp.ones(1),
                         reward=jnp.ones(1),
                         done=False,
                         info=None)

    for i in range(5):
        buffer.append(timestamp, value=1, action=0, pi=0)

    result_trajectory = buffer.drain(gamma=0.5, lmbda=1)

    assert (result_trajectory.rtg == jnp.array([4, 3, 2, 1, 0])).all()
    assert (result_trajectory.gae == jnp.array([1,2,3,4,5])).all()
