import gym
import ray
from ray import tune

from ray.tune.registry import register_env

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper


class UnityEnvWrapper(gym.Env):
    def __init__(self, env_config):
        self.vector_index = env_config.vector_index
        self.worker_index = env_config.worker_index
        self.worker_id = env_config["unity_worker_id"] + env_config.worker_index
        env_name = '/home/miku/PythonObjects/unity-exercise/envs/Basic/Basic.x86_64'
        env = UnityEnvironment(env_name, worker_id=self.worker_id, no_graphics=True)
        self.env = UnityToGymWrapper(env)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


if __name__ == "__main__":
    register_env("unity_env", lambda config: UnityEnvWrapper(config))
    ray.init()
    tune.run(
        "PPO",
        stop={
            "timesteps_total": 100000,
        },
        config={
            "env": "unity_env",
            "num_workers": 3,
            "num_gpus": 1,
            "env_config": {
                "unity_worker_id": 52
            },
            "train_batch_size": 500,
        },
        checkpoint_at_end=True,
    )
