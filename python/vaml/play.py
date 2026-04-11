import numpy as np
from vaml.env.make import make_env


def play(env_name: str, seed: int = 42):
    env = make_env(env_name, 1, seed, None)

    print(f"\n--- {env_name.upper()} ---")
    print(f"Instructions: {env.instructions()}")
    print("Type 'quit' to exit.\n")

    idx = np.array([0], dtype=np.int32)
    obs = env.reset(idx)
    print(f"[ENV] {obs[0]}\n")

    total_reward = 0.0
    episode = 1

    while True:
        try:
            action = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if action.strip().lower() == "quit":
            break

        obs, rewards, dones = env.step(idx, [action])
        reward = rewards[0]
        done = dones[0]
        total_reward += reward

        if reward > 0:
            print(f"[REWARD] {reward:.2f}")

        if done:
            print(f"[DONE] Episode {episode} total reward: {total_reward:.2f}")
            total_reward = 0.0
            episode += 1
            print()

        print(f"[ENV] {obs[0]}\n")
