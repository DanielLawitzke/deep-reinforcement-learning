from unityagents import UnityEnvironment
import numpy as np

print("=" * 50)
print("Unity Environment Test")
print("=" * 50)

# Environment laden
env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")
print("Environment geladen!")

# Get default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

print(f"\nBrain Name: {brain_name}")
print(f"Action Size: {brain.vector_action_space_size}")

# Reset environment
env_info = env.reset(train_mode=True)[brain_name]
state = env_info.vector_observations[0]

print(f"State Size: {len(state)}")
print(f"First few state values: {state[:5]}")

# Random action test
action = np.random.randint(0, brain.vector_action_space_size)
env_info = env.step(action)[brain_name]

print(f"\nReward: {env_info.rewards[0]}")
print(f"Done: {env_info.local_done[0]}")

print("\n" + "=" * 50)
print("UNITY ENVIRONMENT FUNKTIONIERT!")
print("=" * 50)

env.close()
