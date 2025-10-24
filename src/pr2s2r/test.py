"""Test script for prbench environments."""

from datetime import datetime

import matplotlib.pyplot as plt

from pr2s2r import prbench
from pr2s2r.prbench.envs.geom2d.clutteredstorage2d import (
    ObjectCentricClutteredStorage2DEnv,
)

prbench.register_all_environments()
env = prbench.make("prbench/ClutteredStorage2D-b1-v0")  # 1 block
obs, info = env.reset()  # procedural generation
env = ObjectCentricClutteredStorage2DEnv(num_blocks=1)
obs, _ = env.reset(seed=123)
print(obs.pretty_str())
action = env.action_space.sample()
next_obs, reward, terminated, truncated, info = env.step(action)
img = env.render()  # type: ignore[var-annotated]

# Save the rendered image
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"prbench_test_image_{timestamp}.png"
plt.imsave(filename, img)  # type: ignore[arg-type]
print(f"Image saved to {filename}")
