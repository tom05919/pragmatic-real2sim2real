"""Main entry point for running experiments.

Examples:
    python experiments/run_experiment.py env=obstruction2d-o0 seed=0

    python experiments/run_experiment.py -m env=obstruction2d-o0 seed='range(0,10)'

    python experiments/run_experiment.py -m env=obstruction2d-o0 seed=0 \
        samples_per_step=1,5,10

    python experiments/run_experiment.py -m env=stickbutton2d-b3 seed=0 \
        max_abstract_plans=1,5,10,20
"""

import logging
import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium.core import Env
from gymnasium.wrappers import RecordVideo
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from prpl_utils.utils import sample_seed_from_rng, timer

from pr2s2r import prbench
from pr2s2r.prbench_bilevel_planning.agent import AgentFailure, BilevelPlanningAgent
from pr2s2r.prbench_bilevel_planning.env_models import create_bilevel_planning_models
from pr2s2r.real_to_sim.bounding_box import measure_object


@hydra.main(version_base=None, config_name="config", config_path="conf/")
def _main(cfg: DictConfig) -> None:

    logging.info(f"Running seed={cfg.seed}, env={cfg.env.env_name}")

    if "real_image_path" in cfg and cfg.real_image_path is not None:
        logging.info(f"Detecting object from real image: {cfg.real_image_path}")
        real_image_path = cfg.real_image_path

        if not os.path.isabs(real_image_path):
            project_root = Path(__file__).resolve().parents[4]
            real_image_path = str(project_root / real_image_path)

        current_dir = HydraConfig.get().runtime.output_dir
        detection_output_path = os.path.join(current_dir, "detected_object.png")
        try:
            measurement = measure_object(real_image_path, detection_output_path)

            # Scale dimensions
            scale_factor = 1500.0
            blocker_width = measurement.total_width / scale_factor
            blocker_height = measurement.total_height / scale_factor

            logging.info(
                f"Detected dimensions: {measurement.total_width:.2f}x{measurement.total_height:.2f}"
            )
            logging.info(f"Scaled dimensions: {blocker_width:.4f}x{blocker_height:.4f}")

            # Update config
            if "make_kwargs" not in cfg.env:
                OmegaConf.update(cfg.env, "make_kwargs", {}, force_add=True)

            OmegaConf.set_struct(cfg.env.make_kwargs, False)
            cfg.env.make_kwargs["blocker_width"] = blocker_width
            cfg.env.make_kwargs["blocker_height"] = blocker_height
            OmegaConf.set_struct(cfg.env.make_kwargs, True)

        except Exception as e:
            logging.error(f"Failed to process real image: {e}")
            raise e

    # Create the environment.
    prbench.register_all_environments()
    env = prbench.make(**cfg.env.make_kwargs, render_mode="rgb_array")

    # Record videos.
    if cfg.make_videos:
        video_path = Path(cfg.video_folder)
        video_path.mkdir(parents=True, exist_ok=True)
        env = RecordVideo(env, str(video_path), episode_trigger=lambda _: True)

    # Create the env models.
    env_models = create_bilevel_planning_models(
        cfg.env.env_name,
        env.observation_space,
        env.action_space,
        **cfg.env.env_model_kwargs,
    )

    # Create the agent.
    agent: BilevelPlanningAgent = BilevelPlanningAgent(
        env_models,
        cfg.seed,
        max_abstract_plans=cfg.max_abstract_plans,
        samples_per_step=cfg.samples_per_step,
        max_skill_horizon=cfg.max_skill_horizon,
        heuristic_name=cfg.heuristic_name,
        planning_timeout=cfg.planning_timeout,
    )

    # Evaluate.
    rng = np.random.default_rng(cfg.seed)
    metrics: list[dict[str, float]] = []
    current_dir = HydraConfig.get().runtime.output_dir
    for eval_episode in range(25):
        logging.info(f"Starting evaluation episode {eval_episode}")
        episode_metrics = _run_single_episode_evaluation(
            agent,
            env,
            rng,
            max_eval_steps=cfg.max_eval_steps,
            eval_episode=eval_episode,
            output_dir=current_dir,
        )
        episode_metrics["eval_episode"] = eval_episode
        metrics.append(episode_metrics)

    # Aggregate and save results.
    df = pd.DataFrame(metrics)

    # Save the metrics dataframe.
    results_path = os.path.join(current_dir, "results.csv")
    df.to_csv(results_path, index=False)
    logging.info(f"Saved results to {results_path}")

    # Save the full hydra config.
    config_path = os.path.join(current_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)
    logging.info(f"Saved config to {config_path}")

    # Finish.
    env.close()  # type: ignore


def _run_single_episode_evaluation(
    agent: BilevelPlanningAgent,
    env: Env,
    rng: np.random.Generator,
    max_eval_steps: int,
    eval_episode: int,
    output_dir: str,
) -> dict[str, float]:
    steps = 0
    success = False
    seed = sample_seed_from_rng(rng)
    obs, info = env.reset(seed=seed)

    # Capture and save the first frame
    first_frame = env.render()  # type: ignore
    first_frame_path = os.path.join(
        output_dir, f"episode_{eval_episode}_first_frame.png"
    )
    plt.imsave(first_frame_path, first_frame)  # type: ignore
    logging.info(f"Saved first frame to {first_frame_path}")

    planning_time = 0.0  # measure the time taken by the approach only
    planning_failed = False
    with timer() as result:
        try:
            agent.reset(obs, info)
        except AgentFailure:
            logging.info("Agent failed during reset().")
            planning_failed = True
    planning_time += result["time"]
    if planning_failed:
        # Save last frame even on failure
        last_frame = env.render()  # type: ignore
        last_frame_path = os.path.join(
            output_dir, f"episode_{eval_episode}_last_frame.png"
        )
        plt.imsave(last_frame_path, last_frame)  # type: ignore
        return {"success": False, "steps": steps, "planning_time": planning_time}
    for _ in range(max_eval_steps):
        step_failed = False
        with timer() as result:
            try:
                action = agent.step()
            except AgentFailure:
                logging.info("Agent failed during step().")
                step_failed = True
        planning_time += result["time"]
        if step_failed:
            # Save last frame on step failure
            last_frame = env.render()  # type: ignore
            last_frame_path = os.path.join(
                output_dir, f"episode_{eval_episode}_last_frame.png"
            )
            plt.imsave(last_frame_path, last_frame)  # type: ignore
            return {"success": False, "steps": steps, "planning_time": planning_time}
        obs, rew, done, truncated, info = env.step(action)
        reward = float(rew)
        assert not truncated
        with timer() as result:
            agent.update(obs, reward, done, info)
        planning_time += result["time"]
        if done:
            success = True
            break
        steps += 1

    # Capture and save the last frame
    last_frame = env.render()  # type: ignore
    last_frame_path = os.path.join(output_dir, f"episode_{eval_episode}_last_frame.png")
    plt.imsave(last_frame_path, last_frame)  # type: ignore
    logging.info(f"Saved last frame to {last_frame_path}")

    logging.info(f"Success result: {success}")
    return {"success": success, "steps": steps, "planning_time": planning_time}


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
