from pathlib import Path

import habitat_sim
import numpy as np
import quaternion

from spatial_rag.config import SCENE_PATH
from spatial_rag.explorer import Explorer


class SemanticExplorer(Explorer):
    """
    Explorer variant that records RGB, depth, and semantic observations at
    every scan pose while reusing the existing navigation logic.
    """

    def __init__(
        self,
        scene_path=SCENE_PATH,
        scene_dataset_config_file=None,
        require_semantics=False,
    ):
        self.scene_dataset_config_file = self._resolve_scene_dataset_config(
            scene_path,
            scene_dataset_config_file,
        )
        self.require_semantics = bool(require_semantics)
        if self.require_semantics and self.scene_dataset_config_file is None:
            raise ValueError(
                "SemanticExplorer could not find a Habitat scene-dataset config. "
                "Pass scene_dataset_config_file=... explicitly."
            )
        super().__init__(scene_path=scene_path)

    def _resolve_scene_dataset_config(self, scene_path, explicit_path):
        if explicit_path:
            return str(explicit_path)

        scene = Path(scene_path).expanduser()
        candidate_roots = []
        try:
            scene_resolved = scene.resolve(strict=False)
        except Exception:
            scene_resolved = scene
        candidate_roots.append(scene_resolved.parent)
        if not scene.is_absolute():
            candidate_roots.append((Path.cwd() / scene).resolve(strict=False).parent)

        seen = set()
        for root in candidate_roots:
            for parent in (root, *root.parents):
                parent_key = str(parent)
                if parent_key in seen:
                    continue
                seen.add(parent_key)

                direct = parent / "scene_dataset_config.json"
                if direct.is_file():
                    return str(direct)

                matches = sorted(parent.glob("*.scene_dataset_config.json"))
                if matches:
                    return str(matches[0])

        return None

    def _make_sim_settings(self, scene_path):
        settings = super()._make_sim_settings(scene_path)
        settings["depth_sensor"] = True
        settings["semantic_sensor"] = True
        settings["scene_dataset_config_file"] = self.scene_dataset_config_file
        return settings

    def _make_hab_cfg(self, settings):
        sim_cfg = habitat_sim.SimulatorConfiguration()
        import torch

        if torch.cuda.is_available():
            sim_cfg.gpu_device_id = 0

        sim_cfg.scene_id = settings["scene"]
        scene_dataset_config_file = settings.get("scene_dataset_config_file")
        if scene_dataset_config_file:
            sim_cfg.scene_dataset_config_file = scene_dataset_config_file

        sensor_specs = []

        if settings["color_sensor"]:
            color_sensor_spec = habitat_sim.CameraSensorSpec()
            color_sensor_spec.uuid = "color_sensor"
            color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            color_sensor_spec.resolution = [
                settings["height"],
                settings["width"],
            ]
            color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
            color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            color_sensor_spec.hfov = settings.get("fov", 90)
            sensor_specs.append(color_sensor_spec)

        if settings.get("depth_sensor", False):
            depth_sensor_spec = habitat_sim.CameraSensorSpec()
            depth_sensor_spec.uuid = "depth_sensor"
            depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor_spec.resolution = [
                settings["height"],
                settings["width"],
            ]
            depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
            depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            depth_sensor_spec.hfov = settings.get("fov", 90)
            sensor_specs.append(depth_sensor_spec)

        if settings.get("semantic_sensor", False):
            semantic_sensor_spec = habitat_sim.CameraSensorSpec()
            semantic_sensor_spec.uuid = "semantic_sensor"
            semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
            semantic_sensor_spec.resolution = [
                settings["height"],
                settings["width"],
            ]
            semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
            semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            semantic_sensor_spec.hfov = settings.get("fov", 90)
            sensor_specs.append(semantic_sensor_spec)

        down_sensor_spec = habitat_sim.CameraSensorSpec()
        down_sensor_spec.uuid = "down_sensor"
        down_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        down_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        down_sensor_spec.resolution = [160, 160]
        down_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        down_sensor_spec.orientation = [-np.pi / 2, 0.0, 0.0]
        down_sensor_spec.hfov = 70
        sensor_specs.append(down_sensor_spec)

        top_down_spec = habitat_sim.CameraSensorSpec()
        top_down_spec.uuid = "top_down"
        top_down_spec.sensor_type = habitat_sim.SensorType.COLOR
        top_down_spec.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
        top_down_spec.resolution = [2048, 2048]
        top_down_spec.position = [0.0, 20.0, 0.0]
        top_down_spec.orientation = [-np.pi / 2, 0.0, 0.0]
        top_down_spec.ortho_scale = 10.0
        sensor_specs.append(top_down_spec)

        center_top_spec = habitat_sim.CameraSensorSpec()
        center_top_spec.uuid = "center_top_view"
        center_top_spec.sensor_type = habitat_sim.SensorType.COLOR
        center_top_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        center_top_spec.resolution = [1200, 1200]
        center_top_spec.position = [0.0, 0.0, 0.0]
        center_top_spec.orientation = [-np.pi / 2, 0.0, 0.0]
        center_top_spec.hfov = 120.0
        sensor_specs.append(center_top_spec)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward",
                habitat_sim.agent.ActuationSpec(amount=1.0),
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left",
                habitat_sim.agent.ActuationSpec(amount=60.0),
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right",
                habitat_sim.agent.ActuationSpec(amount=60.0),
            ),
        }

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def _format_observation_bundle(self, obs):
        rgb = np.asarray(obs["color_sensor"])
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]

        depth = np.asarray(obs["depth_sensor"], dtype=np.float32)
        if depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth[:, :, 0]

        semantic = np.asarray(obs["semantic_sensor"])
        if semantic.ndim == 3 and semantic.shape[2] == 1:
            semantic = semantic[:, :, 0]

        return {
            "rgb": rgb.copy(),
            "depth": depth.copy(),
            "semantic": semantic.copy(),
        }

    def capture_observation_bundle_at_pose(self, position, orientation_deg):
        state = self.agent.get_state()
        state.position = np.array(position, dtype=np.float32)
        yaw = np.deg2rad(float(orientation_deg))
        state.rotation = quaternion.from_rotation_vector([0.0, yaw, 0.0])
        self.agent.set_state(state)

        obs = self.sim.get_sensor_observations()
        return self._format_observation_bundle(obs)

    def capture_semantic_at_pose(self, position, orientation_deg):
        return self.capture_observation_bundle_at_pose(position, orientation_deg)["semantic"]

    def capture_depth_at_pose(self, position, orientation_deg):
        return self.capture_observation_bundle_at_pose(position, orientation_deg)["depth"]

    def _capture_scan(self, position, scan_angles, captures, poses):
        state = self.agent.get_state()
        state.position = np.array(position, dtype=np.float32)

        for angle_deg in scan_angles:
            yaw = np.deg2rad(float(angle_deg))
            state.rotation = quaternion.from_rotation_vector([0.0, yaw, 0.0])
            self.agent.set_state(state)

            obs = self.sim.get_sensor_observations()
            captures.append(self._format_observation_bundle(obs))
            poses.append(
                {
                    "position": state.position.copy(),
                    "rotation": state.rotation,
                }
            )

    @staticmethod
    def split_modalities(captures):
        return {
            "rgb_frames": [capture["rgb"] for capture in captures],
            "depth_frames": [capture["depth"] for capture in captures],
            "semantic_frames": [capture["semantic"] for capture in captures],
        }

    def explore_full_house(
        self,
        meters_per_step=None,
        walk_step_m=None,
        scan_angles=None,
    ):
        profile = self._choose_scene_tour_profile()
        if meters_per_step is None:
            meters_per_step = profile["meters_per_step"]
        if walk_step_m is None:
            walk_step_m = profile["walk_step_m"]
        if scan_angles is None:
            scan_angles = profile["scan_angles"]
        scan_angles = tuple(scan_angles)

        start_state = self.agent.get_state()
        sorted_waypoints = self._plan_room_tour_waypoints(
            start_state.position,
            meters_per_step,
        )

        captures = []
        poses = []
        walk_step_m = max(float(walk_step_m), 0.1)

        current_pos = start_state.position

        for target in sorted_waypoints:
            path = habitat_sim.ShortestPath()
            path.requested_start = current_pos
            path.requested_end = target

            if self.sim.pathfinder.find_path(path):
                prev = np.array(current_pos, dtype=np.float32)

                for p_raw in path.points[1:]:
                    curr = np.array(p_raw, dtype=np.float32)
                    state = self.agent.get_state()

                    direction = curr - prev

                    if np.linalg.norm(direction) > 1e-5:
                        yaw = np.arctan2(direction[0], direction[2])
                        state.rotation = quaternion.from_rotation_vector(
                            [0, yaw, 0]
                        )
                    seg_len = float(np.linalg.norm(direction))
                    substeps = max(1, int(np.ceil(seg_len / walk_step_m)))
                    for s in range(1, substeps + 1):
                        alpha = float(s) / float(substeps)
                        interp = prev + direction * alpha
                        state.position = interp.astype(np.float32)
                        self.agent.set_state(state)
                    prev = curr

                self._capture_scan(target, scan_angles, captures, poses)
                current_pos = target

        return captures, poses

    def explore_full_house_split(
        self,
        meters_per_step=None,
        walk_step_m=None,
        scan_angles=None,
    ):
        captures, poses = self.explore_full_house(
            meters_per_step=meters_per_step,
            walk_step_m=walk_step_m,
            scan_angles=scan_angles,
        )
        return self.split_modalities(captures), poses

    def explore_custom_tour(
        self,
        num_steps=50,
        step_size=1.0,
        scan_angles=(0, 90, 180, 270),
        seed=None,
        max_attempts_per_step=32,
        include_start_scan=True,
    ):
        if num_steps <= 0:
            return [], []

        step_size = float(step_size)
        if step_size <= 0.0:
            raise ValueError("step_size must be > 0")

        max_attempts_per_step = int(max_attempts_per_step)
        if max_attempts_per_step <= 0:
            raise ValueError("max_attempts_per_step must be > 0")

        scan_angles = tuple(scan_angles)
        rng = np.random.default_rng(seed)

        captures = []
        poses = []

        start_state = self.agent.get_state()
        current = np.array(start_state.position, dtype=np.float32)

        if include_start_scan:
            self._capture_scan(current, scan_angles, captures, poses)

        for _ in range(int(num_steps)):
            found_next = False
            chosen_yaw = None
            next_pos = None

            for _attempt in range(max_attempts_per_step):
                yaw = float(rng.uniform(0.0, 2.0 * np.pi))
                dx = np.sin(yaw) * step_size
                dz = np.cos(yaw) * step_size

                candidate = current.copy()
                candidate[0] = float(current[0] + dx)
                candidate[1] = float(current[1])
                candidate[2] = float(current[2] + dz)

                if not self.sim.pathfinder.is_navigable(candidate):
                    continue

                path = habitat_sim.ShortestPath()
                path.requested_start = current
                path.requested_end = candidate
                if not self.sim.pathfinder.find_path(path):
                    continue

                actual_step = float(np.linalg.norm(candidate - current))
                if abs(actual_step - step_size) > 1e-3:
                    continue

                found_next = True
                chosen_yaw = yaw
                next_pos = candidate
                break

            if not found_next or next_pos is None or chosen_yaw is None:
                break

            state = self.agent.get_state()
            state.position = next_pos.astype(np.float32)
            state.rotation = quaternion.from_rotation_vector([0.0, chosen_yaw, 0.0])
            self.agent.set_state(state)

            self._capture_scan(next_pos, scan_angles, captures, poses)
            current = next_pos

        return captures, poses

    def explore_custom_tour_split(
        self,
        num_steps=50,
        step_size=1.0,
        scan_angles=(0, 90, 180, 270),
        seed=None,
        max_attempts_per_step=32,
        include_start_scan=True,
    ):
        captures, poses = self.explore_custom_tour(
            num_steps=num_steps,
            step_size=step_size,
            scan_angles=scan_angles,
            seed=seed,
            max_attempts_per_step=max_attempts_per_step,
            include_start_scan=include_start_scan,
        )
        return self.split_modalities(captures), poses


ExplorerSemantic = SemanticExplorer

__all__ = ["SemanticExplorer", "ExplorerSemantic"]
