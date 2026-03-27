import habitat_sim
import numpy as np
import quaternion
import cv2
import magnum as mn
from collections import deque
from spatial_rag.config import (
    SCENE_PATH,
    AGENT_HEIGHT,
    AGENT_RADIUS,
    SENSOR_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    FOV,
)


class Explorer:
    OVERLAY_FLOOR_HEIGHT = 0.20991861820220947

    def __init__(self, scene_path=SCENE_PATH):
        self.sim_settings = self._make_sim_settings(scene_path)
        self.cfg = self._make_hab_cfg(self.sim_settings)
        self._last_top_down_projection = None
        self._last_center_highest_config = None
        self._last_tour_profile = None
        self._default_floor_height = None

        try:
            self.sim = habitat_sim.Simulator(self.cfg)
        except Exception as e:
            print(f"Error initializing Habitat Simulator: {e}")
            raise e

        self._initialize_agent()

    def get_overlay_floor_height(self):
        """
        Return the canonical floor height used by overlay-style top-down rendering.
        """
        # Aligned with recent query overlays where actual_world_position[1] was stable.
        return float(self.OVERLAY_FLOOR_HEIGHT)

    def _make_sim_settings(self, scene_path):
        return {
            "scene": scene_path,
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "sensor_height": SENSOR_HEIGHT,
            "color_sensor": True,
            "fov": FOV,
        }

    def _make_hab_cfg(self, settings):
        sim_cfg = habitat_sim.SimulatorConfiguration()
        import torch

        if torch.cuda.is_available():
            sim_cfg.gpu_device_id = 0

        sim_cfg.scene_id = settings["scene"]

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
        # top_down_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        top_down_spec.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
        top_down_spec.resolution = [2048, 2048]
        top_down_spec.position = [0.0, 20.0, 0.0]
        top_down_spec.orientation = [-np.pi / 2, 0.0, 0.0]
        # Set an initial value; render_true_floor_plan() updates this per-scene.
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

    def _initialize_agent(self):
        self.agent = self.sim.initialize_agent(0)
        start_state = habitat_sim.AgentState()

        try:
            bounds = self.sim.pathfinder.get_bounds()
            min_x, max_x = bounds[0][0], bounds[1][0]
            min_z, max_z = bounds[0][2], bounds[1][2]
            y = bounds[0][1]

            found = False
            for x in np.arange(min_x, max_x, 0.5):
                for z in np.arange(min_z, max_z, 0.5):
                    pt = np.array([x, y, z], dtype=np.float32)
                    if self.sim.pathfinder.is_navigable(pt):
                        start_state.position = pt
                        found = True
                        break
                if found:
                    break
        except Exception:
            pass

        self.agent.set_state(start_state)
        try:
            self._default_floor_height = float(self.agent.get_state().position[1])
        except Exception:
            self._default_floor_height = None

    def _extract_connected_components(self, navigable_cells):
        """
        Split grid cells into 4-neighbor connected components.
        """
        unvisited = set(navigable_cells.keys())
        components = []

        while unvisited:
            seed = unvisited.pop()
            stack = [seed]
            comp = [seed]

            while stack:
                ix, iz = stack.pop()
                for nb in (
                    (ix + 1, iz),
                    (ix - 1, iz),
                    (ix, iz + 1),
                    (ix, iz - 1),
                ):
                    if nb in unvisited:
                        unvisited.remove(nb)
                        stack.append(nb)
                        comp.append(nb)

            components.append(comp)

        return components

    def _build_component_sweep(self, component_keys, navigable_cells):
        """
        Build a boustrophedon sweep order for one connected region.
        """
        xs = [k[0] for k in component_keys]
        zs = [k[1] for k in component_keys]
        span_x = max(xs) - min(xs)
        span_z = max(zs) - min(zs)

        ordered_keys = []
        if span_x >= span_z:
            by_z = {}
            for ix, iz in component_keys:
                by_z.setdefault(iz, []).append(ix)
            for row_idx, iz in enumerate(sorted(by_z.keys())):
                row = sorted(by_z[iz], reverse=(row_idx % 2 == 1))
                ordered_keys.extend([(ix, iz) for ix in row])
        else:
            by_x = {}
            for ix, iz in component_keys:
                by_x.setdefault(ix, []).append(iz)
            for col_idx, ix in enumerate(sorted(by_x.keys())):
                col = sorted(by_x[ix], reverse=(col_idx % 2 == 1))
                ordered_keys.extend([(ix, iz) for iz in col])

        return [navigable_cells[k] for k in ordered_keys]

    def _neighbor_keys(self, key):
        ix, iz = key
        return (
            (ix + 1, iz),
            (ix - 1, iz),
            (ix, iz + 1),
            (ix, iz - 1),
        )

    def _find_best_entry_for_component(
        self,
        current_pos,
        component_keys,
        navigable_cells,
        max_candidates=12,
    ):
        """
        Pick a reachable entry key for a component using geodesic cost.
        """
        current = np.array(current_pos, dtype=np.float32)
        ranked = sorted(
            component_keys,
            key=lambda k: float(np.linalg.norm(navigable_cells[k] - current)),
        )
        ranked = ranked[: max(1, min(max_candidates, len(ranked)))]

        best_key = ranked[0]
        best_dist = float("inf")
        best_path_points = None
        for k in ranked:
            path = habitat_sim.ShortestPath()
            path.requested_start = current
            path.requested_end = navigable_cells[k]
            if self.sim.pathfinder.find_path(path):
                dist = float(path.geodesic_distance)
                if dist < best_dist:
                    best_dist = dist
                    best_key = k
                    best_path_points = [np.array(p, dtype=np.float32) for p in path.points]
        return best_key, best_dist, best_path_points

    def _path_to_nearest_unvisited(self, start_key, unvisited, component_set):
        """
        BFS in grid topology to find a shortest path from start_key to
        the nearest unvisited key.
        """
        if not unvisited:
            return None

        q = deque([start_key])
        parent = {start_key: None}

        while q:
            key = q.popleft()
            if key in unvisited:
                # Reconstruct path start_key -> key
                path = [key]
                cur = key
                while parent[cur] is not None:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                return path
            for nb in self._neighbor_keys(key):
                if nb in component_set and nb not in parent:
                    parent[nb] = key
                    q.append(nb)

        # Fallback: should rarely happen for a true connected component.
        fallback = next(iter(unvisited))
        return [start_key, fallback]

    def _build_component_local_order(self, component_keys, entry_key):
        """
        Build a locally continuous visit order by repeatedly selecting nearest
        unvisited cell in grid-graph distance.
        """
        component_set = set(component_keys)
        if entry_key not in component_set:
            entry_key = next(iter(component_set))

        order = [entry_key]
        unvisited = set(component_set)
        unvisited.discard(entry_key)
        current = entry_key

        while unvisited:
            path = self._path_to_nearest_unvisited(current, unvisited, component_set)
            if not path:
                break
            for key in path[1:]:
                order.append(key)
                unvisited.discard(key)
            current = order[-1]

        return order

    def _sample_points_along_path(self, path_points, sample_step):
        """
        Downsample path points with approximately fixed spacing.
        """
        if not path_points:
            return []
        sample_step = max(float(sample_step), 1e-3)
        sampled = [np.array(path_points[0], dtype=np.float32)]
        dist_from_last_sample = 0.0
        seg_start = np.array(path_points[0], dtype=np.float32)

        for p in path_points[1:]:
            seg_end = np.array(p, dtype=np.float32)
            seg_vec = seg_end - seg_start
            seg_len = float(np.linalg.norm(seg_vec))
            if seg_len < 1e-8:
                seg_start = seg_end
                continue

            while dist_from_last_sample + seg_len >= sample_step:
                t = (sample_step - dist_from_last_sample) / seg_len
                new_point = seg_start + seg_vec * t
                sampled.append(new_point.astype(np.float32))
                seg_start = new_point
                seg_vec = seg_end - seg_start
                seg_len = float(np.linalg.norm(seg_vec))
                dist_from_last_sample = 0.0
                if seg_len < 1e-8:
                    break

            dist_from_last_sample += seg_len
            seg_start = seg_end

        final_point = np.array(path_points[-1], dtype=np.float32)
        if np.linalg.norm(sampled[-1] - final_point) > 1e-5:
            sampled.append(final_point)
        return sampled

    def _plan_room_tour_waypoints(self, start_pos, meters_per_step):
        """
        Plan a room-tour path: sweep each connected region and visit nearby regions first.
        """
        bounds = self.sim.pathfinder.get_bounds()
        min_x, max_x = float(bounds[0][0]), float(bounds[1][0])
        min_z, max_z = float(bounds[0][2]), float(bounds[1][2])
        agent_y = float(start_pos[1])

        xs = np.arange(min_x, max_x, meters_per_step)
        zs = np.arange(min_z, max_z, meters_per_step)

        navigable_cells = {}
        for ix, x in enumerate(xs):
            for iz, z in enumerate(zs):
                pt = np.array([x, agent_y, z], dtype=np.float32)
                if self.sim.pathfinder.is_navigable(pt):
                    navigable_cells[(ix, iz)] = pt

        if not navigable_cells:
            return []

        components = self._extract_connected_components(navigable_cells)
        remaining_components = [set(comp) for comp in components]

        ordered_waypoints = []
        current = np.array(start_pos, dtype=np.float32)
        while remaining_components:
            best_idx = None
            best_entry = None
            best_transition = None
            best_dist = float("inf")
            has_reachable = False

            for idx, comp_keys in enumerate(remaining_components):
                entry_key, dist, path_points = self._find_best_entry_for_component(
                    current,
                    list(comp_keys),
                    navigable_cells,
                )
                if np.isfinite(dist):
                    has_reachable = True
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
                    best_entry = entry_key
                    best_transition = path_points

            if best_idx is None or best_entry is None:
                break
            if not has_reachable:
                # Remaining components are not reachable from current tour island.
                break

            selected = remaining_components.pop(best_idx)
            # Add transition waypoints so scan poses do not visually jump across regions.
            connector = self._sample_points_along_path(best_transition, meters_per_step)
            if connector:
                if ordered_waypoints and np.linalg.norm(connector[0] - ordered_waypoints[-1]) < 1e-5:
                    connector = connector[1:]
                ordered_waypoints.extend(connector)
            local_order = self._build_component_local_order(list(selected), best_entry)
            ordered_waypoints.extend([navigable_cells[k] for k in local_order])
            current = np.array(navigable_cells[local_order[-1]], dtype=np.float32)

        return ordered_waypoints

    def _choose_scene_tour_profile(self):
        """
        Choose adaptive tour parameters from scene geometry and navigable coverage.
        """
        bounds = self.sim.pathfinder.get_bounds()
        dx = float(bounds[1][0] - bounds[0][0])
        dz = float(bounds[1][2] - bounds[0][2])
        scene_span = max(dx, dz)
        bbox_area = max(dx * dz, 1e-3)

        try:
            nav_area = float(self.sim.pathfinder.navigable_area)
        except Exception:
            nav_area = bbox_area * 0.5
        nav_density = float(np.clip(nav_area / bbox_area, 0.0, 1.0))

        if scene_span < 12.0:
            meters_per_step = 0.9
        elif scene_span < 25.0:
            meters_per_step = 1.2
        elif scene_span < 40.0:
            meters_per_step = 1.6
        else:
            meters_per_step = 2.0

        if nav_density < 0.25:
            meters_per_step *= 0.85
        elif nav_density > 0.60:
            meters_per_step *= 1.10
        meters_per_step = float(np.clip(meters_per_step, 0.7, 2.3))

        walk_step_m = float(np.clip(meters_per_step / 3.0, 0.3, 0.8))
        # scan_angles = (0,30,60,90,120,150,180,210,240,270,300,330)
        scan_angles = (0,90,180,270)


        profile = {
            "scene_span": scene_span,
            "nav_density": nav_density,
            "meters_per_step": meters_per_step,
            "walk_step_m": walk_step_m,
            "scan_angles": scan_angles,
        }
        self._last_tour_profile = profile
        return profile

    def get_last_tour_profile(self):
        """
        Returns the latest auto-selected tour profile.
        """
        return self._last_tour_profile

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

        frames = []
        poses = []
        walk_step_m = max(float(walk_step_m), 0.1)

        def capture_360_scan(position):
            state = self.agent.get_state()
            state.position = position

            for angle_deg in scan_angles:
                yaw = np.deg2rad(angle_deg)
                state.rotation = quaternion.from_rotation_vector(
                    [0, yaw, 0]
                )
                self.agent.set_state(state)

                obs = self.sim.get_sensor_observations()
                rgb = obs["color_sensor"]

                if rgb.shape[2] == 4:
                    rgb = rgb[:, :, :3]

                frames.append(rgb)
                poses.append(
                    {
                        "position": position.copy(),
                        "rotation": state.rotation,
                    }
                )

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

                capture_360_scan(target)
                current_pos = target

        return frames, poses

    def explore_custom_tour(
        self,
        num_steps=50,
        step_size=1.0,
        scan_angles=(0, 90, 180, 270),
        seed=None,
        max_attempts_per_step=32,
        include_start_scan=True,
    ):
        """
        Explore with a fixed-step random route.

        Args:
            num_steps: number of movement steps to attempt.
            step_size: fixed Euclidean distance (meters) between consecutive waypoints.
            scan_angles: camera yaw angles (degrees) captured at each accepted waypoint.
            seed: RNG seed for reproducible random routes.
            max_attempts_per_step: attempts to find a valid next waypoint for each step.
            include_start_scan: whether to capture scan at the initial position.

        Returns:
            frames, poses (same format as explore_full_house).
        """
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

        frames = []
        poses = []

        def capture_scan(position):
            state = self.agent.get_state()
            state.position = np.array(position, dtype=np.float32)

            for angle_deg in scan_angles:
                yaw = np.deg2rad(float(angle_deg))
                state.rotation = quaternion.from_rotation_vector([0.0, yaw, 0.0])
                self.agent.set_state(state)

                obs = self.sim.get_sensor_observations()
                rgb = obs["color_sensor"]
                if rgb.shape[2] == 4:
                    rgb = rgb[:, :, :3]

                frames.append(rgb)
                poses.append(
                    {
                        "position": state.position.copy(),
                        "rotation": state.rotation,
                    }
                )

        start_state = self.agent.get_state()
        current = np.array(start_state.position, dtype=np.float32)

        if include_start_scan:
            capture_scan(current)

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

                # Keep the route fixed-step in Euclidean distance.
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

            capture_scan(next_pos)
            current = next_pos

        return frames, poses

    def render_true_floor_plan(self, floor_height=None):
        """
        Generates a stable top-down floor plan from the navmesh occupancy view.
        """
        bounds = self.sim.pathfinder.get_bounds()
        min_x, max_x = bounds[0][0], bounds[1][0]
        min_z, max_z = bounds[0][2], bounds[1][2]
        size_x = float(max_x - min_x)
        size_z = float(max_z - min_z)
        max_size = max(size_x, size_z)
        target_pixels = 2048.0
        meters_per_pixel = max(max_size / target_pixels, 0.01)

        # Default to the canonical overlay floor height unless caller explicitly overrides.
        if floor_height is None:
            floor_height = self.get_overlay_floor_height()
        else:
            floor_height = float(floor_height)
        topdown_bool = self.sim.pathfinder.get_topdown_view(meters_per_pixel, floor_height)
        topdown_u8 = np.where(topdown_bool, 235, 90).astype(np.uint8)
        img_bgr = cv2.cvtColor(topdown_u8, cv2.COLOR_GRAY2BGR)

        self._last_top_down_projection = {
            "view_min_x": float(min_x),
            "view_max_x": float(max_x),
            "view_min_z": float(min_z),
            "view_max_z": float(max_z),
        }
        return img_bgr

    def render_true_floor_plan_with_trajectory(self, poses):
        """
        Convenience API: render true floor plan and overlay trajectory on it.
        """
        return self.draw_trajectory_on_floor_plan(None, poses)

    def draw_trajectory_on_true_floor_plan(self, poses):
        """
        Alias of render_true_floor_plan_with_trajectory for readability.
        """
        return self.render_true_floor_plan_with_trajectory(poses)

    def render_textured_floor_plan(
        self,
        meters_per_pixel=0.03,
        sample_spacing=0.25,
    ):
        """
        Builds a textured top-down map by projecting downward camera patches onto the navmesh map.
        """
        bounds = self.sim.pathfinder.get_bounds()
        min_x, max_x = float(bounds[0][0]), float(bounds[1][0])
        min_z, max_z = float(bounds[0][2]), float(bounds[1][2])
        floor_height = float(self.agent.get_state().position[1])

        topdown_bool = self.sim.pathfinder.get_topdown_view(
            meters_per_pixel,
            floor_height,
        )
        map_h, map_w = topdown_bool.shape

        color_sum = np.zeros((map_h * map_w, 3), dtype=np.float32)
        weight = np.zeros((map_h * map_w,), dtype=np.float32)

        prev_state = self.agent.get_state()
        state = self.agent.get_state()
        state.rotation = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)

        denom_x = max(max_x - min_x, 1e-6)
        denom_z = max(max_z - min_z, 1e-6)

        step_px = max(1, int(round(sample_spacing / meters_per_pixel)))
        sample_pixels = []
        for pz in range(0, map_h, step_px):
            row = topdown_bool[pz]
            for px in range(0, map_w, step_px):
                if row[px]:
                    sample_pixels.append((px, pz))

        if not sample_pixels:
            self.agent.set_state(prev_state)
            fallback = np.where(topdown_bool[..., None], 180, 70).astype(np.uint8)
            self._last_top_down_projection = {
                "view_min_x": min_x,
                "view_max_x": max_x,
                "view_min_z": min_z,
                "view_max_z": max_z,
            }
            return fallback

        first_px, first_pz = sample_pixels[0]
        first_x = min_x + (first_px / max(map_w - 1, 1)) * denom_x
        first_z = min_z + (first_pz / max(map_h - 1, 1)) * denom_z
        state.position = np.array([first_x, floor_height, first_z], dtype=np.float32)
        self.agent.set_state(state)
        first_obs = self.sim.get_sensor_observations()
        first_down = first_obs["down_sensor"]
        if first_down.shape[2] == 4:
            first_down = first_down[:, :, :3]
        down_h, down_w = first_down.shape[:2]

        down_sensor = self.agent._sensors["down_sensor"]
        down_spec = down_sensor.specification()
        hfov_rad = np.deg2rad(float(down_spec.hfov))
        vfov_rad = 2.0 * np.arctan(np.tan(hfov_rad / 2.0) * (down_h / max(down_w, 1)))
        fx = (down_w / 2.0) / np.tan(hfov_rad / 2.0)
        fy = (down_h / 2.0) / np.tan(vfov_rad / 2.0)
        cx = (down_w - 1) / 2.0
        cy = (down_h - 1) / 2.0
        sensor_h = float(down_spec.position[1])

        patch_half = max(10, min(down_h, down_w) // 4)
        u0 = int(max(0, np.floor(cx) - patch_half))
        u1 = int(min(down_w, np.floor(cx) + patch_half))
        v0 = int(max(0, np.floor(cy) - patch_half))
        v1 = int(min(down_h, np.floor(cy) + patch_half))
        u_idx = np.arange(u0, u1, dtype=np.float32)
        v_idx = np.arange(v0, v1, dtype=np.float32)
        uu, vv = np.meshgrid(u_idx, v_idx)

        u_off = (uu - cx) / fx * sensor_h
        v_off = (vv - cy) / fy * sensor_h
        dx = u_off
        dz = v_off

        for px, pz in sample_pixels:
            x = min_x + (px / max(map_w - 1, 1)) * denom_x
            z = min_z + (pz / max(map_h - 1, 1)) * denom_z

            pos = np.array([x, floor_height, z], dtype=np.float32)
            if not self.sim.pathfinder.is_navigable(pos):
                continue

            state.position = pos
            self.agent.set_state(state)
            obs = self.sim.get_sensor_observations()
            down = obs["down_sensor"]
            if down.shape[2] == 4:
                down = down[:, :, :3]
            down_bgr = cv2.cvtColor(down, cv2.COLOR_RGB2BGR)
            patch = down_bgr[v0:v1, u0:u1]
            if patch.size == 0:
                continue

            wx = x + dx
            wz = z + dz
            mpx = ((wx - min_x) / denom_x * (map_w - 1)).astype(np.int32)
            mpz = ((wz - min_z) / denom_z * (map_h - 1)).astype(np.int32)

            in_bounds = (
                (mpx >= 0)
                & (mpx < map_w)
                & (mpz >= 0)
                & (mpz < map_h)
            )
            if not np.any(in_bounds):
                continue

            mpx_v = mpx[in_bounds]
            mpz_v = mpz[in_bounds]
            nav_ok = topdown_bool[mpz_v, mpx_v]
            if not np.any(nav_ok):
                continue

            flat_idx = (mpz_v[nav_ok] * map_w + mpx_v[nav_ok]).astype(np.int32)
            colors = patch[in_bounds][nav_ok].astype(np.float32)
            np.add.at(color_sum[:, 0], flat_idx, colors[:, 0])
            np.add.at(color_sum[:, 1], flat_idx, colors[:, 1])
            np.add.at(color_sum[:, 2], flat_idx, colors[:, 2])
            np.add.at(weight, flat_idx, 1.0)

        self.agent.set_state(prev_state)

        textured = np.zeros((map_h * map_w, 3), dtype=np.uint8)
        valid = weight > 0
        if np.any(valid):
            textured[valid] = np.clip(color_sum[valid] / weight[valid, None], 0, 255).astype(np.uint8)
        textured = textured.reshape(map_h, map_w, 3)

        coverage = valid.reshape(map_h, map_w)
        missing_inside = topdown_bool & (~coverage)
        if np.any(missing_inside):
            textured = cv2.inpaint(
                textured,
                (missing_inside.astype(np.uint8) * 255),
                3,
                cv2.INPAINT_TELEA,
            )
        smoothed = cv2.GaussianBlur(textured, (0, 0), sigmaX=0.8, sigmaY=0.8)
        textured[topdown_bool] = smoothed[topdown_bool]

        background = np.full((map_h, map_w, 3), 70, dtype=np.uint8)
        background[topdown_bool] = textured[topdown_bool]

        self._last_top_down_projection = {
            "view_min_x": min_x,
            "view_max_x": max_x,
            "view_min_z": min_z,
            "view_max_z": max_z,
        }
        return background

    def _get_center_highest_pose(self):
        bounds = self.sim.pathfinder.get_bounds()
        center_x = float((bounds[0][0] + bounds[1][0]) / 2.0)
        center_z = float((bounds[0][2] + bounds[1][2]) / 2.0)
        highest_y = float(bounds[1][1])
        return center_x, highest_y, center_z

    def _capture_center_highest_view(self, hfov=120.0):
        prev_state = self.agent.get_state()
        state = self.agent.get_state()

        center_x, highest_y, center_z = self._get_center_highest_pose()
        state.position = np.array([center_x, highest_y, center_z], dtype=np.float32)
        state.rotation = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
        self.agent.set_state(state)

        sensor = self.agent._sensors["center_top_view"]
        spec = sensor.specification()
        spec.hfov = float(hfov)
        sensor.set_projection_params(spec)

        obs = self.sim.get_sensor_observations()
        img = obs["center_top_view"]
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        render_camera = sensor.render_camera
        camera_matrix = render_camera.camera_matrix
        projection_matrix = render_camera.projection_matrix

        self._last_center_highest_config = {
            "center_x": center_x,
            "highest_y": highest_y,
            "center_z": center_z,
            "hfov": float(hfov),
        }
        self.agent.set_state(prev_state)
        return img_bgr, camera_matrix, projection_matrix

    def render_center_highest_view(self, hfov=120.0):
        """
        Renders a top-looking image from scene center at maximum scene height.
        """
        img_bgr, _, _ = self._capture_center_highest_view(hfov=hfov)
        return img_bgr

    def draw_trajectory_on_center_highest_view(
        self,
        center_view,
        poses,
        hfov=120.0,
    ):
        """
        Projects trajectory points into the center-highest camera image.
        """
        map_vis = center_view.copy()
        height, width = map_vis.shape[:2]
        _, camera_matrix, projection_matrix = self._capture_center_highest_view(hfov=hfov)

        def project_world_to_pixel(world_xyz):
            cam = camera_matrix.transform_point(
                mn.Vector3(float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2]))
            )
            if cam[2] >= 0:
                return None
            ndc = projection_matrix.transform_point(cam)
            px = int(round((float(ndc[0]) * 0.5 + 0.5) * (width - 1)))
            py = int(round((1.0 - (float(ndc[1]) * 0.5 + 0.5)) * (height - 1)))
            if 0 <= px < width and 0 <= py < height:
                return (px, py)
            return None

        points = []
        arrows = []
        for pose in poses:
            pos = pose["position"]
            head_xz = self._rotation_to_forward_xz(pose.get("rotation"))
            p0 = project_world_to_pixel(pos)
            if p0 is None:
                continue
            points.append(p0)

            if head_xz is not None:
                # Match floor-map style with a fixed pixel arrow length.
                probe_len = 0.6
                probe_world = np.array(
                    [pos[0] + head_xz[0] * probe_len, pos[1], pos[2] + head_xz[1] * probe_len],
                    dtype=np.float32,
                )
                p_probe = project_world_to_pixel(probe_world)
                if p_probe is not None:
                    vec = np.array([p_probe[0] - p0[0], p_probe[1] - p0[1]], dtype=np.float32)
                    vec_norm = float(np.linalg.norm(vec))
                    if vec_norm > 1e-6:
                        vec = vec / vec_norm * 28.0
                        tip = (
                            int(np.clip(p0[0] + vec[0], 0, width - 1)),
                            int(np.clip(p0[1] + vec[1], 0, height - 1)),
                        )
                        arrows.append((p0, tip))

        self._draw_stepwise_trajectory(
            map_vis,
            points,
            arrows=arrows,
            line_color=(255, 0, 0),
            line_thickness=3,
            step_color=(255, 255, 0),
            step_radius=7,
            start_color=(0, 255, 0),
            end_color=(0, 0, 255),
            endpoint_radius=12,
            arrow_thickness=2,
            arrow_tip_length=0.60,
        )

        return map_vis

    def render_center_highest_view_with_trajectory(self, poses, hfov=120.0):
        """
        Convenience API: render center-highest image and overlay trajectory.
        """
        base = self.render_center_highest_view(hfov=hfov)
        return self.draw_trajectory_on_center_highest_view(base, poses, hfov=hfov)

    def _draw_stepwise_trajectory(
        self,
        image,
        points,
        arrows=None,
        line_color=(255, 0, 0),
        line_thickness=3,
        step_color=(255, 255, 0),
        step_radius=7,
        start_color=(0, 255, 0),
        end_color=(0, 0, 255),
        endpoint_radius=12,
        arrow_color=(0, 165, 255),
        arrow_thickness=1,
        arrow_tip_length=0.20,
    ):
        if len(points) > 1:
            for i in range(1, len(points)):
                cv2.line(image, points[i - 1], points[i], line_color, line_thickness)

        if arrows:
            for p0, p1 in arrows:
                cv2.arrowedLine(
                    image,
                    p0,
                    p1,
                    arrow_color,
                    arrow_thickness,
                    tipLength=arrow_tip_length,
                )

        for p in points:
            cv2.circle(image, p, step_radius, step_color, -1)

        if points:
            cv2.circle(image, points[0], endpoint_radius, start_color, -1)
            cv2.circle(image, points[-1], endpoint_radius, end_color, -1)

    def _rotation_to_forward_xz(self, rotation):
        if rotation is None:
            return None
        try:
            if isinstance(rotation, quaternion.quaternion):
                q = rotation
            elif hasattr(rotation, "w") and hasattr(rotation, "x") and hasattr(rotation, "y") and hasattr(rotation, "z"):
                q = quaternion.quaternion(rotation.w, rotation.x, rotation.y, rotation.z)
            elif isinstance(rotation, (list, tuple, np.ndarray)) and len(rotation) == 4:
                q = quaternion.quaternion(
                    float(rotation[0]),
                    float(rotation[1]),
                    float(rotation[2]),
                    float(rotation[3]),
                )
            else:
                return None

            rot_m = quaternion.as_rotation_matrix(q)
            forward = rot_m.dot(np.array([0.0, 0.0, 1.0], dtype=np.float32))
            xz = np.array([forward[0], forward[2]], dtype=np.float32)
            norm = float(np.linalg.norm(xz))
            if norm < 1e-6:
                return None
            xz /= norm
            return float(xz[0]), float(xz[1])
        except Exception:
            return None

    def _select_best_floor_height_for_poses(self, poses, num_samples=41, margin=2.0):
        """
        Pick a floor height that best matches the given trajectory poses.
        Strategy:
        - Sample candidate heights around the pose median Y.
        - Prefer the height where most trajectory points are navigable.
        - Break ties by larger top-down navigable area.
        """
        if not poses:
            return self.get_overlay_floor_height()

        ys = []
        xzs = []
        for pose in poses:
            pos = pose.get("position")
            if pos is None or len(pos) < 3:
                continue
            x = float(pos[0])
            y = float(pos[1])
            z = float(pos[2])
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                ys.append(y)
                xzs.append((x, z))

        if not ys or not xzs:
            return self.get_overlay_floor_height()

        bounds = self.sim.pathfinder.get_bounds()
        y_min = float(bounds[0][1])
        y_max = float(bounds[1][1])
        y_center = float(np.median(np.array(ys, dtype=np.float32)))
        lo = max(y_min, y_center - float(margin))
        hi = min(y_max, y_center + float(margin))
        if hi <= lo:
            return y_center

        candidates = np.linspace(lo, hi, int(max(3, num_samples)))
        evaluated = []

        for h in candidates:
            h = float(h)
            hit = 0
            for x, z in xzs:
                pt = np.array([x, h, z], dtype=np.float32)
                if self.sim.pathfinder.is_navigable(pt):
                    hit += 1

            try:
                top = self.sim.pathfinder.get_topdown_view(0.05, h)
                area = int(np.count_nonzero(top))
            except Exception:
                area = 0

            hit_ratio = float(hit) / float(len(xzs))
            evaluated.append((hit_ratio, area, h))

        # Sort by hit ratio first, then area.
        evaluated.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return float(evaluated[0][2]) if evaluated else y_center

    def draw_trajectory_on_floor_plan(self, floor_map, poses, hfov=120.0, crop_to_trajectory=False):
        """
        Draw trajectory on a floor-like base map.

        Behavior:
        - If floor_map is provided: draw on it directly (backward compatible).
        - If floor_map is None: use query-overlay-like base selection:
          1) choose a suitable floor height from poses, then render_true_floor_plan()
          2) fallback to center_highest_view and project trajectory there
        """
        if floor_map is None:
            auto_floor_height = self._select_best_floor_height_for_poses(poses)

            try:
                floor_map = self.render_true_floor_plan(floor_height=auto_floor_height)
            except Exception:
                base = self.render_center_highest_view(hfov=hfov)
                return self.draw_trajectory_on_center_highest_view(base, poses, hfov=hfov)

        bounds = self.sim.pathfinder.get_bounds()
        min_x, max_x = bounds[0][0], bounds[1][0]
        min_z, max_z = bounds[0][2], bounds[1][2]

        map_vis = floor_map.copy()
        height, width = map_vis.shape[:2]

        projection = self._last_top_down_projection
        if projection is not None and "view_min_x" in projection:
            view_min_x = projection["view_min_x"]
            view_max_x = projection["view_max_x"]
            view_min_z = projection["view_min_z"]
            view_max_z = projection["view_max_z"]
        elif projection is not None:
            half = projection["ortho_scale"] / 2.0
            view_min_x = projection["center_x"] - half
            view_max_x = projection["center_x"] + half
            view_min_z = projection["center_z"] - half
            view_max_z = projection["center_z"] + half
        else:
            view_min_x, view_max_x = min_x, max_x
            view_min_z, view_max_z = min_z, max_z

        denom_x = max(view_max_x - view_min_x, 1e-6)
        denom_z = max(view_max_z - view_min_z, 1e-6)

        points = []
        arrows = []
        for pose in poses:
            pos = pose["position"]
            x, z = pos[0], pos[2]

            px = int((x - view_min_x) / denom_x * width)
            pz = int((z - view_min_z) / denom_z * height)
            px = int(np.clip(px, 0, width - 1))
            pz = int(np.clip(pz, 0, height - 1))

            points.append((px, pz))

            head_xz = self._rotation_to_forward_xz(pose.get("rotation"))
            if head_xz is not None:
                vec = np.array(
                    [
                        head_xz[0] / denom_x * width,
                        head_xz[1] / denom_z * height,
                    ],
                    dtype=np.float32,
                )
                vec_norm = float(np.linalg.norm(vec))
                if vec_norm > 1e-6:
                    vec = vec / vec_norm * 28.0
                    tip = (
                        int(np.clip(px + vec[0], 0, width - 1)),
                        int(np.clip(pz + vec[1], 0, height - 1)),
                    )
                    arrows.append(((px, pz), tip))

        self._draw_stepwise_trajectory(
            map_vis,
            points,
            arrows=arrows,
            line_color=(255, 0, 0),
            line_thickness=3,
            step_color=(255, 255, 0),
            step_radius=7,
            start_color=(0, 255, 0),
            end_color=(0, 0, 255),
            endpoint_radius=12,
            arrow_thickness=2,
            arrow_tip_length=0.60,
        )

        if crop_to_trajectory and points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_min = min(xs)
            x_max = max(xs)
            y_min = min(ys)
            y_max = max(ys)

            h, w = map_vis.shape[:2]
            pad = max(40, int(0.08 * max(h, w)))
            x0 = int(np.clip(x_min - pad, 0, w - 1))
            x1 = int(np.clip(x_max + pad, 0, w - 1))
            y0 = int(np.clip(y_min - pad, 0, h - 1))
            y1 = int(np.clip(y_max + pad, 0, h - 1))

            if x1 > x0 and y1 > y0:
                map_vis = map_vis[y0 : y1 + 1, x0 : x1 + 1]

        return map_vis

    def close(self):
        self.sim.close()
