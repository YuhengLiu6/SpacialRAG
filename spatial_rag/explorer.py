# ===== explorer.py =====
import habitat_sim
import numpy as np
import quaternion
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
    def __init__(self, scene_path=SCENE_PATH):
        self.sim_settings = self._make_sim_settings(scene_path)
        self.cfg = self._make_hab_cfg(self.sim_settings)
        try:
            self.sim = habitat_sim.Simulator(self.cfg)
        except Exception as e:
            print(f"Error initializing Habitat Simulator: {e}")
            # Fallback for testing without actual scene file if needed, or raise
            raise e

        self._initialize_agent()

    def _make_sim_settings(self, scene_path):
        return {
            "scene": scene_path,
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "sensor_height": SENSOR_HEIGHT,
            "color_sensor": True,
        }

    def _make_hab_cfg(self, settings):
        sim_cfg = habitat_sim.SimulatorConfiguration()
        import torch
        if torch.cuda.is_available():
            sim_cfg.gpu_device_id = 0
        sim_cfg.scene_id = settings["scene"]

        # Define sensor
        sensor_specs = []
        if settings["color_sensor"]:
            color_sensor_spec = habitat_sim.CameraSensorSpec()
            color_sensor_spec.uuid = "color_sensor"
            color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            color_sensor_spec.resolution = [settings["height"], settings["width"]]
            color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
            color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(color_sensor_spec)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
            ),
        }

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def _initialize_agent(self):
        # Initialize agent at a navigable point
        self.agent = self.sim.initialize_agent(0)
        start_state = habitat_sim.AgentState()
        # Snap to navmesh if possible, otherwise use default
        try:
             # Basic random navigable point
             start_state.position = self.sim.pathfinder.get_random_navigable_point()
        except:
             pass 
        self.agent.set_state(start_state)

    def step_random(self):
        """
        Takes a random action and returns observation and state.
        """
        action = np.random.choice(["move_forward", "turn_left", "turn_right"])
        observations = self.sim.step(action)
        
        rgb = observations["color_sensor"]
        # Remove alpha channel if present
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
            
        state = self.agent.get_state()
        position = state.position # np.array [x, y, z]
        rotation = state.rotation # quaternion
        
        return rgb, position, rotation

    def close(self):
        self.sim.close()
