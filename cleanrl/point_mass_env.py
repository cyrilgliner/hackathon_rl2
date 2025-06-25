import gymnasium as gym
import numpy as np

class PointMassEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(PointMassEnv, self).__init__()
        
        # Change to discrete action space with 3 actions: left (-1), stay (0), right (1)
        self.action_space = gym.spaces.Discrete(3, start=0)
        
        # Observation: [position, velocity]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        
        # Explicitly define single action/observation spaces as properties
        # This ensures they're always accessible, even if the environment is wrapped
        
        # Physics parameters
        self.mass = 1.0  # kg
        self.dt = 0.02  # seconds per timestep
        self.max_position = 10.0  # out of bounds condition
        
        # Action mapping: 0 -> -1.0, 1 -> 0.0, 2 -> 1.0
        self.action_map = {0: -1.0, 1: 0.0, 2: 1.0}
        
        self.state = None
    
    @property
    def single_action_space(self):
        return self.action_space
        
    @property
    def single_observation_space(self):
        return self.observation_space

    def reset(self, seed=None, options=None):
        # Start at position 0 with velocity 0
        self.state = np.array([0.0, 0.0], dtype=np.float32)
        return self.state , {}

    def step(self, action):
        position, velocity = self.state
        
        # Map discrete action to force
        if action == 0:
            force = -1.0
        elif action == 1:
            force = 0  # Move left
        elif action == 2:
            force = 1.0
        else:
            raise ValueError("Invalid action")
        
        # Newton's second law: F = m * a -> a = F / m
        acceleration = force / self.mass
        
        # Update state
        velocity += acceleration * self.dt
        position += velocity * self.dt
        self.state = np.array([position, velocity], dtype=np.float32)
        
        # Define reward and done condition
        reward = -abs(position)  # Encourage staying near zero
        done = abs(position) > self.max_position  # Terminate if too far
        
        truncation = False  # No truncation condition in this simple environment
        return self.state, reward, done, truncation, {}

    def render(self, mode='human'):
        # Simple print for now
        position, _ = self.state
        print(f"Position: {position:.2f}")

    def close(self):
        pass
