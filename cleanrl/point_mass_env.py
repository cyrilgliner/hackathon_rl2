import gymnasium as gym
import numpy as np
import pygame
from pygame import gfxdraw
import os
import cv2

class PointMassEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, render_mode=None):
        super(PointMassEnv, self).__init__()

        self.counter = 0
        
        # Change to discrete action space with 3 actions: left (-1), stay (0), right (1)
        self.action_space = gym.spaces.Discrete(3, start=0)
        
        # Observation: [position, velocity]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        
        # Action history for plotting
        self.action_history = []
        self.max_history_length = 100  # Keep last 100 actions for plotting
        self.last_action = None
        
        # Physics parameters
        self.mass = 1.0  # kg
        self.dt = 0.2  # seconds per timestep
        self.max_position = 10.0  # out of bounds condition
        
        # Action mapping: 0 -> -1.0, 1 -> 0.0, 2 -> 1.0
        self.action_map = {0: -1.0, 1: 0.0, 2: 1.0}
        
        # Rendering-related attributes
        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        
        # Video recording attributes
        self.video_path = None
        self.video_writer = None
        self.record_video = True
        self.fps = 30
        
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
        self.counter = 0
        
        # Reset action history
        self.action_history = []
        self.last_action = None
        
        # Set up video writer at the beginning of each episode
        if self.record_video:
            # Create a videos directory if it doesn't exist
            os.makedirs('videos/point_mass', exist_ok=True)
            
            # Create a unique filename for the video
            import time
            timestamp = int(time.time())
            self.video_path = f'videos/point_mass/point_mass_episode_{timestamp}.mp4'
            
            # Create the video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.video_path, 
                fourcc, 
                self.fps, 
                (self.screen_width, self.screen_height)
            )
        
        return self.state, {}

    def step(self, action):
        position, velocity = self.state

        self.counter += 1
        
        # Store the action for plotting
        self.last_action = action
        self.action_history.append(action)
        if len(self.action_history) > self.max_history_length:
            self.action_history.pop(0)  # Remove oldest action if history is too long
        
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
        reward = - 1 # Penalize each step to encourage faster solutions

        reward += 1/max(0.1, (self.max_position - position)) * 2    # Reward for staying within bounds

        done = position > self.max_position  # Terminate if too far

        if done:
            reward = 1000

        if self.counter >= 500:
            truncation = True
        else:
            truncation = False  # No truncation condition in this simple environment

        # Close video at end of episode if done
        if done or truncation:
            self._close_video()
            
        return self.state, reward, done, truncation, {}

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            
            # Always create a surface for rendering
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()
        
        self.screen.fill((255, 255, 255))
        
        # Draw the track
        pygame.draw.line(
            self.screen,
            (0, 0, 0),
            (0, self.screen_height / 2),
            (self.screen_width, self.screen_height / 2),
            1
        )
        
        # Draw the boundaries
        left_boundary = 0
        right_boundary = self.screen_width
        for x in [left_boundary, right_boundary]:
            pygame.draw.line(
                self.screen,
                (255, 0, 0),  # Red line
                (x, self.screen_height / 2 - 20),
                (x, self.screen_height / 2 + 20),
                2
            )
        
        # Draw the center position (goal)
        center_x = self.screen_width / 2
        pygame.draw.line(
            self.screen,
            (0, 255, 0),  # Green line
            (center_x, self.screen_height / 2 - 20),
            (center_x, self.screen_height / 2 + 20),
            2
        )
        
        # Draw the point mass
        position, velocity = self.state
        cart_x = (position + self.max_position) * (self.screen_width / (2 * self.max_position))
        cart_x = np.clip(cart_x, 10, self.screen_width - 10)
        
        gfxdraw.filled_circle(
            self.screen,
            int(cart_x),
            int(self.screen_height / 2),
            10,
            (0, 0, 255)  # Blue circle
        )
        gfxdraw.aacircle(
            self.screen,
            int(cart_x),
            int(self.screen_height / 2),
            10,
            (0, 0, 0)  # Black outline
        )
        
        # Draw velocity vector
        if abs(velocity) > 0.1:
            vel_line_length = velocity * 20
            pygame.draw.line(
                self.screen,
                (255, 165, 0),  # Orange line
                (cart_x, self.screen_height / 2),
                (cart_x + vel_line_length, self.screen_height / 2),
                2
            )
        
        # Add text for position and velocity
        font = pygame.font.Font(None, 36)
        position_text = font.render(f"Position: {position:.2f}", True, (0, 0, 0))
        velocity_text = font.render(f"Velocity: {velocity:.2f}", True, (0, 0, 0))
        self.screen.blit(position_text, (10, 10))
        self.screen.blit(velocity_text, (10, 50))
        
        # Draw action plot in the bottom part of the screen
        self._draw_action_plot()
        
        # Get the frame as an RGB array
        frame = np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), 
            axes=(1, 0, 2)
        )
        
        # Write frame to video if recording
        if self.record_video and self.video_writer is not None:
            # OpenCV uses BGR instead of RGB
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame_bgr)
        
        if mode == "human":
            self.clock.tick(self.fps)
            return None
        else:  # rgb_array
            return frame
    
    def _draw_action_plot(self):
        """Draw a plot of recent actions at the bottom of the screen"""
        if not self.action_history:
            return
            
        # Plot dimensions
        plot_height = 100
        plot_top = self.screen_height - plot_height - 10
        plot_bottom = self.screen_height - 10
        plot_left = 50
        plot_right = self.screen_width - 50
        plot_width = plot_right - plot_left
        
        # Draw plot background and borders
        pygame.draw.rect(
            self.screen, 
            (240, 240, 240),  # Light gray background
            (plot_left, plot_top, plot_width, plot_height)
        )
        pygame.draw.rect(
            self.screen, 
            (0, 0, 0),  # Black border
            (plot_left, plot_top, plot_width, plot_height),
            1  # Border width
        )
        
        # Draw y-axis labels and grid lines
        font = pygame.font.Font(None, 24)
        action_labels = {0: "Left", 1: "Stay", 2: "Right"}
        y_positions = {
            0: plot_bottom - 20,  # Left action (bottom)
            1: plot_top + plot_height // 2,  # Stay action (middle)
            2: plot_top + 20  # Right action (top)
        }
        
        # Draw horizontal grid lines and labels
        for action, y_pos in y_positions.items():
            # Grid line
            pygame.draw.line(
                self.screen,
                (200, 200, 200),  # Light gray
                (plot_left, y_pos),
                (plot_right, y_pos),
                1
            )
            
            # Label
            label = font.render(action_labels[action], True, (0, 0, 0))
            self.screen.blit(label, (plot_left - 45, y_pos - 10))
        
        # Draw title
        title = font.render("Action History", True, (0, 0, 0))
        self.screen.blit(title, (plot_left + plot_width // 2 - 50, plot_top - 25))
        
        # Draw the action history points and lines
        if len(self.action_history) > 1:
            x_step = plot_width / min(len(self.action_history), self.max_history_length)
            
            # Colors for each action
            action_colors = {
                0: (255, 0, 0),  # Red for left
                1: (0, 0, 0),    # Black for stay
                2: (0, 0, 255)   # Blue for right
            }
            
            # Draw the action points and lines
            for i in range(len(self.action_history) - 1):
                x1 = plot_left + i * x_step
                x2 = plot_left + (i + 1) * x_step
                y1 = y_positions[self.action_history[i]]
                y2 = y_positions[self.action_history[i + 1]]
                
                # Line connecting points
                pygame.draw.line(
                    self.screen,
                    (100, 100, 100),  # Gray line
                    (x1, y1),
                    (x2, y2),
                    2
                )
                
                # Point for current action
                pygame.draw.circle(
                    self.screen,
                    action_colors[self.action_history[i]],
                    (int(x1), int(y1)),
                    4
                )
            
            # Draw the last point
            pygame.draw.circle(
                self.screen,
                action_colors[self.action_history[-1]],
                (int(plot_left + (len(self.action_history) - 1) * x_step), 
                 int(y_positions[self.action_history[-1]])),
                4
            )
            
        # Draw the current action prominently
        if self.last_action is not None:
            current_action_text = f"Current Action: {action_labels[self.last_action]}"
            current_label = font.render(current_action_text, True, (0, 0, 0))
            self.screen.blit(current_label, (plot_right - 180, plot_top - 25))
    
    def _close_video(self):
        """Close the video writer if it exists"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            print(f"Video saved to {self.video_path}")
    
    def close(self):
        self._close_video()
        
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.isopen = False
