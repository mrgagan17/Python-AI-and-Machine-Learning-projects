import streamlit as st
import numpy as np
import time
import math
import pygame
import torch
from PIL import Image
from dqn_architecture import DQNAgent

# 1. Dashboard Layout & Styling
st.set_page_config(page_title="Self-Driving RL Agent", page_icon="🚗", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #0E0E10; color: #E2E2E2; }
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #FF9900, #FF5E62, #9B72CB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
    }
    .metric-box { background-color: #1A1A1E; padding: 20px; border-radius: 10px; border: 1px solid #3C4043; text-align: center; margin-top: 10px;}
</style>
""", unsafe_allow_html=True)

st.title("🚗 Autonomous Driving RL Simulation")
st.markdown("Watch a Deep Q-Network (DQN) reinforcement learning agent interact with its environment, gaining a reward for driving forward and penalties for colliding with borders.")
st.markdown("---")

# Parameters
TRACK_WIDTH, TRACK_HEIGHT = 500, 400
CAR_WIDTH, CAR_HEIGHT = 16, 28
RADAR_ANGLE_STEP = 30 # degrees

# 2. Pygame Environment Logic
class SimulationEnv:
    def __init__(self):
        pygame.init()
        # Non-visible Pygame surface
        self.surface = pygame.Surface((TRACK_WIDTH, TRACK_HEIGHT))
        self.reset()

    def reset(self):
        self.car_x = TRACK_WIDTH // 2
        self.car_y = 310
        self.car_angle = 180  # Degrees
        self.speed = 3
        self.distance_traveled = 0
        self.steps = 0
        self.done = False
        return self._get_observation()

    def _get_border_points(self):
        # Coordinates defining track boundaries
        outer = pygame.Rect(30, 30, TRACK_WIDTH - 60, TRACK_HEIGHT - 60)
        inner = pygame.Rect(120, 110, TRACK_WIDTH - 240, TRACK_HEIGHT - 220)
        return outer, inner

    def _get_observation(self):
        # Custom Lidar radars (5 distance measurements)
        radars = []
        outer, inner = self._get_border_points()
        angles = [-60, -30, 0, 30, 60]
        
        for a in angles:
            rad_angle = math.radians(self.car_angle + a)
            dist = 0
            while dist < 120:  # Max laser sweep range
                test_x = int(self.car_x + math.cos(rad_angle) * dist)
                test_y = int(self.car_y + math.sin(rad_angle) * dist)
                
                # Boundary conditions
                inside_outer = outer.collidepoint(test_x, test_y)
                inside_inner = inner.collidepoint(test_x, test_y)
                if not inside_outer or inside_inner:
                    break
                dist += 4
            radars.append(dist / 120.0) # Normalize state space between [0, 1]
        
        return np.array(radars, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        
        # Action mappings (0: Left, 1: Right, 2: Straight)
        if action == 0:
            self.car_angle -= 4
        elif action == 1:
            self.car_angle += 4
            
        rad_angle = math.radians(self.car_angle)
        self.car_x += math.cos(rad_angle) * self.speed
        self.car_y += math.sin(rad_angle) * self.speed
        self.distance_traveled += self.speed
        
        # Collision validation
        outer, inner = self._get_border_points()
        inside_outer = outer.collidepoint(self.car_x, self.car_y)
        inside_inner = inner.collidepoint(self.car_x, self.car_y)
        
        reward = 1.0  # Reward for staying alive
        
        # Impact penalty & terminal condition
        if not inside_outer or inside_inner or self.steps > 1500:
            self.done = True
            reward = -100.0 if self.steps <= 1500 else 0
            
        return self._get_observation(), reward, self.done

    def render(self):
        # Back buffer draw loop
        self.surface.fill((20, 20, 25))
        
        outer, inner = self._get_border_points()
        pygame.draw.rect(self.surface, (50, 50, 60), outer, 6, border_radius=15)
        pygame.draw.rect(self.surface, (50, 50, 60), inner, 6, border_radius=15)
        
        # Draw Start Line
        pygame.draw.line(self.surface, (255, 150, 0), (TRACK_WIDTH//2, 310-25), (TRACK_WIDTH//2, 310+25), 4)
        
        # Draw Car
        car_surf = pygame.Surface((CAR_HEIGHT, CAR_WIDTH), pygame.SRCALPHA)
        pygame.draw.polygon(car_surf, (255, 94, 98), [(0, 0), (CAR_HEIGHT, CAR_WIDTH//2), (0, CAR_WIDTH)])
        rotated_car = pygame.transform.rotate(car_surf, -self.car_angle)
        rect = rotated_car.get_rect(center=(int(self.car_x), int(self.car_y)))
        self.surface.blit(rotated_car, rect.topleft)
        
        # Convert to raw PIL Image for streamlit insertion
        string_image = pygame.image.tostring(self.surface, 'RGB')
        return Image.frombytes('RGB', (TRACK_WIDTH, TRACK_HEIGHT), string_image)

# 3. Streamlit dashboard control state initialization
if 'env' not in st.session_state:
    st.session_state.env = SimulationEnv()
    st.session_state.agent = DQNAgent(state_dim=5, action_dim=3)
    st.session_state.history = []

col1, col2 = st.columns([2, 1])

with col1:
    sim_display = st.empty()

with col2:
    st.subheader("Training Metrics")
    status_text = st.markdown("**Initial State: Ready**")
    total_episodes_st = st.metric("Completed Episodes", "0")
    total_steps_st = st.metric("Current Episode Steps", "0")
    avg_r_st = st.metric("Avg Episode Reward", "0.0")

if st.button("Train Autonomous RL Agent ✦", use_container_width=True):
    env = st.session_state.env
    agent = st.session_state.agent
    episodes_target = 30
    batch_size = 32
    
    for ep in range(1, episodes_target + 1):
        status_text.markdown(f"🔴 **Episode {ep}/{episodes_target} Running...**")
        state = env.reset()
        episode_reward = 0
        
        while not env.done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            # Display current frame updates at 30 FPS
            if env.steps % 4 == 0:
                sim_display.image(env.render(), caption="Dynamic Top-Down Driving track", width="stretch")
                time.sleep(0.01)
                
            total_steps_st.metric("Current Episode Steps", f"{env.steps}")
            
            if done:
                break
                
        agent.replay(batch_size)
        agent.update_target()
        
        st.session_state.history.append(episode_reward)
        avg_reward = sum(st.session_state.history) / len(st.session_state.history)
        
        total_episodes_st.metric("Completed Episodes", f"{ep}")
        avg_r_st.metric("Avg Episode Reward", f"{avg_reward:.1f}")
        
    status_text.markdown("✅ **Simulation Complete**")