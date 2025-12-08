import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import random
from heapq import heappush, heappop
import pandas as pd

# ============================================================================
# Page Config and Initial Setup
# ============================================================================
st.set_page_config(
    page_title="RL Maze Solver",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🤖"
)

st.title("♾️Advanced RL Maze Solver")
st.markdown("""
This application demonstrates a Reinforcement Learning agent using **SARSA with Prioritized Sweeping** to solve guaranteed-solvable mazes.

1.  **Generate a Maze**: Use the sidebar controls to define the maze size and click the button.
2.  **Train the Agent**: Adjust the hyperparameters and click 'Train Agent' to start the learning process.
3.  **Test & Visualize**: Once trained, run a test episode to see the optimal path found by the agent.
""")

# ============================================================================
# Maze Generation Class (from imp.py)
# ============================================================================

class MazeGenerator:
    def __init__(self, width=21, height=21):
        self.width = width if width % 2 == 1 else width + 1
        self.height = height if height % 2 == 1 else height + 1

    def generate(self):
        maze = np.ones((self.height, self.width), dtype=int)
        start_node = (1, 1)
        maze[start_node] = 0
        stack = [start_node]
        visited = {start_node}
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]

        while stack:
            y, x = stack[-1]
            neighbors = []
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 1 <= ny < self.height - 1 and 1 <= nx < self.width - 1 and (ny, nx) not in visited:
                    neighbors.append((ny, nx, dy, dx))

            if neighbors:
                ny, nx, dy, dx = random.choice(neighbors)
                maze[y + dy // 2, x + dx // 2] = 0
                maze[ny, nx] = 0
                visited.add((ny, nx))
                stack.append((ny, nx))
            else:
                stack.pop()

        start_pos = (1, 1)
        end_pos = (self.height - 2, self.width - 2)
        return maze, start_pos, end_pos

# ============================================================================
# RL Agent Class (from imp.py, adapted for Streamlit)
# ============================================================================

class AdvancedMazeAgent:
    def __init__(self, maze, start, end, lr, gamma, epsilon_decay, epsilon_min):
        self.maze = maze
        self.start = start
        self.end = end
        self.h, self.w = maze.shape
        
        self.lr = lr
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.q_table = {}
        self.init_q_value = 10.0
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.model = {}
        self.priority_queue = []
        self.in_queue = set()
        
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_success = []

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), self.init_q_value)

    def choose_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.choice(range(len(self.actions)))
        
        q_values = [self.get_q_value(state, a) for a in range(len(self.actions))]
        max_q = max(q_values)
        best_actions = [i for i, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def get_next_state(self, state, action_idx):
        dy, dx = self.actions[action_idx]
        y, x = state
        ny, nx = y + dy, x + dx
        
        if 0 <= ny < self.h and 0 <= nx < self.w and self.maze[ny, nx] == 0:
            return (ny, nx)
        return state

    def get_reward(self, state, next_state):
        if next_state == self.end:
            return 1000
        elif state == next_state:
            return -10
        else:
            old_dist = abs(state[0] - self.end[0]) + abs(state[1] - self.end[1])
            new_dist = abs(next_state[0] - self.end[0]) + abs(next_state[1] - self.end[1])
            return 10 * (old_dist - new_dist) - 1

    def prioritized_update(self, state, action, priority, max_queue_size=1000):
        if (state, action) not in self.in_queue:
            if len(self.priority_queue) >= max_queue_size:
                # Prevent queue from growing indefinitely
                heappop(self.priority_queue)
            heappush(self.priority_queue, (-abs(priority), state, action))
            self.in_queue.add((state, action))

    def planning_step(self, n_steps=10):
        for _ in range(min(n_steps, len(self.priority_queue))):
            if not self.priority_queue:
                break
            
            _, state, action = heappop(self.priority_queue)
            self.in_queue.discard((state, action))
            
            if (state, action) in self.model:
                next_state, reward = self.model[(state, action)]
                current_q = self.get_q_value(state, action)
                max_next_q = max([self.get_q_value(next_state, a) for a in range(len(self.actions))])
                new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
                self.q_table[(state, action)] = new_q

    def train_episode(self, max_steps=500):
        state = self.start
        action = self.choose_action(state)
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            next_state = self.get_next_state(state, action)
            reward = self.get_reward(state, next_state)
            next_action = self.choose_action(next_state)
            
            self.model[(state, action)] = (next_state, reward)
            
            current_q = self.get_q_value(state, action)
            next_q = self.get_q_value(next_state, next_action)
            td_error = reward + self.gamma * next_q - current_q
            new_q = current_q + self.lr * td_error
            self.q_table[(state, action)] = new_q
            
            if abs(td_error) > 0.01:
                self.prioritized_update(state, action, abs(td_error))
            
            self.planning_step(n_steps=5)
            
            total_reward += reward
            steps += 1
            state = next_state
            action = next_action
            
            if state == self.end:
                break
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        success = state == self.end
        self.episode_rewards.append(total_reward)
        self.episode_steps.append(steps)
        self.episode_success.append(success)
        
        return total_reward, steps, success

    def test_episode(self, max_steps=500):
        state = self.start
        path = [state]
        steps = 0
        visited = {state}
        
        for step in range(max_steps):
            action = self.choose_action(state, training=False)
            next_state = self.get_next_state(state, action)
            
            if next_state in visited and next_state != self.end:
                q_values = [self.get_q_value(state, a) for a in range(len(self.actions))]
                q_values[action] = float('-inf')
                if max(q_values) > float('-inf'):
                    action = q_values.index(max(q_values))
                    next_state = self.get_next_state(state, action)
            
            path.append(next_state)
            visited.add(next_state)
            steps += 1
            state = next_state
            
            if state == self.end:
                break
        
        return state == self.end, path

# ============================================================================
# Streamlit UI and Logic
# ============================================================================

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Universe Controls")

with st.sidebar.expander("1. Maze Generation", expanded=True):
    maze_width = st.slider("Maze Width", 11, 101, 21, 2, help="Must be an odd number.")
    maze_height = st.slider("Maze Height", 11, 101, 21, 2, help="Must be an odd number.")
    if st.button("Generate New Maze", use_container_width=True):
        gen = MazeGenerator(width=maze_width, height=maze_height)
        maze, start, end = gen.generate()
        st.session_state.maze_data = {
            "maze": maze,
            "start": start,
            "end": end
        }
        st.session_state.agent = None
        st.session_state.training_history = None
        st.toast("New maze generated!", icon="🗺️")
        st.rerun()

with st.sidebar.expander("2. Agent Hyperparameters", expanded=True):
    lr = st.slider("Learning Rate (α)", 0.1, 1.0, 0.3, 0.05)
    gamma = st.slider("Discount Factor (γ)", 0.9, 0.999, 0.99, 0.001)
    epsilon_decay = st.slider("Epsilon Decay", 0.99, 0.9999, 0.9995, 0.0001, format="%.4f")
    epsilon_min = st.slider("Min Epsilon (ε)", 0.01, 0.2, 0.05, 0.01)

with st.sidebar.expander("3. Training Configuration", expanded=True):
    episodes = st.number_input("Training Episodes", 100, 10000, 1000, 100)
    max_steps = st.number_input("Max Steps per Episode", 100, 2000, 500, 50)
    early_stop_count = st.number_input("Early Stop (consecutive successes)", 5, 50, 10, 1)

train_button = st.sidebar.button("Train Agent", use_container_width=True, type="primary")

st.sidebar.divider()

if st.sidebar.button("Clear Memory & Reset", use_container_width=True):
    keys_to_clear = ['maze_data', 'agent', 'training_history']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.cache_data.clear()
    st.toast("Memory and cache cleared!", icon="🧼")
    st.rerun()

# --- Main Area ---

if 'maze_data' not in st.session_state:
    st.info("Welcome! Please generate a maze using the controls in the sidebar to begin.")
else:
    maze_data = st.session_state.maze_data
    maze = maze_data['maze']
    start = maze_data['start']
    end = maze_data['end']

    # Initialize or retrieve agent from session state
    if 'agent' not in st.session_state or st.session_state.agent is None:
        st.session_state.agent = AdvancedMazeAgent(maze, start, end, lr, gamma, epsilon_decay, epsilon_min)

    agent = st.session_state.agent

    # Display Maze
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Maze Environment")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(maze, cmap='binary')
        ax.plot(start[1], start[0], 'go', markersize=15, label='Start')
        ax.plot(end[1], end[0], 'ro', markersize=15, label='End')
        ax.legend()
        ax.set_title(f"Maze: {maze.shape}")
        ax.axis('off')
        st.pyplot(fig)

    with col2:
        if train_button:
            st.subheader("Training in Progress...")
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            consecutive_successes = 0
            for episode in range(1, episodes + 1):
                reward, steps, success = agent.train_episode(max_steps)
                
                if success:
                    consecutive_successes += 1
                else:
                    consecutive_successes = 0

                status_text.text(f"Episode {episode}/{episodes} | Success: {success} | Steps: {steps} | Epsilon: {agent.epsilon:.3f}")
                progress_bar.progress(episode / episodes)

                if consecutive_successes >= early_stop_count:
                    st.success(f"Early stopping triggered after {episode} episodes!")
                    break
            
            st.session_state.training_history = {
                'rewards': agent.episode_rewards,
                'steps': agent.episode_steps,
                'success': agent.episode_success
            }
            st.rerun()

        if 'training_history' in st.session_state and st.session_state.training_history:
            st.subheader("Training Performance")
            history = st.session_state.training_history
            df = pd.DataFrame(history)
            df['success_rate'] = df['success'].rolling(window=50, min_periods=1).mean()

            fig, ax1 = plt.subplots(figsize=(12, 5))
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Smoothed Reward', color='tab:blue')
            ax1.plot(df['rewards'].rolling(window=50).mean(), color='tab:blue', label='Reward')
            ax1.tick_params(axis='y', labelcolor='tab:blue')

            ax2 = ax1.twinx()
            ax2.set_ylabel('Success Rate', color='tab:green')
            ax2.plot(df['success_rate'], color='tab:green', label='Success Rate')
            ax2.tick_params(axis='y', labelcolor='tab:green')
            ax2.set_ylim(0, 1.1)

            fig.tight_layout()
            st.pyplot(fig)

        st.subheader("Agent Testing")
        if st.button("Run Test & Visualize Path", use_container_width=True):
            with st.spinner("Agent is solving the maze..."):
                success, path = agent.test_episode(max_steps=maze.size)
            
            if success:
                st.success(f"Agent found the solution in {len(path)-1} steps!")
            else:
                st.error(f"Agent failed to find the solution after {len(path)-1} steps.")

            # Visualization
            fig, ax = plt.subplots(figsize=(10, 10))
            vis_maze = maze.copy().astype(float)
            for i, (y, x) in enumerate(path):
                vis_maze[y, x] = 0.3 + 0.4 * (i / len(path))
            
            ax.imshow(vis_maze, cmap='RdYlGn_r', vmin=0, vmax=1)
            path_y = [p[0] for p in path]
            path_x = [p[1] for p in path]
            ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.8)
            ax.plot(start[1], start[0], 'go', markersize=15, label='Start')
            ax.plot(end[1], end[0], 'ro', markersize=15, label='End')
            ax.set_title(f"Agent's Final Path ({'Success' if success else 'Failure'})")
            ax.axis('off')
            st.pyplot(fig)
