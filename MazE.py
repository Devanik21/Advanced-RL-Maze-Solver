import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import random
from heapq import heappush, heappop
import pandas as pd

import json
import zipfile
import io
import ast  # To safely convert string keys back to tuples
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
        
        # --- MAGIC UPDATE: Pre-compute the true distance to end for every cell ---
        self.distance_map = self._compute_distance_map()
        # -----------------------------------------------------------------------

        self.lr = lr
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.q_table = {}
        self.init_q_value = 0.0 # Changed to 0 for better stability with new rewards
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.model = {}
        self.priority_queue = []
        self.in_queue = set()
        
        self.experience_buffer = deque(maxlen=20000)
        self.heuristic_states = []

        self.episode_rewards = []
        self.episode_steps = []
        self.episode_success = []

    def _compute_distance_map(self):
        """
        Uses BFS to calculate the exact number of steps from every cell to the goal.
        This creates a 'gradient' for the agent to follow.
        """
        d_map = np.full((self.h, self.w), fill_value=np.inf)
        d_map[self.end] = 0
        queue = deque([self.end])
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            cy, cx = queue.popleft()
            current_dist = d_map[cy, cx]
            
            for dy, dx in directions:
                ny, nx = cy + dy, cx + dx
                
                # Check bounds and walls (0 is path, 1 is wall)
                if 0 <= ny < self.h and 0 <= nx < self.w:
                    if self.maze[ny, nx] == 0:
                        if d_map[ny, nx] == np.inf:
                            d_map[ny, nx] = current_dist + 1
                            queue.append((ny, nx))
        return d_map

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
            return 100.0  # Big reward for solving
        
        if state == next_state:
            return -5.0  # Punishment for hitting a wall
            
        # Get true distances from our magic map
        current_dist = self.distance_map[state]
        next_dist = self.distance_map[next_state]
        
        # MAGIC: The reward is simply the improvement in true distance
        # If we moved closer to goal (diff is positive), we get + reward
        # If we moved away (diff is negative), we get - reward
        diff = current_dist - next_dist
        
        # We multiply by a factor (e.g., 2.0) to make the signal strong
        # We subtract a small 'step cost' (0.1) to encourage speed
        return (2.0 * diff) - 0.1

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

    def experience_replay(self, batch_size=32):
        if len(self.experience_buffer) < batch_size:
            return
        
        batch = random.sample(self.experience_buffer, batch_size)
        for state, action, reward, next_state, done in batch:
            current_q = self.get_q_value(state, action)
            if done:
                target = reward
            else:
                max_next_q = max([self.get_q_value(next_state, a) for a in range(len(self.actions))])
                target = reward + self.gamma * max_next_q
            
            new_q = current_q + self.lr * (target - current_q)
            self.q_table[(state, action)] = new_q

    def train_episode(self, max_steps=500):
        state = self.start_new_episode()
        action = self.choose_action(state)
        total_reward, steps = 0, 0
        
        for step in range(max_steps):
            next_state = self.get_next_state(state, action)
            reward = self.get_reward(state, next_state)
            next_action = self.choose_action(next_state)
            
            self.model[(state, action)] = (next_state, reward)
            done = next_state == self.end
            self.experience_buffer.append((state, action, reward, next_state, done))
            
            current_q = self.get_q_value(state, action)
            next_q = self.get_q_value(next_state, next_action)
            td_error = reward + self.gamma * next_q - current_q
            new_q = current_q + self.lr * td_error
            self.q_table[(state, action)] = new_q

            if abs(td_error) > 0.01:
                self.prioritized_update(state, action, abs(td_error))
            
            self.planning_step(n_steps=5)
            self.experience_replay(batch_size=16) # Replay from past experiences
            
            total_reward += reward
            steps += 1
            state = next_state
            action = next_action
            
            if state == self.end:
                self.heuristic_states.append(self.start) # Add successful start
                break
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        success = state == self.end
        if success and len(self.heuristic_states) < 200: # Limit heuristic states
            # Add states from the successful path to the heuristic list
            # path is not tracked here, but we can add the state before the end
            pass # This logic is implicitly handled by starting near goal

        self.episode_rewards.append(total_reward)
        self.episode_steps.append(steps)
        self.episode_success.append(success)
        
        return total_reward, steps, success

    def start_new_episode(self):
        if self.heuristic_states and random.random() < 0.3: # 30% chance to start near goal
            return random.choice(self.heuristic_states)
        return self.start

    def test_episode(self, max_steps=500):
        state = self.start
        path = [(state, 'start')] # Store (state, move_type)
        steps = 0
        visited = {state}
        
        for step in range(max_steps):
            action = self.choose_action(state, training=False)
            next_state = self.get_next_state(state, action)
            move_type = 'good' # Assume a good move
            
            if next_state in visited and next_state != self.end:
                move_type = 'suboptimal' # It's a mistake to revisit a cell
                q_values = [self.get_q_value(state, a) for a in range(len(self.actions))]
                q_values[action] = float('-inf')
                if max(q_values) > float('-inf'):
                    action = q_values.index(max(q_values))
                    corrected_next_state = self.get_next_state(state, action)
                    if corrected_next_state == next_state: # Failed to correct, stuck in a loop
                        move_type = 'bad'
                    next_state = corrected_next_state
                else: # No other option, truly stuck
                    move_type = 'bad'
            
            path.append((next_state, move_type))
            visited.add(next_state)
            steps += 1
            state = next_state
            
            if state == self.end:
                break
        return state == self.end, [p[0] for p in path], [p[1] for p in path]



# ============================================================================
# Save/Load Utility Functions
# ============================================================================

def serialize_q_table(q_table):
    """
    JSON only supports string keys. We must convert our tuple keys 
    ((y, x), action_idx) into strings.
    """
    serialized = {}
    for key, value in q_table.items():
        # Convert tuple key to string representation
        serialized[str(key)] = value 
    return serialized

def deserialize_q_table(serialized_q):
    """
    Convert string keys back to actual tuples for the Python code.
    """
    q_table = {}
    for key_str, value in serialized_q.items():
        # ast.literal_eval safely evaluates a string containing a Python literal
        key_tuple = ast.literal_eval(key_str)
        q_table[key_tuple] = value
    return q_table

def create_brain_zip(agent, maze_data):
    # 1. Prepare Agent Data
    agent_state = {
        "q_table": serialize_q_table(agent.q_table),
        "epsilon": agent.epsilon,
        "lr": agent.lr,
        "gamma": agent.gamma,
        "model": {str(k): v for k, v in agent.model.items()}, # Model also uses tuple keys
        "episode_rewards": agent.episode_rewards,
        "episode_steps": agent.episode_steps,
        "episode_success": agent.episode_success
    }
    
    # 2. Prepare Maze Data (Convert numpy to list for JSON)
    env_state = {
        "maze": maze_data['maze'].tolist(),
        "start": maze_data['start'],
        "end": maze_data['end'],
        "width": maze_data['maze'].shape[1],
        "height": maze_data['maze'].shape[0]
    }

    # 3. Write to Zip in Memory
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("agent_brain.json", json.dumps(agent_state))
        zf.writestr("world_map.json", json.dumps(env_state))
    
    buffer.seek(0)
    return buffer

def load_brain_from_zip(uploaded_file):
    try:
        with zipfile.ZipFile(uploaded_file, "r") as zf:
            # Load JSONs
            agent_json = zf.read("agent_brain.json")
            env_json = zf.read("world_map.json")
            
            agent_state = json.loads(agent_json)
            env_state = json.loads(env_json)
            
            # Reconstruct Maze
            maze = np.array(env_state["maze"])
            start = tuple(env_state["start"])
            end = tuple(env_state["end"])
            
            maze_data = {"maze": maze, "start": start, "end": end}
            
            # Reconstruct Agent
            # We initialize a new agent, then overwrite its brain
            new_agent = AdvancedMazeAgent(
                maze, start, end, 
                agent_state['lr'], agent_state['gamma'], 
                0.9995, 0.01 # These defaults will be overwritten or ignored
            )
            
            # Restore the brain
            new_agent.q_table = deserialize_q_table(agent_state['q_table'])
            new_agent.model = deserialize_q_table(agent_state['model'])
            new_agent.epsilon = agent_state['epsilon']
            new_agent.episode_rewards = agent_state['episode_rewards']
            new_agent.episode_steps = agent_state['episode_steps']
            new_agent.episode_success = agent_state['episode_success']
            
            return maze_data, new_agent, agent_state
    except Exception as e:
        st.error(f"Corrupted Brain File: {e}")
        return None, None, None



# ============================================================================
# Streamlit UI and Logic
# ============================================================================

# --- Sidebar Controls ---
# --- Sidebar Controls ---
st.sidebar.header("⚙️ Universe Controls")

with st.sidebar.expander("1. Maze Generation", expanded=True):
    # Changed from slider to number_input to allow infinite size
    # We set min_value=2. The class automatically converts evens to odds, so no warning is needed.
    maze_width = st.number_input("Maze Width", min_value=2, value=21, step=1)
    maze_height = st.number_input("Maze Height", min_value=2, value=21, step=1)
    
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
    max_steps = st.number_input("Max Steps per Episode", 100, 5000, 1000, 50)
    early_stop_count = st.number_input("Early Stop (consecutive successes)", 5, 50, 10, 1)

# --- NEW SECTION STARTS HERE ---
with st.sidebar.expander("4. Brain Storage (Save/Load)", expanded=False):
    st.markdown("Download your agent's brain to disk and reload it later.")
    
    # DOWNLOAD LOGIC
    if 'agent' in st.session_state and st.session_state.agent is not None:
        zip_buffer = create_brain_zip(st.session_state.agent, st.session_state.maze_data)
        st.download_button(
            label="💾 Download Agent Brain (.zip)",
            data=zip_buffer,
            file_name="my_ai_brain.zip",
            mime="application/zip",
            use_container_width=True
        )
    else:
        st.warning("Train or Create an agent first to download.")

    # UPLOAD LOGIC
    uploaded_file = st.file_uploader("Upload Brain (.zip)", type="zip")
    if uploaded_file is not None:
        if st.button("Load Brain & Map", use_container_width=True):
            loaded_maze_data, loaded_agent, raw_state = load_brain_from_zip(uploaded_file)
            if loaded_maze_data:
                st.session_state.maze_data = loaded_maze_data
                st.session_state.agent = loaded_agent
                
                # Restore history for the graph
                st.session_state.training_history = {
                    'rewards': raw_state['episode_rewards'],
                    'steps': raw_state['episode_steps'],
                    'success': raw_state['episode_success']
                }
                st.toast("Brain and Map Restored Successfully!", icon="🧠")
                st.rerun()
# --- NEW SECTION ENDS HERE ---

train_button = st.sidebar.button("Train Agent", use_container_width=True, type="primary")

st.sidebar.divider()
# ... (Keep your Clear Memory button logic here)

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
            status_container = st.empty()
            vis_container = st.empty()
            progress_bar = st.progress(0)
            
            consecutive_successes = 0
            total_successes = 0
            recent_successes = deque(maxlen=100) # For success rate over last 100

            for episode in range(1, episodes + 1):
                reward, steps, success = agent.train_episode(max_steps)
                
                if success:
                    consecutive_successes += 1
                    total_successes += 1
                else:
                    consecutive_successes = 0

                recent_successes.append(success)
                success_rate = sum(recent_successes) / len(recent_successes) if recent_successes else 0.0

                # Update the status dashboard
                status_markdown = f"""
                | Parameter | Value |
                |---|---|
                | **Episode** | `{episode}` / `{episodes}` |
                | **Epsilon (ε)** | `{agent.epsilon:.4f}` |
                | **Q-Table Size** | `{len(agent.q_table)}` |
                | **Last Episode Success** | `{'✅ True' if success else '❌ False'}` |
                | **Last Episode Steps** | `{steps}` |
                | **Success Rate (last 100)** | `{success_rate:.2%}` |
                | **Consecutive Successes** | `{consecutive_successes}` / `{early_stop_count}` |
                """
                status_container.markdown(status_markdown)
                progress_bar.progress(episode / episodes)

                # Visualize path every 100 episodes
                if episode % 100 == 0 or episode == 1:
                    with vis_container.container():
                        st.write(f"Visualizing path at Episode {episode}...")
                        # Run a test episode to get the current path
                        test_success, test_path = agent.test_episode(max_steps=maze.size)

                        # Visualization
                        fig_vis, ax_vis = plt.subplots(figsize=(6, 6))
                        ax_vis.imshow(maze, cmap='binary') # Show maze structure

                        path_y = [p[0] for p in test_path]
                        path_x = [p[1] for p in test_path]
                        ax_vis.plot(path_x, path_y, '-', color='#FF5733', linewidth=1.5, alpha=0.9)
                        # Add start and end markers on top of the black background
                        ax_vis.plot(start[1], start[0], 'go', markersize=8, label='Start') # Green start
                        ax_vis.plot(end[1], end[0], 'ro', markersize=8, label='End') # Red end

                        ax_vis.set_title(f"Path at Episode {episode} ({'Success' if test_success else 'In Progress'})", color='white')
                        ax_vis.axis('off')
                        st.pyplot(fig_vis)

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
                success, path, move_types = agent.test_episode(max_steps=maze.size)
            
            if success:
                st.success(f"Agent found the solution in {len(path)-1} steps!")
            else:
                st.error(f"Agent failed to find the solution after {len(path)-1} steps.")

            # Visualization with colored path segments
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(maze, cmap='binary') # Show maze structure

            # Define colors for move types
            color_map = {'good': 'darkgreen', 'suboptimal': 'darkblue', 'bad': 'darkred'}

            # Plot each path segment with its corresponding color
            for i in range(len(path) - 1):
                p1 = path[i]
                p2 = path[i+1]
                move_type = move_types[i+1] # move_types corresponds to the state it leads to
                color = color_map.get(move_type, 'darkblue')
                ax.plot([p1[1], p2[1]], [p1[0], p2[0]], '-', color=color, linewidth=2, alpha=0.9)

            # Plot Start and End markers
            ax.plot(start[1], start[0], 'go', markersize=15, label='Start') # Green start
            ax.plot(end[1], end[0], 'ro', markersize=15, label='End') # Red end
            ax.legend()

            ax.set_title(f"Agent's Final Path ({'Success' if success else 'Failure'})", color='white')
            ax.axis('off')
            st.pyplot(fig)
