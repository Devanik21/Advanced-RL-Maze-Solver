import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import random
from heapq import heappush, heappop

# ============================================================================
# 1. PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="AI Maze Master - RL Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Genevo: RL Agent Maze Solver")
st.markdown("### Interactive Reinforcement Learning Dashboard")

# ============================================================================
# 2. SIDEBAR: THE 20+ CONTROL PARAMETERS
# ============================================================================
st.sidebar.header("⚙️ Control Panel")

# --- SECTION A: MAZE CONFIGURATION (COMPLEXITY) ---
with st.sidebar.expander("1. Maze Complexity & Geometry", expanded=True):
    # Param 1 & 2: Dimensions directly affect complexity
    MAZE_W = st.slider("Maze Width (Odd #)", 5, 51, 21, step=2, help="Larger width = Exponentially higher complexity")
    MAZE_H = st.slider("Maze Height (Odd #)", 5, 51, 21, step=2, help="Larger height = Exponentially higher complexity")
    # Param 3: Determinism
    SEED = st.number_input("Random Seed", value=42, help="Keep this same to reproduce the exact same maze")
    
# --- SECTION B: AGENT HYPERPARAMETERS ---
with st.sidebar.expander("2. Agent Brain (Hyperparameters)", expanded=False):
    # Param 4: Learning Rate
    LR = st.slider("Learning Rate (Alpha)", 0.01, 1.0, 0.3, 0.01, help="How fast the agent accepts new information")
    # Param 5: Discount Factor
    GAMMA = st.slider("Discount Factor (Gamma)", 0.5, 0.999, 0.99, 0.001, help="Importance of future rewards vs immediate ones")
    # Param 6, 7, 8: Exploration Strategy
    EPS_START = st.slider("Initial Exploration (Epsilon)", 0.1, 1.0, 1.0, 0.1)
    EPS_MIN = st.slider("Min Exploration", 0.01, 0.2, 0.05, 0.01)
    EPS_DECAY = st.slider("Epsilon Decay", 0.900, 0.9999, 0.9995, 0.0001, format="%.4f")
    # Param 9: Initialization
    INIT_Q = st.number_input("Optimistic Q-Init", value=10.0, help="Encourages exploration of unknown states")

# --- SECTION C: PLANNING (PRIORITIZED SWEEPING) ---
with st.sidebar.expander("3. Advanced Planning Logic", expanded=False):
    # Param 10: Model steps
    PLAN_STEPS = st.slider("Planning Steps", 0, 50, 10, help="How much the agent 'thinks' using its internal model per step")
    # Param 11: Priority
    PRIO_THRESH = st.number_input("Priority Threshold", value=0.01, format="%.3f")

# --- SECTION D: REWARD ENGINEERING ---
with st.sidebar.expander("4. Reward Structure", expanded=False):
    # Param 12, 13, 14: Shaping behavior
    GOAL_REWARD = st.number_input("Goal Reward", value=1000)
    WALL_PENALTY = st.number_input("Wall Hit Penalty", value=-10)
    STEP_PENALTY = st.number_input("Step/Distance Cost", value=-1)

# --- SECTION E: TRAINING CONTROLS ---
with st.sidebar.expander("5. Training Execution", expanded=False):
    # Param 15, 16, 17
    EPISODES = st.slider("Total Episodes", 100, 2000, 500, 100)
    MAX_STEPS = st.slider("Max Steps/Episode", 100, 1000, 500, 50)
    EARLY_STOP = st.slider("Stop after N Successes", 5, 50, 10, help="Stop training if agent wins this many times in a row")

# --- SECTION F: VISUALIZATION AESTHETICS ---
with st.sidebar.expander("6. Visuals", expanded=False):
    # Param 18, 19, 20
    PATH_COLOR = st.color_picker("Path Color", "#0000FF")
    START_COLOR = st.color_picker("Start Color", "#00FF00")
    END_COLOR = st.color_picker("End Color", "#FF0000")
    SHOW_TRAINING = st.checkbox("Show Live Metrics?", value=True, help="Uncheck for faster training speed")

# ============================================================================
# 3. LOGIC CLASSES (Refactored for Streamlit)
# ============================================================================

class MazeGenerator:
    def __init__(self, width=21, height=21, seed=42):
        self.width = width if width % 2 == 1 else width + 1
        self.height = height if height % 2 == 1 else height + 1
        random.seed(seed)
        np.random.seed(seed)
        
    def generate(self):
        maze = np.ones((self.height, self.width), dtype=int)
        start = (1, 1)
        maze[start] = 0
        stack = [start]
        visited = {start}
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        
        while stack:
            y, x = stack[-1]
            neighbors = []
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if (1 <= ny < self.height - 1 and 
                    1 <= nx < self.width - 1 and 
                    (ny, nx) not in visited):
                    neighbors.append((ny, nx, dy, dx))
            
            if neighbors:
                ny, nx, dy, dx = random.choice(neighbors)
                maze[y + dy//2, x + dx//2] = 0
                maze[ny, nx] = 0
                visited.add((ny, nx))
                stack.append((ny, nx))
            else:
                stack.pop()
        
        return maze, start, (self.height - 2, self.width - 2)

class AdvancedMazeAgent:
    def __init__(self, maze, start, end):
        self.maze = maze
        self.start = start
        self.end = end
        self.h, self.w = maze.shape
        
        # Hyperparams from Sidebar
        self.lr = LR
        self.gamma = GAMMA
        self.epsilon = EPS_START
        self.epsilon_min = EPS_MIN
        self.epsilon_decay = EPS_DECAY
        
        self.q_table = {}
        self.init_q_value = INIT_Q
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.model = {}
        self.priority_queue = []
        self.in_queue = set()
        
        # History for plotting
        self.history_rewards = []
        self.history_success = []

    def get_q_value(self, state, action):
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = self.init_q_value
        return self.q_table[(state, action)]
    
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
            return GOAL_REWARD
        elif state == next_state:
            return WALL_PENALTY
        else:
            # Manhattan distance shaping
            old_dist = abs(state[0] - self.end[0]) + abs(state[1] - self.end[1])
            new_dist = abs(next_state[0] - self.end[0]) + abs(next_state[1] - self.end[1])
            return 10 * (old_dist - new_dist) + STEP_PENALTY

    def prioritized_update(self, state, action, priority_threshold=PRIO_THRESH):
        if (state, action) not in self.in_queue:
            heappush(self.priority_queue, (-abs(priority_threshold), state, action))
            self.in_queue.add((state, action))
    
    def planning_step(self):
        for _ in range(min(PLAN_STEPS, len(self.priority_queue))):
            if not self.priority_queue: break
            _, state, action = heappop(self.priority_queue)
            self.in_queue.discard((state, action))
            if (state, action) in self.model:
                next_state, reward = self.model[(state, action)]
                current_q = self.get_q_value(state, action)
                max_next_q = max([self.get_q_value(next_state, a) for a in range(len(self.actions))])
                new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
                self.q_table[(state, action)] = new_q

    def train_episode(self, max_steps=MAX_STEPS):
        state = self.start
        action = self.choose_action(state)
        total_reward = 0
        
        for _ in range(max_steps):
            next_state = self.get_next_state(state, action)
            reward = self.get_reward(state, next_state)
            next_action = self.choose_action(next_state)
            
            self.model[(state, action)] = (next_state, reward)
            
            current_q = self.get_q_value(state, action)
            next_q = self.get_q_value(next_state, next_action)
            td_error = reward + self.gamma * next_q - current_q
            self.q_table[(state, action)] = current_q + self.lr * td_error
            
            if abs(td_error) > PRIO_THRESH:
                self.prioritized_update(state, action, abs(td_error))
            
            self.planning_step()
            
            total_reward += reward
            state = next_state
            action = next_action
            
            if state == self.end:
                break
                
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        success = (state == self.end)
        self.history_rewards.append(total_reward)
        self.history_success.append(1 if success else 0)
        return success

    def solve(self):
        # Deterministic solve for visualization
        path = [self.start]
        state = self.start
        visited = {state}
        
        for _ in range(MAX_STEPS * 2):
            # Greedy action only
            q_values = [self.get_q_value(state, a) for a in range(len(self.actions))]
            action = np.argmax(q_values)
            next_state = self.get_next_state(state, action)
            
            # Simple loop avoidance for visualization
            if next_state in visited and next_state != self.end:
                # Try second best
                q_values[action] = -99999
                action = np.argmax(q_values)
                next_state = self.get_next_state(state, action)
            
            path.append(next_state)
            visited.add(next_state)
            state = next_state
            if state == self.end:
                break
        return path, (state == self.end)

# ============================================================================
# 4. MAIN APP LOGIC
# ============================================================================

# Initialize Session State
if 'maze' not in st.session_state:
    st.session_state.maze_obj = None
    st.session_state.agent = None

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"🗺️ Current Maze ({MAZE_W}x{MAZE_H})")
    
    # GENERATE BUTTON
    if st.button("🔄 Generate New Maze", type="secondary"):
        gen = MazeGenerator(width=MAZE_W, height=MAZE_H, seed=SEED)
        maze, start, end = gen.generate()
        st.session_state.maze_obj = {'grid': maze, 'start': start, 'end': end}
        # Reset agent when maze changes
        st.session_state.agent = AdvancedMazeAgent(maze, start, end)
        st.success("New Maze Generated!")

    # Display Maze Area
    place_holder_maze = st.empty()

    if st.session_state.maze_obj is not None:
        # Plot helper
        def plot_maze(path=None, title="Maze"):
            grid = st.session_state.maze_obj['grid']
            start = st.session_state.maze_obj['start']
            end = st.session_state.maze_obj['end']
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(grid, cmap='binary')
            
            # Plot Path
            if path:
                py = [p[0] for p in path]
                px = [p[1] for p in path]
                ax.plot(px, py, color=PATH_COLOR, linewidth=3, alpha=0.7, label='Agent Path')
            
            # Start/End
            ax.plot(start[1], start[0], 'o', color=START_COLOR, markersize=15, label='Start')
            ax.plot(end[1], end[0], 'o', color=END_COLOR, markersize=15, label='End')
            
            ax.axis('off')
            ax.set_title(title)
            return fig

        # Show initial empty maze
        place_holder_maze.pyplot(plot_maze(title="Current Environment"))

# --- TRAINING SECTION ---
with col2:
    st.subheader("🎓 Training")
    
    if st.session_state.agent is not None:
        if st.button("🚀 Start Training", type="primary"):
            agent = st.session_state.agent
            
            # Training Loop
            progress_bar = st.progress(0)
            status_text = st.empty()
            metric_col1, metric_col2 = st.columns(2)
            
            consecutive_wins = 0
            
            for i in range(1, EPISODES + 1):
                success = agent.train_episode()
                
                if success:
                    consecutive_wins += 1
                else:
                    consecutive_wins = 0
                
                # Update UI
                if i % 10 == 0 or i == 1:
                    progress_bar.progress(i / EPISODES)
                    if SHOW_TRAINING:
                        status_text.text(f"Episode {i}/{EPISODES} | Epsilon: {agent.epsilon:.3f}")
                        avg_rew = np.mean(agent.history_rewards[-50:])
                        metric_col1.metric("Avg Reward", f"{avg_rew:.1f}")
                        metric_col2.metric("Win Streak", f"{consecutive_wins}")
                
                if consecutive_wins >= EARLY_STOP:
                    st.success(f"Early Stopping: Solved {EARLY_STOP} times in a row!")
                    break
            
            progress_bar.progress(100)
            
            # Final Solve & Visualize
            final_path, solved = agent.solve()
            
            # Update the main maze plot with the path
            with col1:
                place_holder_maze.pyplot(plot_maze(final_path, 
                    title=f"Result: {'SOLVED' if solved else 'FAILED'} ({len(final_path)} steps)"))
            
            # Plot Training Curves
            st.subheader("📈 Performance Analysis")
            fig_perf, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            
            # Rewards
            ax1.plot(agent.history_rewards, alpha=0.6)
            # Smooth line
            if len(agent.history_rewards) > 20:
                smooth = np.convolve(agent.history_rewards, np.ones(20)/20, mode='valid')
                ax1.plot(smooth, 'r-', linewidth=2)
            ax1.set_title("Rewards per Episode")
            ax1.set_xlabel("Episode")
            
            # Success Rate (Cumulative)
            w = 50
            if len(agent.history_success) > w:
                succ_rate = np.convolve(agent.history_success, np.ones(w)/w, mode='valid')
                ax2.plot(succ_rate, color='green')
            else:
                ax2.plot(agent.history_success)
            ax2.set_title("Success Rate (Moving Avg)")
            ax2.set_xlabel("Episode")
            ax2.set_ylim(0, 1.1)
            
            st.pyplot(fig_perf)

    else:
        st.info("Please Generate a Maze first.")

st.markdown("---")
st.caption("Developed by Nik | Powered by Streamlit & Python")
