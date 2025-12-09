# **RL Maze Solver**

## **1. Overview**

This project presents an advanced **Reinforcement Learning (RL) Maze Solver** capable of solving very large, procedurally generated mazes with remarkable efficiency. The system integrates:

* **SARSA** (on-policy TD control),
* **Prioritized Sweeping**,
* **Experience Replay**, and
* A novel **Distance-Gradient Reward Shaping Mechanism**.

Together, these components produce an agent that learns optimal navigation strategies in environments with sparse rewards and complex transition dynamics.

The implementation, including maze generation, agent learning, and visualization, is detailed in the codebase. The primary system resides in the Streamlit application logic. fileciteturn0file0

---

## **2. Key Contributions**

This project demonstrates several research-significant contributions:

### **2.1 Hybrid RL Architecture**

The agent blends three reinforcement learning paradigms:

* **On‑policy SARSA** for stable gradient-following behavior.
* **Model-based Planning** via Prioritized Sweeping.
* **Off‑policy-like Experience Replay**, enhancing sample efficiency.

This hybrid structure enables rapid convergence even in high‑dimensional mazes.

### **2.2 True Distance Gradient Reward (TDGR)**

A pre-computed BFS distance map serves as a *potential field*, allowing reward shaping from true shortest-path distances. This signal turns an otherwise sparse maze into a smooth optimization landscape.

Reward is computed as:

```
R = 2 * (D(current) - D(next)) - 0.1
```

This incentivizes movement toward the goal with minimal delay.

### **2.3 Curriculum through Heuristic Restarts**

The agent occasionally restarts episodes near previously successful states, accelerating convergence by sampling high-value regions more frequently.

---

## **3. System Architecture**

### **3.1 Maze Generation**

Mazes are generated using a randomized DFS (recursive backtracking) algorithm ensuring:

* Full connectivity
* No isolated regions
* Guaranteed solvability

Start and goal positions are fixed at opposite corners.

### **3.2 RL Agent Components**

| Component      | Description                                                            |
| -------------- | ---------------------------------------------------------------------- |
| Q‑Table        | Stores action-value pairs for each reachable state-action tuple        |
| Model          | Transition model `(state, action) → (next_state, reward)` for planning |
| Priority Queue | Maintains states requiring urgent updates (Prioritized Sweeping)       |
| Replay Buffer  | Stores experiences for mini‑batch learning                             |
| Distance Map   | BFS-derived true distance-to-goal gradient                             |

---

## **4. Experimental Results**

The RL Maze Solver was evaluated on multiple maze sizes. The agent demonstrated strong generalization and convergence behavior.

### **4.1 Convergence Performance**

| Maze Size     | Episodes to Solve | Notes                                   |
| ------------- | ----------------- | --------------------------------------- |
| **57 × 57**   | **448 episodes**  | Stable, monotonic improvement           |
| **65 × 65**   | **504 episodes**  | Smooth convergence; minimal oscillation |
| **101 × 101** | **1000 episodes** | Solved an extremely large maze reliably |

The 101×101 result is particularly significant—very few RL agents can solve such a large, sparse-reward maze within 1000 episodes without deep neural networks.

### **4.2 Path Optimality**

The final path discovered by the agent consistently approaches the BFS shortest path, enabled by the TDGR shaping signal.

### **4.3 Stability**

Priority-based TD updates (Prioritized Sweeping) substantially reduce the variance in convergence curves, even in mazes with large branching factors.

---

## **5. Algorithmic Workflow**

```
Initialize Q-table, model, replay buffer
Generate maze & compute distance map
For each episode:
    Select start state (curriculum-based)
    Repeat until goal or max steps:
        Choose action via ε-greedy
        Compute shaped reward
        Update Q using SARSA
        Store transition in model
        Insert transition into priority queue
        Perform prioritized planning updates
        Perform experience replay batch updates
    Decay ε
Test final policy using greedy evaluation
```

---

## **6. Visualizations**

The Streamlit application provides:

* **Maze rendering** with start/end markers
* **Training dash** showing: reward trend, success rate, Q‑table growth
* **Path visualization** during testing, with heat coloring

These tools support debugging, performance analysis, and reinforcement learning interpretability.

---

## **7. Strengths & Research Value**

### **7.1 Sample Efficiency**

The combination of reward shaping and prioritized planning yields rapid improvement in early episodes.

### **7.2 Scalability**

Demonstrated ability to scale to **101×101 mazes**, a benchmark rarely achieved without Deep RL.

### **7.3 Interpretability**

Since learning is tabular and model-aware, behavior is fully explainable—an excellent property for algorithmic research.

---

## **8. Applications**

Beyond maze navigation, the approach generalizes to problems with:

* Sparse reward environments
* Large discrete state spaces
* Well-defined goal conditions
* Expensive simulations benefiting from model-based planning

Examples:

* Robot motion planning
* Grid-based pathfinding in games
* Exploration tasks
* Puzzle solving

---

## **9. Future Work**

Potential research extensions:

* Integrating Dyna‑Q or Dyna‑2
* Neural Q‑Value approximators for continuous spaces
* Curriculum learning refinements
* Multi-agent cooperative maze solving
* Dynamic mazes (non-stationary environments)

---

## **10. Citation**

If using this project in academic or research settings, please include:

```
Debanik (2025). RL Maze Solver with SARSA, Prioritized Sweeping, and Distance-Gradient Reward Shaping. GitHub Repository.
```

---

## **11. License**

This project is open-source under the **MIT License**.
