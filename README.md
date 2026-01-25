rl_path_finding

Phase 1: Environment Implementation

This project implements the environment for a grid-based path finding task
used in a Reinforcement Learning assignment.

Phase 1 focuses exclusively on the environment logic and does NOT include
any learning or policy optimization.

Implemented features:
- Grid-based environment with multiple layouts
- Cell types: empty, obstacle, target, starting line
- State representation: (x, y, vx, vy)
- Discrete action space with velocity increments
- Velocity constraints and zero-velocity rule
- Step / transition logic
- Path-based collision detection (walls and obstacles)
- Reset logic after collision
- Episode termination when target is reached

The environment was tested independently using random actions before
proceeding to reinforcement learning.

Reinforcement Learning logic (Monte Carlo control) is implemented in Phase 2.
