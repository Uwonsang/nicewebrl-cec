# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Installation and Setup
```bash
# Basic installation
pip install -e .

# Install with specific optional dependencies
pip install -e ".[craftax]"      # For Craftax examples
pip install -e ".[jaxmarl]"      # For multi-agent examples (JaxMARL/Overcooked)
pip install -e ".[xland]"        # For XLand-Minigrid examples
pip install -e ".[xland-assistant]"  # For LLM assistant examples
```

### Running Examples
Examples are located in the `examples/` directory. Each example has its own web application:

```bash
# Navigate to example directory and run
cd examples/craftax && python web_app.py
cd examples/xland-minigrid && python web_app.py
cd examples/minigrid-ppo && python web_app.py
cd examples/overcooked-CEC && python web_app.py 'counter_circuit'
cd examples/xland-LLM-assistant && python web_app_assistant.py
```

### Code Quality and Testing
```bash
# Run linting (uses ruff)
ruff check .

# Run formatting
ruff format .

# Running individual example tests (where available)
cd examples/minigrid-ppo && python test.py
```

### Development Installation
```bash
# For development with editable install
pip install -e .

# Or with conda environment
conda create -n nicewebrl python=3.11 pip wheel -y
conda activate nicewebrl
pip install -e .
```

## Project Architecture

### Core Framework Structure
- **`nicewebrl/`** - Main library code
  - `experiment.py` - Core experiment orchestration (`Experiment`, `SimpleExperiment`)
  - `stages.py` - Stage management system (`Stage`, `EnvStage`, `FeedbackStage`, `Block`)
  - `container.py` - Base container class for data management
  - `nicejax.py` - JAX integration layer (`JaxWebEnv`, `MultiAgentJaxWebEnv`, `TimeStep`)
  - `utils.py` - Utility functions for user sessions, data handling, UI interactions
  - `logging.py` - Logging configuration
  - `data_analysis.py` - Analysis utilities for experiment data

### Key Design Patterns

**Experiment Flow**: The framework follows a stage-based experiment design where:
1. `Container` - Base class providing user data management and session handling
2. `Stage` - Individual experiment phases (instructions, environment interaction, feedback)
3. `Block` - Collections of stages that can be randomized or repeated
4. `Experiment` - Top-level orchestrator managing stage execution order

**JAX Integration**: 
- `JaxWebEnv` wraps JAX environments for web deployment with serialization
- `MultiAgentJaxWebEnv` extends this for multi-agent scenarios
- `TimeStep` standardizes environment step data across different JAX environments

**Web Interface**: Built on NiceGUI for Python-based UI development with real-time updates and user interaction handling.

### Example Structure
Each example in `examples/` typically contains:
- `web_app.py` - Main application entry point
- `experiment_structure.py` - Experiment configuration and stage definitions
- `README.md` - Specific setup and run instructions
- Optional: `requirements.txt`, `Dockerfile`, deployment configs

**Important**: Examples are Git submodules. When making changes to example directories:
1. Commit changes within each submodule directory first
2. Then commit in the parent repository to update submodule references

## Important Notes

### Environment Compatibility
The framework is designed to work with JAX-based RL environments, enabling the same environment code to be used for both:
- Training ML agents (high-performance JAX computation)
- Human subject experiments (web-based interaction)

### Multi-Agent Support
- Single-agent environments use `JaxWebEnv`
- Multi-agent environments use `MultiAgentJaxWebEnv`
- Framework supports human-human, human-AI, and AI-AI interaction modes

### Data Management
- User data is automatically saved using msgpack format
- Session management handles concurrent users
- Database integration via Tortoise ORM for experiment logging

## Technical Implementation Details

### JAX Environment Integration
The framework provides wrapper classes to bridge JAX environments with web interfaces:
- `JaxWebEnv` - Core wrapper for single-agent JAX environments
- `MultiAgentJaxWebEnv` - Extends wrapper for multi-agent scenarios
- Environments must implement standard JAX environment interface (reset, step)
- Observations are serialized for web transport using base64 encoding for images

### User Session Management
- Each user session maintains isolated state in `app.storage.user`
- Random number generators are seeded and tracked per session
- Session data persists across page reloads using NiceGUI's storage system
- Multi-user experiments coordinate through shared database state

### Stage System Design
Experiments are built using a hierarchical stage system:
1. **Stage** - Atomic experiment unit (instruction, environment interaction, feedback)
2. **Block** - Groups stages with optional randomization and repetition
3. **Experiment** - Orchestrates complete experimental flow across stages/blocks

### NiceGUI Integration Patterns
- UI components are created dynamically based on stage type
- Real-time updates use NiceGUI's reactive system
- Environment rendering converts JAX arrays to displayable images
- User interactions trigger JAX environment steps through async handlers