# Memory-Assisted Reinforcement Learning 
By Annik Carson

# Basic
-- Project with Kyle Nealy Fall2020
Writing workflow to run experiments with Unity/OpenMaze environments. 
> To do: 
> - write tests for environments (make sure they all behave the same) 
> - gridworld environment to openai standards
> - representation learning as function of agent object
> - arbitration between MF and EC action selection as function of agent object
> - saving results of experiments / logging parameters 

## File Structure
```
basic
│   README.md    
│
└───Agents
|   |    Define class which is basic operator in Environment
|   |    Agent basic functionality is to take in observation from Environment and produce action
|   |    Standard agent contains:
|   |       - Model-Free Control Network (basic functional unit to produce action from state information)
|   |       - Episodic Control Module (optional) - produce action from state information
|   |       - Representation Learning Module - produce state representation from Env observation
|   |       - Transition Cache - basic storage unit for keeping track of encountered states
│   └─── Networks
│       │   - Network objects to be used by Agent class for model-free control
│   └─── EpisodicMemory
│       │   - Episodic Memory object to be used by Agent class for episodic control
│   └─── RepresentationLearning [** Unfinished **]
│       │   - Network to learn state representations from observations provided by Environment
│   └─── TransitionCache
|       |   - Structural object to store state experiences
|
└───Envs
|   └─── Gridworld (python as openai-like environment)   
|   └─── Unity Environments (to be used with openai gym wrapper)
|   |   └─── Windows
|   |   └─── Linux
|
└───Experiment
|   |   Define class for standard experiment run 
|   |   Stores data from runs
|   |   Save elements to appropriate output with unique run ID 
|
└──Tests
|   |   Run tests to make sure each element works as expected
|
└───Utils
|   |   Basic functions to be used across packages
|   |   Plotting functions
|
└───Data
|   |   For data storage -- currently unused
|
└───Analysis
|   |   Functions for analyzing collected data -- currently unused


 
```

## Experiments
Top level to interface with. Collects and logs data from trial runs including total reward and loss. 
Currently data saved to csv using a unique ID which all results share (i.e. learned network weights, dictionary containing results of trial, etc)
CSV file stores unique id along with all parameters used in experiment. 

Arguments: agent object, environment object

Returns: Data collected over experiment run


## Environment
Working on getting all environments to work as openai gym environments. Currently using unity environments
with openai-gym wrapper and a gridworld environment written in python in the style of openai environments.

All environments must have functions to reset at start of trial and to take a step at each event. 

The central function of Environment is step().
The step() function takes action information and produces next state, reward, information about task completion 
('done'), and additional information for debugging if necessary. 

## Agent 
Learner in environment. 
Model-free control via a network object (see Agents/Networks/) takes state information and produces a 
policy and value estimate. State information can be either raw observations from the environment or representations
learned by the representationlearning module of the agent

Episodic control via a episodicmemory object (see Agents/EpisodicMemory/) takes state information
and queries a dictionary of saved states and experienced returns. State information should be the same as that passed to
the model free controller (i.e. either raw observation from environment or learned representation). This module is optional.

To Do: 
> * Representation learning module -- a separate network trained to learn a latent representation of state observations 
> in a goal-independent manner (i.e. prior to training the agent's model-free network on a reward task). Learned 
> representations (rather than environment observations) will be used by both model-free and episodic modules for 
> producing behaviour.
>
> * Arbitration module to control action selection between model-free control and episodic control 

## Networks
Define neural networks to learn policy and value functions from state information. Neural network styles (CNN, fully 
connected) are defined as separate files in Agents/Networks/. 

To Do:
> * Write script to generate network based on size/shape/type of environment observation
> or learned representation (depending on what will be used by agent)




# Episodic Control
pre- Oct 2020. Experiments using gridworld environment and basic episodic control. 
#### Environments 
Gridworld or OpenAI gym environments which create the tasks to be solved by the RL network

#### RL Network
Standard RL architecture we develop is an actorcritic network. Can also use Q-learning, etc. 

#### Memory
Episodic caching system used to assist the RL network

#### Sensory
Networks used to create efficient representations of incoming state information. Can be used to supplement the RL network. These may be modified autoencoders, etc.

#### Notebooks
Jupyter notebooks used for running code

#### Data
Storage of data from runs for later analysis



### Example Code



