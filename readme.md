# Memory-Assisted Reinforcement Learning 
By Annik Carson (Last updated July 2018)

# Basic
-- Project with Kyle Nealy Fall2020
Writing workflow to run experiments with Unity/OpenMaze environments. 
> To do: 
> - write tests for environments (make sure they all behave the same) 
> - gridworld environment to openai standards
> - representation learning as function of agent object
> - arbitration between MF and EC action selection as function of agent object
> - saving results of experiments / logging parameters 

## Experiments
Take agent and environment objects. Runs through trials, collects data from agent-environment interactions. Determines whether agent uses model-free or episodic control for action selection (later this will be a feature of the agent class controlled by an arbitration module). 

Logs data collected during runs. Currently this is saved to csv using a unique ID which all results share (i.e. learned network weights, dictionary containing results of trial, etc) 


## Environment
An OpenAI style environment. Custom gridworld environment and imported environments from Unity are made to fit the OpenAI scheme such that all environments behave similarly. 

## Agent 
Takes network object, optional memory (episodic) object. Function get_action(obs) takes the observation from the environment and returns an action. get_action() refers to model free (network based) action selection by default. Can be changed to select action from Episodic module (later agent will have an arbitration function which assigns get_action to either MF_action or EC_action based on feedback about agent's performance). 

To Do: Representation learning module -- a separate network trained to learn a latent representation of state observations in a goal-independent manner (i.e. prior to training the agent's model-free network on a reward task). Learned representations (rather than environment observations) will be used by both model-free and episodic modules for producing behaviour.

## Networks
Networks are designed to take environment observation information and return an appropriate action. Currently we have both convolutional and fully connected networks to manage different types of environment observations. When the agent class is modified to include representation learning, we will predominantly rely on fully connected networks which learn from these vectorized state representations. 





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



