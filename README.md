[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Deep Deterministic Policy Gradients (DDPG)
# Continuous Control Project

### Introduction

For this project, uses the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

For this project solves the environment using two separate versions of the Unity environment:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

### Solving the Environment using BOTH approaches

The project has been created so it can solve both the 1 and the 20 agents versions of the environment with a single code base. There are some pretty interesting results, showing that the 20 agents approach scored much better results and turned to be way more stable than the 1 agent approach.

The task is episodic, and in order to solve the environment,  the agent must averagde score of +30 over 100 consecutive episodes in both the 1 and 20 agents environment variations.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

   **I solved the environment using both the 1 and 20 agents approach**
   For my environment which was *windows based*, I downloaded both versions of the agents, and stored them with the following folder names:
  * `Reacher_01_Windows_x86_64`
  * `Reacher_20_Windows_x86_64`
  
   The code base has been tailored to advantage of both solutions. It simply deals with `x-agents` list, 
   * 1 agent solution as a `1-size length` list and the 
   * 20 agents solution as a `20-size length` list

3. Make sure you have downloaded and installed Anaconda. You can download it from https://www.anaconda.com/distribution/

4. Now you can create your environment. Since this environment refers to a Udacity project for the Deep Reinforcement Learning Nanodegree, lets call our environment `DRLND`.

    Linux or Mac:
    ```sh
    conda create --name drlnd python=3.6
    source activate drlnd
    ```
    
    Windows:
    ```sh
    conda create --name drlnd python=3.6 
    activate drlnd
    ```
    
    
5. We will be working with pytorch version 0.4.0 (an early version), so make sure that you install this version of pytorch first by typing:
    ```sh
    conda install pytorch=0.4.0 -c pytorch
    ```
    
6. Perform a minimal installation of the OpenAI Gym environment (see instructions here: https://github.com/openai/gym)

7. For the rest of the prerequisities please do type:
    ```sh
    pip install .
     ```
     The above line of code assumes that at the folder you are working, you have the `setup.py` which includes the UnityAgents and the `requirements.txt` file that contains other useful packages (that exist in that repository).
     
8. Create a Python execution backend for Jupyter for the drlnd environment
    ```sh
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```
     
Now you are not only ready to use the UnityAgents evnironment, but the OpenAI Gym as well.  You are all set to start playing with reinforcement learning environments! Yay!

Other useful utilities will also be installed if you follow these directions, including Jupyter Notebook, so consider the above installation guide as a complete guide to setup your RL environments!

### Instructions

Run the notebooks 
* `Continuous_Control_1_agent.ipynb` for the 1 Agent Solution  
* `Continuous_Control_20_agents.ipynb` for the 20 Agents Solution 

*Both notebooks are identical, they simply call the two environment variations, where you can see for the exactly same architecture, how the two different approaches compare to training and stability*

### Report

Other than than running the notebooks, I include for your convenience a `Report.pdf` file which contains the detailed instructions about how I approached this problem, leaving out the code and all complexities, but also compares how the exactly same architecture is affected by the 1 and the 20 agents environments during training.

Use this report file to your understanding about what really happens under the hood, without the hassle of the code.
