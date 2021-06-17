# Resetting the Wave: 
## A High Reward Strategy Employed by a Hebbian Cellular Automaton Policy, Effectively Gaming a Change-in-Center-of-Mass Mobility Reward Across Multiple B3/Sxxx Life-Like Rules.  

<div align="center">
<a href="https://github.com/riveSunder/harli_learning/blob/master/assets/harli_reset_wave_strategy.gif">
<img src="assets/harli_reset_wave_strategy_small.gif">
</a>
</div>

## Introduction

This is a discussion/demonstration of a strategy employed by a machine agent to generate high rewards in a cellular automata reinforcement learning environment, CARLE, that has a change-in-center-of-mass reward proxy wrapper applied. 

In my opinion the interesting things about the agent policy described here are:

* The agent policy acts on cellular automata universes following various Life-like rule sets, _and_ the agent itself is implemented as a (continuously valued) neural cellular automata policy.
* The agent policy parameters are initialized randomly at every call to `agent.reset`.
* Rather than evolving policy parameters directly, the policy consists of a set of learning parameters defining Hebbian learning rules that are optimized using a covariance matrix adaptation evolution strategy. 
* The wave/reset strategy employed by this policy generates a much higher mean reward (~3 to 40x higher) than the type of strategy this reward scheme was intended to promote, _i.e_ the design or discovery of spaceships and gliders.
* Unlike say, a lightweight spaceship, this strategy is effective across a range of Life-like rules, including the rules it was trained on (B3/S023, B3/S236) 


If any of that sounds interesting to you, read on. You may also want to check out the interactive demo of the reward-hacking policy from this experiment on mybinder:

[https://mybinder.org/v2/gh/riveSunder/harli_learning/master?urlpath=/proxy/5006/bokeh-app](https://mybinder.org/v2/gh/riveSunder/harli_learning/master?urlpath=/proxy/5006/bokeh-app)

You can also fork this repository and try your hand at replicating the experiment. 

```
experiment.py -mg  128  -ms  256  -p  32  -sm  1  -v  1  -d  cuda:1  -dim  128  -s 13 42 1337 -a  HARLI  -w  RND2D  SpeedDetector  -tr  B3/S023  B3/S236  B3/S237  B3/S238  -vr  B3/S23  -tag  _harli_glider_experiment_
```

## The Agent

The Hebbian Automata Reinforcement Learning Improviser (HARLI) agent is a neural cellular automaton-based meta-learning agent designed to operate in Cellular Automata Reinforcement Learning Environment (CARLE). In other words HARLI is a meta-learning cellular automata agent designed to meta-learn while interacting with a cellular automata environment. 

<div align="center">
<img src="assets/hofstadter.png">
[xkcd 917 by Randall Munroe](https://xkcd.com/917/)
</div>

HARLI takes inspiration in part from the differentiable neural cellular automata described in the [thread](https://distill.pub/2020/selforg/) on distill.pub on differentiable self-organizing systems, and the Hebbian learning from random parameter initializations is adapted from Najarro and Risi's [Meta-Learning through Hebbian Plasticity in Random Networks](https://arxiv.org/abs/2007.02686). The source code is available for your perusal [here](https://github.com/riveSunder/harli_learning/blob/master/game_of_carle/agents/harli.py).

## The Strategy

HARLI employs a strategy of generating waves of live cells and repeatedly resetting the CARLE environment. In practice this looks something like this:

<div align="center">
<img src="assets/strategy_demo_127.gif">
</div>

Note that if you'd like to see HARLI in action at a much better resolution, try out the [interactive demo](https://mybinder.org/v2/gh/riveSunder/harli_learning/master?urlpath=/proxy/5006/bokeh-app), or clone this repository and try running the [demonstration notebook](https://github.com/riveSunder/harli_learning/blob/master/notebooks/evaluation.ipynb) locally. 

Note that this reset wave strategy is far more rewarding than say, generating a single Life glider. A glider has an average reward around ~1.35 in this reward scheme, whereas the reset wave strategy can easily generate an average reward of 40 to 50 or more. 

<div align="center">
<img src="assets/glider_reward.gif">
</div>

## Effectiveness Across Rules

This strategy works pretty well across several Life-like rules closely related to [Conway's Game of Life](https://www.conwaylife.com/wiki/Conway%27s_Game_of_Life): B3/S023 (DotLife), B3/S236, B3/S237, B3/S238, which HARLI was trained on, and Life itself (B3/S23) that was used as the validation rules but not included during optimization steps. This strategy may appear a bit heavy-handed, but consider how success a strategy of just trying to build the lightweight spaceship from Game of Life in a few of the rule sets used in this experiment:

<div align="center">
<img src="assets/spaceships.gif">
</div>

Of the five rule sets (4 used during training), the lightweight spaceship only works in 2 of them.

## Closing Remarks

Although I was looking forward to training an agent that might rediscover canonical spaceships and gliders in Life-like cellular automata, this reward-hacking strategy was an interesting find in itself. Given the rewards generated by this strategy in comparison to conventional mobile artifacts in discrete CA, we certainly could not expect an agent trained with this reward scheme to prefer neat and tidy spaceships instead of a robust, lucrative, but messy reset wave strategy. It's more robust than any given spaceship pattern, as it works in quite a few different rule sets. It was a surprise, which is one of the goals of building a more open-ended RL environment like CARLE. 

This exercise also serves as a reminder that we can't in general expect machine or alien intelligences to behave according to what we would expect given a human's anthro-centric theory of mind, or even a theory of mind that would otherwise be reasonably good at guessing motives and predicting non-human animal behavior. It's only a thin slice of what can be explored in the space of machine creativity as applied to Life-like cellular automata, and it is also my hope to encourage ML/AI researchers to not just seek out specific expected behaviors, but to be open from learning from the unexpected.  

<em>
Good luck, thanks for reading, and I look forward to seeing what your creations create.  
</em>

