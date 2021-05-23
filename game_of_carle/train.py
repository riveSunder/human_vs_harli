import os
import time

import numpy as np

import torch
import torch.nn as nn

from carle.env import CARLE
from carle.mcl import AE2D, RND2D, CornerBonus

from game_of_carle.agents import ConvGRNNAgent, CARLA 
from game_of_carle.algos import CMAPopulation

def train(wrappers = [CornerBonus]):

    max_generations = int(1e3)
    max_steps = 768
    my_instances = 4
    number_steps = 0

    # defin environment and exploration bonus wrappers
    env = CARLE(instances = my_instances, use_cuda = True)

    my_device = env.my_device

    for wrapper in wrappers:

        env = wrapper(env)

    env.rules_from_string("B3/S345678")
    
    my_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"../policies/")
    #agent = ConvGRNNAgent(instances=my_instances, save_path=my_path)
    agent = CMAPopulation(CARLA, device="cuda", save_path=my_path)
    
    agent.population_size = 16
    agent.max_episodes = 1

    results = {"generation": [],\
            "fitness_max": [],\
            "fitness_min": [],\
            "fitness_mean": [],\
            "fitness std. dev.": []}

    for generation in range(max_generations):

        t0 = time.time()

        obs = env.reset()

        rewards = torch.Tensor([]).to(my_device)
        reward_sum = []
        number_steps = 0
        temp_generation = 0.0 + agent.generation

        while agent.generation <= temp_generation:
        #len(agent.fitness) <= (agent.population_size * agent.max_episodes):
            
            if number_steps >= max_steps:
                number_steps = 0
                reward_sum.append(np.sum(rewards.detach().cpu().numpy()))
                agent.step(reward_sum[-1])

                obs = env.reset()
                rewards = torch.Tensor([]).to(my_device)

            action = agent(obs)

            obs, reward, done, info = env.step(action)

            rewards = torch.cat([rewards, reward])
            number_steps += 1

        t1 = time.time()

        results["generation"].append(generation)
        results["fitness_max"].append(np.max(reward_sum))
        results["fitness_min"].append(np.min(reward_sum))
        results["fitness_mean"].append(np.mean(reward_sum))
        results["fitness std. dev."].append(np.std(reward_sum))

        np.save("carla_results.npy", results, allow_pickle=True)

        print("generation {}, mean, max, min, std. dev. fitness: ".format(generation), \
                 "{:.3e}, {:.3e}, {:.3e}, {:.3e}".format(\
                np.mean(reward_sum), np.max(reward_sum), np.min(reward_sum), \
                np.std(reward_sum)))
        print("steps per second = {:.4e}".format(\
                (env.inner_env.instances * max_steps \
                * agent.max_episodes * agent.population_size) / (t1 - t0)))
            


if __name__ == "__main__":

    train()
