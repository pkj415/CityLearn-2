from citylearn import  CityLearn, building_loader, auto_size
from energy_models import HeatPump, EnergyStorage, Building
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
np.random.seed(3)

building_ids = [8]

data_folder = Path("data/")

demand_file = data_folder / "AustinResidential_TH.csv"
weather_file = data_folder / 'Austin_Airp_TX-hour.csv'

heat_pump, heat_tank, cooling_tank = {}, {}, {}

#Ref: Assessment of energy efficiency in electric storage water heaters (2008 Energy and Buildings)
loss_factor = 0.19/24
buildings = []
for uid in building_ids:
    heat_pump[uid] = HeatPump(nominal_power = 9e12, eta_tech = 0.22, t_target_heating = 45, t_target_cooling = 10)
    heat_tank[uid] = EnergyStorage(capacity = 9e12, loss_coeff = loss_factor)
    cooling_tank[uid] = EnergyStorage(capacity = 9e12, loss_coeff = loss_factor)
    buildings.append(Building(uid, heating_storage = heat_tank[uid], cooling_storage = cooling_tank[uid], heating_device = heat_pump[uid], cooling_device = heat_pump[uid]))
    buildings[-1].state_space(np.array([24.0, 40.0, 1.001]), np.array([1.0, 17.0, -0.001]))
    buildings[-1].action_space(np.array([0.2]), np.array([-0.2]))
    
building_loader(demand_file, weather_file, buildings)  
auto_size(buildings, t_target_heating = 45, t_target_cooling = 10)

env = CityLearn(demand_file, weather_file, buildings = buildings, time_resolution = 1, simulation_period = (3500,6000))

from reward_function import reward_function
from agent import Q_Learning, TD3_Agents #RL_Agents

#Extracting the state-action spaces from the buildings to feed them to the agent(s)
observations_space, actions_space = [],[]
for building in buildings:
    observations_space.append(building.observation_spaces)
    actions_space.append(building.action_spaces)

#Instantiatiing the control agent(s)
agents = TD3_Agents(observations_space,actions_space)
agents = Q_Learning()

k = 0
cost, cum_reward = {}, {}
'''
Every episode runs the RL controller for over 2500 hours in the simulation (controls the storage of cooling energy over about 3 Summer months)
Multiple episodes will run the RL controller over the same period over and over again, improving the policy and maximizing the score (less negative scores are better)
CHALLENGE:
Running the RL algorithm to coordinate as many buildings as possible (at least 3 or 4) and find a good control policy in the minimum number of episodes as possible (ideally within a single episode)
'''
episodes = 100
for e in range(episodes): #A stopping criterion can be added, which is based on whether the cost has reached some specific threshold or is no longer improving
    cum_reward[e] = 0
    state = env.reset()
    # print("Init", state)
    # print(buildings[0].sim_results['hour'][3500])
    # print(buildings[0].sim_results['t_out'][3500:6001].describe())

    # break
    done = False
    while not done:
        if k%500==0:
            print('hour: '+str(k)+' of '+str(2500*episodes))
        # print("State b4", state)
        action = agents.select_action(state, e, episodes)
        # print("State", state)
        # print("Actions", action)
        next_state, reward, done, _ = env.step([action])
        # print("Next State", next_state)
        reward = reward_function(reward) #See comments in reward_function.py
        agents.add_to_batch(state, action, reward, next_state, done, e, episodes)
        state = next_state
        cum_reward[e] += reward[0]
        # break
        
        k+=1
    cost[e] = env.cost()

print(cum_reward)
print(cost)