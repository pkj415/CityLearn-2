import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from citylearn import  CityLearn, building_loader, auto_size
from energy_models import HeatPump, EnergyStorage, Building


def value_iteration(env, theta: float):
    # Temperature Range: 17.7 - 39.1 discretize to 17 - 39 (or 0 - 23)
    # Time: 1 - 24
    # Charge State: 0.0 - 1.0 (11 discrete steps)
    # Action: -0.5 - 0.5 (11 discrete steps)

    V = np.zeros((24, 24, 11))
    delta = np.inf
    while delta >= theta:
        old_V = V
        
        delta = np.max(np.abs(old_V - V))



def get_cost_of_building(building_uid: int, **kwargs):
    building_ids = [building_uid] #[i for i in range(8,77)]

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
        buildings[-1].action_space(np.array([0.5]), np.array([-0.5]))

    building_loader(demand_file, weather_file, buildings)  
    auto_size(buildings, t_target_heating = 45, t_target_cooling = 10)

    env = CityLearn(demand_file, weather_file, buildings = buildings, time_resolution = 1,
        simulation_period = (kwargs["start_time"], kwargs["end_time"]))

    return value_iteration(env, 1e-4)

parser = argparse.ArgumentParser()
parser.add_argument('--num_tiles', help='Num of action/charge levels. Must be odd',
                     type=int, required=True)
parser.add_argument('--min_action_val', help='Min action value >= -1.',
                     type=float, default=-1.)
parser.add_argument('--max_action_val', help='Max action value <= 1.',
                     type=float, default=1.)
parser.add_argument('--min_charge_val', help='Min charge value >= 0.',
                     type=float, default=0.)
parser.add_argument('--max_charge_val', help='Max charge value <= 1.',
                     type=float, default=1.)
parser.add_argument('--start_time', help='Start hour >= 3500 <= 6000', 
                     type=int, default=3500)
parser.add_argument('--end_time', help='End hour', type=int, required=True)
parser.add_argument('--building_uid', help='Building UID', type=int, required=True)
args = parser.parse_args()

elect_consump = get_cost_of_building(args.building_uid, start_time=args.start_time, end_time=args.end_time,
    num_tiles=args.num_tiles, min_action_val=args.min_action_val, max_action_val=args.max_action_val,
    min_charge_val=args.min_action_val, max_charge_val=args.max_action_val)