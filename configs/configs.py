import json
from typing import Tuple
import os

def get_configs(env_name, agent_name, conf_dir='./configs') -> Tuple[dict, dict, dict, dict]:
    try:
        file_location = os.path.join(conf_dir, env_name)
        # Load the JSON with the configs for the selected environment
        with open(f"{file_location}.json") as json_file:
            params = json.load(json_file)

        # Return the agent's hyper params and training configs
        return params[agent_name]["agent_config"], params["trainer_config"], params["env_config"], params["tester_config"], params["tuner_config"], params
    except KeyError:
        print("Can't find agent_config/trainer_config/env_config combo for the current problem! Check JSON file in configs dir!")
