import json
from typing import Tuple

def get_configs(env_name, agent_name) -> Tuple[dict, dict, dict, dict]:
    try:
        # Load the JSON with the configs for the selected environment
        with open(f"./configs/{env_name}.json") as json_file:
            params = json.load(json_file)

        # Return the agent's hyper params and training configs
        return params[agent_name]["agent_config"], params["trainer_config"], params["env_config"], params["tester_config"]
    except KeyError:
        print("Can't find agent_config/trainer_config/env_config combo for the current problem! Check JSON file in configs dir!")
