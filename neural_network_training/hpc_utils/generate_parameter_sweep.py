import copy

import yaml
import os
from pathlib import Path
import copy


def sweep_parameters(params, params_sweep):
    params_results = []

    keys = list(params_sweep.keys())
    key = keys[0]

    if len(keys) != 1:
        new_params_sweep = copy.deepcopy(params_sweep)
        new_params_sweep.pop(key)

        new_params = sweep_parameters(params, new_params_sweep)
    else:
        new_params = [copy.deepcopy(params)]

    for v in params_sweep[key]:
        for p in new_params:
            params_results.append(copy.deepcopy(p))
            params_results[-1][key] = v

    return params_results


if __name__ == "__main__":
    name = "pendulum"
    param_target_path = f"{Path.home()}/hpc_parameters/{name}/"
    param_path = "../parameters/pendulum.yaml"
    param_sweep_path = "../parameters/pendulum_sweep.yaml"

    # param_path = "../src/parameters/electric_devices/electric_devices.yaml"
    # param_sweep_path = "../src/parameters/electric_devices/electric_devices_sweep.yaml"

    if not os.path.exists(param_target_path):
        os.makedirs(param_target_path)

    with open(param_path, "r") as file:
        params = yaml.safe_load(file)
    with open(param_sweep_path, "r") as file:
        params_sweep = yaml.safe_load(file)

    swept_params = sweep_parameters(params, params_sweep)
    for comb_idx in range(len(swept_params)):
        with open(param_target_path + f"params{comb_idx}.yaml", "w") as file:
            params = yaml.safe_dump(swept_params[comb_idx], file)


