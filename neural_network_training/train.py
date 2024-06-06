import jax.random

from trainer.run_training import AMPCTrainer
import yaml
import argparse

from neural_networks.jax_models import AMPCNN


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training NN on HPC cluster")
    # comb_id = from sbatch --export=comb_id=X flag
    parser.add_argument("-c", "--comb_id", help="ID of the hyperparameter combination", type=int)
    # iter_id = SLURM_ARRAY_TASK_ID
    parser.add_argument("-i", "--iter_id", help="ID of the current iteration", type=int)

    parser.add_argument("-r", "--pruning_step", help="current pruning step", type=int)

    # slurm_job_id = SLURM_ARRAY_JOB_ID
    parser.add_argument("-s", "--slurm_job_id", help="ID of the SLURM job", type=int)

    parser.add_argument("-p", "--params", help="path to the .yaml file specifying the parameters", type=str)

    parser.add_argument("-d", "--dropout_prob", help="message loss probability", type=float)

    parser.add_argument("-n", "--name", help="name of parameter sweep", type=str)

    # parser.add_argument("-p", "--iteration", help="path to the .yaml file specifying the parameters", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    parser = parse_arguments()
    parameter_path = "parameters/pendulum.yaml"
    with open(parameter_path, "r") as file:
        params = yaml.safe_load(file)
    trainer = AMPCTrainer(params, 0 if parser.iter_id is None else parser.iter_id)
    #trainer.run_training()

    path = trainer.get_iter_id_path(0)
    model = AMPCNN(num_layers=params["num_layers"], num_neurons=params["num_neurons"],
                   num_sys_states=params["num_sys_states"], num_sys_inputs=params["num_sys_inputs"],
                   num_aug_params=params["num_aug_params"], rng_key=jax.random.PRNGKey(1),
                   activation_function=params["activation_function"])
    model = model.load_model_from_file(path)

    params["train_aug"] = True
    #params["learning_rate"] = 1e-3
    params["add_weight"] = 0.02
    #params["num_neurons"] = 100
    params["num_layers"] = 10
    params["activation_function"] = "tanh"
    AMPCTrainer(params, 0 if parser.iter_id is None else parser.iter_id, model).run_training()
