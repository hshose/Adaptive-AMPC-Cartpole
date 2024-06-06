from tester import run_testing
import argparse
import os

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
    run_testing.run(model_path=f"{os.path.expanduser('~')}/hpc_data/pendulum_swingupLinearLoss/trainer",
                    iteration=0 if parser.iter_id is None else parser.iter_id,
                    num_layers_contr=5, num_neurons_contr=50,
                    num_layers_aug=8, num_neurons_aug=50,
                    num_init_points=200)
