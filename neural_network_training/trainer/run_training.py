import jax.random
from torch.utils.data import DataLoader
import optax
import equinox as eqx
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import os

from neural_networks.jax_models import AMPCNN, AMPCAUGNN
from neural_networks import jax_loss
from trainer import utils
from trainer.early_stopping import EarlyStopping
from data.dataset import AMPCDataset, normalize, denormalize

from tester import run_testing

import torch
import numpy as np
from tqdm import tqdm

import time

import jax.numpy as jnp

import yaml
import pickle


def loss(y, y_pred):
    if len(y_pred.shape) == 1 or True:
        return 1.0*jnp.mean(jnp.abs(y - y_pred)**1.5) + jnp.mean(jnp.abs(y - y_pred)) * 0.1
    else:
        return 0.0*jnp.mean(jnp.abs((y - y_pred) / (y_pred+1e-2)) ** 2) + jnp.mean(jnp.abs((y - y_pred)))


def loss_func(model, sys_state, y_pred):
    def model_wrapper(x_):
        # per batch, we use the same rng key for all elements in the batch. Makes calculation faster
        return model(x_)

    y = jax.vmap(model_wrapper)(sys_state)

    return jnp.mean(jax.vmap(loss)(y, y_pred))


@eqx.filter_jit
def loss_filtered(model, sys_state, y):
    return loss_func(model, sys_state, y)


@eqx.filter_jit
def make_step(model, opt_state, optim, sys_state, y):
    loss_value, grads = eqx.filter_value_and_grad(loss_func)(model, sys_state, y)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


class AMPCTrainer:
    def __init__(self, params, iteration, model_nom=None):

        self.__params = params
        self.__hp_comb_string = f"{params['num_layers']}x{params['num_neurons']}" + ("_aug" if params["train_aug"] else "")
        self.__model_nom = model_nom

        rng_key = jax.random.PRNGKey(iteration+1)
        self.__rng_key, init_key = jax.random.split(rng_key)

        # load data
        print("Loading data...")
        self.__dataset = AMPCDataset(params)
        print("Loading data completed.")

        # init model
        if not params["train_aug"]:
            self.__model = AMPCNN(num_layers=params["num_layers"], num_neurons=params["num_neurons"],
                                  num_sys_states=self.__dataset.num_sys_states, num_sys_inputs=self.__dataset.num_sys_inputs,
                                  num_aug_params=self.__dataset.num_aug_params, rng_key=init_key,
                                  activation_function=params["activation_function"])
        else:
            self.__model = AMPCAUGNN(num_layers=params["num_layers"], num_neurons=params["num_neurons"],
                   num_sys_states=self.__dataset.num_sys_states, num_sys_inputs=self.__dataset.num_sys_inputs,
                   num_aug_params=self.__dataset.num_aug_params, rng_key=init_key,
                   activation_function=params["activation_function"])

        # Initializing logger
        self.__metrics = utils.Metrics(gamma=0.0)  # Default gamma is too high for evaluation loss, thus EarlyStopping
        self.__tb_writer = SummaryWriter(log_dir=str(self.get_tensorboard_path()))

        self.__iteration = iteration

        self.__name_training_vals = "sys_input" if not params["train_aug"] else "params_aug_gradient"

        self.__train_dl = utils.get_dataloader(self.__dataset.train_ds, batch_size=self.__params["batch_size"],
                                               num_workers=self.__params["dataloader_num_workers"])
        self.__eval_dl = utils.get_dataloader(self.__dataset.eval_ds,
                                              batch_size=self.__params["batch_size_testing"],
                                              num_workers=self.__params["dataloader_num_workers"])
        #self.__test_dl = utils.get_dataloader(self.__dataset.test_ds,
        #                                      batch_size=self.__params["batch_size_testing"],
        #                                      num_workers=self.__params["dataloader_num_workers"])

        self.__early_stopping_patience = params["early_stopping_patience"]

        scheduler = optax.piecewise_constant_schedule(init_value=params["learning_rate"],
                                                      boundaries_and_scales={100*len(self.__train_dl): 0.5 for i in range(1, 10)})
        self.__jax_optim = optax.adamw(learning_rate=scheduler)
        self.__opt_state = self.__jax_optim.init(eqx.filter(self.__model, eqx.is_array))

        self.__rng_key, rng_key_init_values = jax.random.split(self.__rng_key)
        self.__testing_points = run_testing.get_random_init_points(rng_key_init_values, 10)

        self.__testing_points = self.__testing_points.at[0].set(jnp.array([0, -3.1415926535, 0, 0]))
        self.__testing_points = self.__testing_points.at[1].set(jnp.array([0, +3.1415926535, 0, 0]))

        self.__simulator = run_testing.get_simulator(params)

        if not os.path.exists(self.get_iter_id_path(iteration)):
            # Create a new directory because it does not exist
            os.makedirs(self.get_iter_id_path(iteration))

        with open(self.get_iter_id_path(iteration) / "params.yaml", 'w') as outfile:
            yaml.dump(params, outfile, default_flow_style=False)

        with open(self.get_iter_id_path(iteration) / "normalization.p", 'wb') as outfile:
            pickle.dump(self.__dataset.normalization, outfile)

    def run_training(self):
        early_stopping = EarlyStopping(path=str(self.get_iter_id_path(self.__iteration)),
                                       delta=0.00001, patience=self.__early_stopping_patience)
        print(f"\nInstantiated EarlyStopping with delta=0.005")
        for epoch in range(self.__params["max_epochs"]):
            start = time.time()
            print(f"\n------- Start: Epoch {epoch + 1} -------")

            # Training loop on all batches
            print(f"Epoch {epoch + 1}/{self.__params['max_epochs']}: start trainer on {len(self.__train_dl)} batches...")
            for batch_nr, batch in tqdm(enumerate(self.__train_dl)):
                loss, _ = self.__train_one_step(batch)
                self.__tb_writer.add_scalar("Loss/trainer", np.array(loss), (epoch + 1) * len(
                    self.__train_dl) + batch_nr)  # 1 TensorBoard step = model exposed to 1 batch

            print(f"\nEpoch {epoch + 1}/{self.__params['max_epochs']}: trainer finished")

            # Evaluating the model performance after each trainer epoch.
            print(f"Epoch {epoch + 1}/{self.__params['max_epochs']}: start evaluation...")

            self.evaluate(epoch)

            self.__model.save_model_to_file_temp(self.get_iter_id_path(iteration=self.__iteration))

            print(f"Epoch {epoch + 1}/{self.__params['max_epochs']}: evaluation finished")

            # Logging
            print(f"\nEpoch {epoch + 1}/{self.__params['max_epochs']}: trainer and validation results")
            print(f'Average train loss: {self.get_train_loss():.3f}')  # average train loss per epoch
            # if len(self.__eval_data)>0:  # don't really know why this if statement was used... wrap in try for safety
            try:
                print(f'Average eval loss: {self.get_eval_loss():.3f}')  # average eval loss per epoch
            except Exception as e:
                print(repr(e))
            early_stopping(self.get_eval_loss(), self.__model)

            if early_stopping.early_stop:
                print("Early stopping!")
                break
            print(f"Epoch took {time.time() - start}")

        self.__metrics.plot(self.get_iter_id_path(self.__iteration))
        self.__metrics.to_csv(self.get_iter_id_path(self.__iteration))
        print(f"\n------- Training done! -------\n")

    def __eval_one_step(self, batch):
        """Evaluate model on one batch
        Args:
            batch: Tuple with batch of images and targets from the evaluation set
        """
        loss_calc = loss_filtered(self.__model, batch["sys_state"].numpy(), y=batch[self.__name_training_vals].numpy())
        return loss_calc

    def __train_one_step(self, batch):
        """Train model on one batch
        Args:
            batch: Tuple with batch of inputs and targets from the trainer set
        """
        self.__model, self.__opt_state, loss_value = make_step(model=self.__model, opt_state=self.__opt_state,
                                                               optim=self.__jax_optim,
                                                               sys_state=batch["sys_state"].numpy(),
                                                               y=batch[self.__name_training_vals].numpy())
        self.__metrics.add('train_loss', torch.from_numpy(np.array(loss_value)))
        return loss_value, None

    def evaluate(self, epoch):
        """Evaluate model on entire evaluation set -> single batch! -> faster to iterate over eval_ds not eval_dl!"""
        # differentiation between DataLoader and Subset during __init__
        loss = 0
        accuracy = 0
        num_batches = 0
        for batch_nr, batch in tqdm(enumerate(self.__eval_dl)):
            part_loss = self.__eval_one_step(batch)
            loss += part_loss * len(batch["sys_state"])
            num_batches += len(batch["sys_state"])
        loss /= num_batches

        self.__tb_writer.add_scalar("Loss/evaluation", np.array(loss), (epoch + 1))
        self.__metrics.add('eval_loss', torch.from_numpy(np.array(loss)))
        if not self.__params["train_aug"]:
            loss_control = self.eval_control()
            self.__metrics.add('eval_loss_control', torch.from_numpy(np.array(loss_control)))
            self.__tb_writer.add_scalar("Loss/evaluation_control", np.array(loss_control), (epoch + 1))

    def eval_control(self):
        if not self.__params["train_aug"]:
            X = run_testing.simulate_plant(self.__testing_points, self.__simulator, self.__model,
                                           self.__dataset.normalization, self.__params)
        else:
            X = run_testing.simulate_plant(self.__testing_points, self.__simulator, self.__model_nom,
                                           self.__dataset.normalization, self.__params, model_aug=self.__model)

        if run_testing.check_constraints(X) and run_testing.check_final_state_constraints(X):
            return self.__metrics.get_history("eval_loss")[-1]  #jnp.mean(X[:, 0:2, :]**2)  #
        else:
            return 1e5

    def get_train_loss(self):
        """Returns average trainer loss from last epoch."""
        return torch.mean(torch.stack(self.__metrics.get_history("train_loss")[-len(self.__dataset.train_ds):]))

    def get_eval_loss(self):
        """Returns average validation loss from last epoch."""
        if not self.__params["train_aug"]:
            return self.__metrics.get_history("eval_loss_control")[-1]
        else:
            return self.__metrics.get_history("eval_loss")[-1]

    def get_log_path(self):
        return Path(self.__params["hpc_data_path"]) / "trainer" / f"model_{self.__hp_comb_string}"

    def get_iter_id_path(self, iteration):
        return self.get_log_path() / f"It{iteration}"

    def get_tensorboard_path(self):
        return self.get_log_path()

