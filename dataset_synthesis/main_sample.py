# MIT License

# Copyright (c) 2024 Henrik Hose

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from nlmpc import Controller
from nlmpc import global_run
import os
import numpy as np
import csv
from tqdm import tqdm
import fire

from datetime import datetime
import subprocess
import sys
import shutil

from utils import MaximumReinitializations

def append_to_file(csv_file_path, data):
    if os.path.exists(csv_file_path):
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            for row in data:
                writer.writerow(row)
    else:
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            for row in data:
                writer.writerow(row)


def concat_csv(input_files, output_file):
    if os.path.exists(output_file):
        output_csvfile = open(output_file, mode='a', newline='')
    else:
        output_csvfile = open(output_file, mode='w', newline='')
    
    # # Open the output CSV file in append mode
    # with open(output_file, 'a', newline='') as output_csvfile:
    output_writer = csv.writer(output_csvfile)

    # Iterate through the list of input CSV files
    for input_file in tqdm(input_files):
        if os.path.exists(input_file):
            with open(input_file, 'r', newline='') as input_csvfile:
                input_reader = csv.reader(input_csvfile)

                # Iterate through the rows in the input CSV file and append them to the output CSV file
                for row in input_reader:
                    output_writer.writerow(row)

def delete_dirs(directories_to_delete):
    # Iterate through the list of directories
    for directory in directories_to_delete:
        try:
            # Use shutil.rmtree to recursively delete the directory and its contents
            shutil.rmtree(directory)
            print(f"Deleted directory: {directory}")
        except FileNotFoundError:
            print(f"Directory not found: {directory}")
        except Exception as e:
            print(f"Error deleting directory {directory}: {e}")

def concat_dataset(out="dataset", delete_inputs=False):
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Specify the directory path you want to list folder names from
    directory_path = 'data'

    # Initialize an empty list to store folder names
    folder_names = []

    # Iterate over the items in the directory
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)

        # Check if the item is a directory (folder)
        if os.path.isdir(item_path):
            folder_names.append(item_path)
    
    concat_outdir = os.path.join("data", f"{now}_{out}")
    if not os.path.exists(concat_outdir):
        os.makedirs(concat_outdir)
    filenames = ["x0.csv","X.csv","U.csv","J.csv"]
    for file in filenames:
        print(f"Concatenating {file}...")
        concat_csv([os.path.join(dir, file) for dir in folder_names],
                   os.path.join(concat_outdir, file))
    
    if delete_inputs:
        delete_dirs(folder_names)

def sample(
    outdir="",
    numberofsamples=int(100),
    nearterminal=False
    ):
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    x0_csv_file_path = os.path.join(outdir,"x0.csv")
    X_csv_file_path = os.path.join(outdir, "X.csv")
    U_csv_file_path = os.path.join(outdir, "U.csv")
    J_csv_file_path = os.path.join(outdir, "J.csv")

    dt = 160e-3
    N = 25
    
    np.savetxt(os.path.join(outdir, "dt.csv"), [dt])
    np.savetxt(os.path.join(outdir, "N.csv") , [N])
    
    x_sample_max = np.array([0.35, 2*np.pi, 1.2, 10])
    x_sample_min = np.array([-0.35, -2*np.pi, -1.2, -10])

    x_init = None
    u_init = None
    
    for i in tqdm(range(numberofsamples)):
        x0 =x_sample_min + np.random.rand(4) * ( x_sample_max - x_sample_min )
        try:
            X, U, J = global_run(x0, N, dt)
            append_to_file( x0_csv_file_path, [np.array( x0 ).flatten()] )
            append_to_file( X_csv_file_path,  [np.array( X.transpose()   ).flatten()] )
            append_to_file( U_csv_file_path,  [np.array( U.transpose()   ).flatten()] )
            append_to_file( J_csv_file_path,  [np.array( J.transpose()   ).flatten()] )
            x_init = X
            u_init = U
        except MaximumReinitializations as e:
            print(f"Error at initial condition {x0=}")
            print(e)

def parallel_sample(node_number=0, instances=4, samplesperinstance=5, nearterminal=False):
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    logdir = "logs"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
        
    processes = []
    outdirs = [os.path.join("data", f"{now}_{node_number}_{i}") for i in range(instances)]
    for i in range(instances):
        command = [
            "python3",
            "main_sample.py",
            "sample",
            "--outdir="+outdirs[i],
            "--numberofsamples="+str(samplesperinstance),
            "--nearterminal="+str(nearterminal)]

        with open(os.path.join('logs',f"{now}_{node_number}_{i}"+".log"),"wb") as out:
            p = subprocess.Popen(command,
                stdout=out, stderr=out)
            processes.append(p)

    for p in processes:
        p.wait()
    
    concat_outdir = os.path.join("data", f"{now}_{node_number}_concat")
    if not os.path.exists(concat_outdir):
        os.makedirs(concat_outdir)
    filenames = ["x0.csv","X.csv","U.csv","J.csv"]
    for file in filenames:
        concat_csv([os.path.join(dir, file) for dir in outdirs],
                   os.path.join(concat_outdir, file))
    
    delete_dirs(outdirs)

if __name__=='__main__':
    fire.Fire({
        'sample':                   sample,
        'parallel_sample':       parallel_sample,
        'concat_dataset':       concat_dataset
        })
