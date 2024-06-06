# Dataset Synthesis
In this folder, you find scripts to synthesize a large dataset of states and optimal actions from an MPC, that performs cartpole swingup and stabilization.

We recommend using a virtual enviorment for this:
```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Than you can start the dataset synthesis in parallel on 12 cores to produce 50 samples per core by calling:
```
python3 main_sample.py parallel_sample \
    --instances=12 \
    --samplesperinstance=50 
```

