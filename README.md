# Introduction
This repository is the official implementation of Adaptive-Solver (AS): Adaptive-Solver Framework for Dynamic Strategy Selection in Large Language Model Reasoning, which has been accepted by "Information Processing &amp; Management".

# Installation
pip install -r requirements.txt

# Quick Start
We provide the solving records of each candidate solver on training dataset, as shown in "run_pipeline_configuration/solver2result/gsm8k/train".
You need to fill in your own openai api-key in the beginning of "prediction_runner.py".
Then, we execute the following two steps:

**1. optimize phase: search for the optimal pipeline on the validation set**

python run_pipeline_configuration/main_adapt_solvers_MSPD.py

**2. evaluation phase: run our method AS_MSPD using the optimal pipeline on the test set**

python run_evaluation/main_AUTO_gsm8k.py
