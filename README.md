## About this project
This consists of the code and results from a final project for Stanford's CS 224N (NLP with Deep Learning) course, completed with Arafat Mohammed. Since a key pain point of current neural QA-focused NLP systems is the lack of generalization to never-before-seen data domains, the goal of this project was to use the Reptile meta-learning algorithm applied to multiple prelearning tasks — which we interpret to be topics from within a single dataset — to create a metalearner on which we test out-of-domain QA, in order to hopefully show that this model would be more robust than our given baseline transformer-based QA model.

## Starter code for robustqa track
- Download datasets from [here](https://drive.google.com/file/d/1Fv2d30hY-2niU7t61ktnMsi_HUXS6-Qx/view?usp=sharing)
- Setup environment with `conda env create -f environment.yml`
- Train a baseline MTL system with `python train.py --do-train --eval-every 2000 --run-name baseline`
- Evaluate the system on test set with `python train.py --do-eval --sub-file mtl_submission.csv --save-dir save/baseline-01`
- Upload the csv file in `save/baseline-01` to the test leaderboard. For the validation leaderboard, run `python train.py --do-eval --sub-file mtl_submission_val.csv --save-dir save/baseline-01 --eval-dir datasets/oodomain_val`
