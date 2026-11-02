# Llama 2 70B LoRA Training for Super Computing 2026: Student Cluster Competition
## Introduction
This guide is designed for the Student Cluster Competition to walk participants through running and optimizing the MLPerf Training Benchmark using Llama 2 70B (LoRA Fine-Tuning) across various software and cluster hardware configurations. The goal is to minimize `Time-to-Train (TTT = run_stop - run_start)` while achieving the targeted convergence accuracy.
The benchmark task fine-tunes LLaMA 2 70B on the SCROLLS dataset (question-summary pairs from official government research reports) using Low-Rank Adaptation (LoRA), which optimizes rank decomposition matrices while keeping pre-trained weights frozen. Leads for this track are Miro Hodak (MLCommons/AMD) and Shriya Rishab (MLCommons/Nvidia).

## Scoring
In the SCC, your primary objective is to produce a compliant, fully verified MLPerf training submission. Teams will compete across three main scoring dimensions: Absolute Performance (lowest TTT), Relative Performance (efficient scaling given hardware capacity), and Technical Innovations (creative distributed optimizations).
To receive full credit, teams must complete exactly 10 genuine runs with distinct seeds. Scores are computed as the mean of the middle 8 runs using Olympic trimming (discarding the fastest and slowest runs). Up to 1 failed run is permitted.

## Convergence Requirements and RCP Rules
Every run must reach the quality target of `validation loss <= 0.925 (logged under MLLOG key eval_accuracy)`. Evaluations are conducted every `384 sequences (or CEIL(384 / GBS) * GBS steps if 384 is not divisible by GBS)`, skipping the initial `FLOOR(0.125 * GBS + 2)` evaluation checks.
All runs are subject to Reference Convergence Points (RCPs) defined in [rcps_llama2_70b_lora.json](https://github.com/mlcommons/logging/blob/master/mlperf_logging/rcp_checker/training_6.0.0/rcps_llama2_70b_lora.json). Normalization rules apply: implausibly fast runs fail the benchmark entirely, credibly fast runs are scaled up, and slow runs receive no downward adjustment.
## Hyperparameters, Parallelism, and Precision
Participants have full freedom to adjust unconstrained hyperparameters to optimize cluster performance, including peak learning rate `opt_base_learning_rate`, warmup ratio `opt_learning_rate_warmup_ratio`, total decay steps `opt_learning_rate_training_steps`, global batch size `global_batch_size`, and gradient accumulation steps `gradient_accumulation_steps`.
Any strategy for scale-out parallelism is permitted—including Data Parallelism (DP), Fully Sharded Data Parallelism (FSDP), Tensor Parallelism (TP), Pipeline Parallelism (PP), and Context Parallelism (CP). Any precision model is allowed (BFloat16 reference, FP8, or FP4).
## Artifacts to Submit
Teams must submit a complete package matching the expected folder structure:

- `TEAM_NAME/systems/student_system.json`: Hardware and system details.
- `TEAM_NAME/benchmarks/llama2_70b_lora/`: Execution scripts (README.md, run_and_time.sh, config.sh, source-or-patches/).
- `TEAM_NAME/results/student_system/llama2_70b_lora/`: 10 genuine result log files (result_0.txt through result_9.txt) and scaling.json.

Please submit these artifacts to the repo as a pull request. If you cannot create branches in the repo directly, please create a fork of this repo and submit a pull request from the fork. And sign the automated CLA check so we can merge your code into the repo.
## Run Commands and Verification Workflow
Ensure MLLOGGER integration emits machine-readable :::MLLOG JSON events on rank 0 with proper distributed barriers. To verify compliance before submission, execute the following commands:

```
# Install MLPerf Logging package
pip install "git+https://github.com/mlperf/logging.git@6.0.0-rc6"

# Run RCP Checker
python -m mlperf_logging.rcp_checker TEAM_NAME/results/student_system/llama2_70b_lora --rcp_usage training --rcp_version 6.0.0 --verbose

# Run Package Checker
python -m mlperf_logging.package_checker TEAM_NAME training 6.0.0 --werror --log_output /tmp/package_student_team.log

# Complete Official Verification Script
bash sources/logging/scripts/verify_for_v6.0_training.sh TEAM_NAME
```
