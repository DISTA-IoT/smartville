#!/bin/sh

# Use these commands to reproduce the experiments in the TIGER paper:
# For more information on the parameter values, see the tiger.py module

/pox/pox.py samples.pretty_log smartController.tiger --wb_tracking=True --ai_debug=True --multi_class=True \
    --init_k_shot=4 --batch_size=16 --node_features=False --wb_project_name=TIGER --wb_run_name=DDQN \
    --online_evaluation=False --report_step_freq=50 --inference_freq_secs=1 --online_evaluation_rounds=10 \
    --min_budget=-10 --max_budget=25 --max_episode_steps=750 --greedy_decay=0.999 --use_neural_AD=True \
    --use_neural_KR=True --cti_price_factor=4 --pretrained_inference=True --blocked_benign_cost_factor=40 --agent=DQN 

/pox/pox.py samples.pretty_log smartController.tiger --wb_tracking=True --ai_debug=True --multi_class=True \
    --init_k_shot=4 --batch_size=16 --node_features=False --wb_project_name=TIGER --wb_run_name=DDQN \
    --online_evaluation=False --report_step_freq=50 --inference_freq_secs=1 --online_evaluation_rounds=10 \
    --min_budget=-10 --max_budget=25 --max_episode_steps=750 --greedy_decay=0.999 --use_neural_AD=True \
    --use_neural_KR=True --cti_price_factor=4 --pretrained_inference=True --blocked_benign_cost_factor=40 --agent=DDQN 