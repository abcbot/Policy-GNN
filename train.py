import torch
from dqn_agent_pytorch import DQNAgent
from env.gcn import gcn_env
import numpy as np
import os
import random
from copy import deepcopy


def main():
    torch.backends.cudnn.deterministic=True
    max_timesteps = 10
    ### Experiment Settings ###
    # Cora
    dataset = 'Cora'
    max_episodes = 325

    # Citeseer
    #dataset = 'CiteSeer'
    #max_episodes = 230

    # Pubmed
    #dataset = 'PubMed'
    #max_episodes = 220
    ### Experiment Settings ###

    env = gcn_env(dataset=dataset, max_layer=5)
    env.seed(0)
    agent = DQNAgent(scope='dqn',
                    action_num = env.action_num,
                    replay_memory_size=int(1e4),
                    replay_memory_init_size=500,
                    norm_step=200,
                    state_shape = env.observation_space.shape,
                    mlp_layers=[32, 64, 128, 64, 32],
                    device=torch.device('cpu')
            )
    env.policy = agent
    best_val = 0.0
    test_acc = 0.0
    best_test = 0.0
    # Training meta-policy
    print("Training Meta-policy on Validation Set")
    for i_episode in range(1, max_episodes+1):
        loss, reward, (val_acc, reward) = agent.learn(env, max_timesteps) # debug = (val_acc, reward)
        if val_acc > best_val: # check whether gain improvement on validation set
            best_policy = deepcopy(agent)
        best_val = val_acc 
        print("Training Meta-policy:", i_episode, "; Avg_reward:", reward, "; Val_Acc:", val_acc)

    # Apply meta-policy to re-train GNN
    print("Training GNNs with learned meta-policy")
    new_env = gcn_env(dataset=dataset, max_layer=5)
    new_env.policy = best_policy
    state = new_env.reset2()
    for i_episode in range(1, 1000):
        action = best_policy.eval_step(state)
        state, reward, done, (val_acc, reward) = new_env.step2(action)
        test_acc = new_env.test_batch()
        print("Training GNN", i_episode, "; Val_Acc:", val_acc, "; Test_Acc:", test_acc)

if __name__ == "__main__":
    main()
