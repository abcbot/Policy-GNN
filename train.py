
import torch
from dqn_agent_pytorch import DQNAgent
from env.gcn import gcn_env
import numpy as np
import os

def main():
    ### Experiment Settings ###
    max_timesteps = 10
    max_episodes = 1000
    ### Experiment Settings ###

    dataset = 'Cora'
    #dataset = 'PubMed'
    #dataset = 'CiteSeer'
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
    fw = open(dataset+".csv", 'w')
    for i_episode in range(1, max_episodes+1):
        loss, reward, debug = agent.learn(env, max_timesteps) # debug = (val_acc, test_acc)
        if np.mean(debug[0]) > best_val: # check whether gain improvement on validation set
            test_acc = env.test_batch()
            if test_acc > best_test: # record the best testing accuracy
                best_test = test_acc
        best_val = np.mean(debug[0])
        print("Episode:", i_episode, "Avg_reward:", debug[1], "Val_Acc:", np.mean(debug[0]), "Test_Acc:", test_acc, "Best Test:", best_test)
        fw.write(str(i_episode)+","+str(debug[1])+","+str(np.mean(debug[0]))+","+str(test_acc)+","+str(best_test)+"\n")


if __name__ == "__main__":
    main()
