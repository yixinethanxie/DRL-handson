#!/usr/bin/env python3
import gym
import ptan
import argparse

import torch
import torch.optim as optim

from tensorboardX import SummaryWriter

from lib import dqn_model, common


if __name__ == "__main__":
    params=common.HYPERPARAMS["pong"]
    device=torch.device("cuda")

    '''
    set up environment with following wrappers:
        1. episodic life, marke loss of life as done
        2. noop
        3. max and skip frames
        4. press fire after reset
        5. process frames to 84x84
        6. send np ndarray to torch tensor
        7. stack frames
        8. clip positive rewards to 1, and negative to -1
    '''
    env=ptan.common.wrappers.wrap_dqn(gym.make(params["env_name"]))

    run_name=params["run_name"]
    writer=SummaryWriter(comment=f"-{run_name}")

    net=dqn_model.DQN(env.observation_space.shape,env.action_space.n).to(device)

    '''
    create a deepcopy of the main NN (using copy.deepcopy)
    function sync makes the target NN load the main NN's parameters (state_dict())
    '''
    tgt_net=ptan.agent.TargetNet(net)

    '''
    default selector is argmax, e.g. select action that has the highest q value/output from the main NN
    epsilon (from 0 to 1), probability that the action is selected randomly instead of by selector
    '''
    selector=ptan.actions.EpsilonGreedyActionSelector(epsilon=params["epsilon_start"])

    '''
    1. starts from EPSILON_START
    2. decrease epsilon linearly by frames
    3. until epsilon hits EPSILON_FINAL
    '''
    epsilon_tracker=common.EpsilonTracker(selector,params)

    '''
    takes main NN and action selector
    override __call__ method
    when called, return action selected by selector from the outputs of main NN
    '''
    agent=ptan.agent.DQNAgent(net,selector,device=device)

    '''
    TODO: check code for ExperienceSource
    '''
    exp_source=ptan.experience.ExperienceSourceFirstLast(env,agent,gamma=params["gamma"],steps_count=1)

    '''
    can populate buffer to the specified buffer size
    can random sample from the buffer with a specified batch size
    '''
    buffer=ptan.experience.ExperienceReplayBuffer(exp_source,buffer_size=params["replay_size"])
    optimizer=optim.Adam(net.parameters(),lr=params["learning_rate"])

    frame_idx=0

    '''
    reward tracker is a context manager with following functions:
    1. write reward, frame idx, speed to console
    2. add performance (e.g. rewards) to summary writer for display on tensorboard
    3. check if stop reward is reached
    '''
    with common.RewardTracker(writer,params["stop_reward"]) as reward_tracker:
        while True:
            frame_idx +=1

            '''
            populate buffer by 1 sample for every iteration
            '''
            buffer.populate(1)

            '''
            update epsilon for every frame according to epsilon decay rule
            '''
            epsilon_tracker.frame(frame_idx)

            '''
            get total rewards and empty total rewards from experience source
            '''
            new_rewards=exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0],frame_idx,selector.epsilon):
                    break

            '''
            not train the NNs until buffer's size in more than replay inital
            '''
            if len(buffer)<params["replay_initial"]:
                continue

            optimizer.zero_grad()

            '''
            sample from experience buffer
            '''
            batch=buffer.sample(params["batch_size"])

            '''
            calculate loss between:
            1. actions' q values from main NN using current state
                (WARNING: this is not just the max of the outputs)
            2. output of bell equation:
                max of outputs from TARGET NN using next state*gamma + this step's reward
            '''
            loss_v=common.calc_loss_dqn(batch,net,tgt_net.target_model, gamma=params["gamma"],device=device)
            loss_v.backward()
            optimizer.step()

            '''
            sync target NN with main NN every target_net_sync steps
            '''
            if frame_idx % params["target_net_sync"]==0:
                tgt_net.sync()
