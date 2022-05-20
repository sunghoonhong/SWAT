from __future__ import print_function

import numpy as np
import torch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import utils
import TD3
import json
import time
from arguments import get_args
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import checkpoint as cp
from config import *
import wandb


def train(args):
    # Set up directories ===========================================================
    os.makedirs(DATA_DIR, exist_ok=True)
    exp_name = args.expID
    exp_path = os.path.join(DATA_DIR, exp_name)
    rb_path = os.path.join(BUFFER_DIR, exp_name)
    os.makedirs(exp_path, exist_ok=True)
    # save arguments
    with open(os.path.join(exp_path, "args.txt"), "w+") as f:
        json.dump(args.__dict__, f, indent=2)

    # Retrieve MuJoCo XML files for training ========================================
    envs_train_names = []
    
    # parse positional encoding methods
    args.traversal_types = ['pre', 'inlcrs', 'postlcrs']
                    
    args.graph_dicts = dict()
    args.graphs = dict()
    # existing envs
    if not args.custom_xml:
        for morphology in args.morphologies:
            envs_train_names += [
                name[:-4]
                for name in os.listdir(XML_DIR)
                if ".xml" in name and morphology in name
            ]
        for name in envs_train_names:
            args.graphs[name] = utils.getGraphStructure(
                os.path.join(XML_DIR, "{}.xml".format(name)),
                args.observation_graph_type,
            )
            if args.actor_type in ['transformer', 'smp']:
                args.graph_dicts[name] = {'parents': args.graphs[name]}
            else:
                args.graph_dicts[name] = utils.getGraphDict(
                    args.graphs[name], args.traversal_types,
                )
    # custom envs
    else:
        if os.path.isfile(args.custom_xml):
            assert ".xml" in os.path.basename(args.custom_xml), "No XML file found."
            name = os.path.basename(args.custom_xml)
            env_name = name[:-4]
            envs_train_names.append(env_name)  # truncate the .xml suffix
            args.graphs[env_name] = utils.getGraphStructure(
                args.custom_xml, args.observation_graph_type
            )
            if args.actor_type in ['transformer', 'smp']:
                args.graph_dicts[env_name] = {'parents': args.graphs[env_name]}
            else:
                args.graph_dicts[env_name] = utils.getGraphDict(
                    args.graphs[env_name], args.traversal_types
                )
        elif os.path.isdir(args.custom_xml):
            for name in os.listdir(args.custom_xml):
                if ".xml" in name:
                    env_name = name[:-4]
                    envs_train_names.append(env_name)
                    args.graphs[env_name] = utils.getGraphStructure(
                        os.path.join(args.custom_xml, name), args.observation_graph_type
                    )
                    if args.actor_type in ['transformer', 'smp']:
                        args.graph_dicts[env_name] = {'parents': args.graphs[env_name]}
                    else:
                        args.graph_dicts[env_name] = utils.getGraphDict(
                            args.graphs[env_name], args.traversal_types
                        )                    

    envs_train_names.sort()
    num_envs_train = len(envs_train_names)

    # Set up training env and policy ================================================
    args.limb_obs_size, args.max_action = utils.registerEnvs(
        envs_train_names, args.max_episode_steps, args.custom_xml
    )
    max_num_limbs = max([len(args.graphs[env_name]) for env_name in envs_train_names])
    # create vectorized training env
    obs_max_len = (
        max([len(args.graphs[env_name]) for env_name in envs_train_names])
        * args.limb_obs_size
    )
    envs_train = [
        utils.makeEnvWrapper(name, obs_max_len, args.seed) for name in envs_train_names
    ]
    envs_train = SubprocVecEnv(envs_train)  # vectorized env

    # set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # determine the maximum number of children in all the training envs
    if args.max_children is None:
        args.max_children = utils.findMaxChildren(envs_train_names, args.graphs)

    args.max_num_limbs = max_num_limbs
    if args.actor_type == 'swat':
        args.rel_size = args.graph_dicts[envs_train_names[0]]['relation'].shape[-1]
        
        
    # setup agent policy
    policy = TD3.TD3(args)

    # Create new training instance or load previous checkpoint ========================
    if cp.has_checkpoint(exp_path, rb_path):
        print("*** loading checkpoint from {} ***".format(exp_path))
        (
            total_timesteps,
            episode_num,
            replay_buffer,
            num_samples,
            loaded_path,
            count_last_saving
        ) = cp.load_checkpoint(exp_path, rb_path, policy, args)
        print("*** checkpoint loaded from {} ***".format(loaded_path))
    else:
        print("*** training from scratch ***")
        # init training vars
        total_timesteps = 0
        episode_num = 0
        num_samples = 0
        count_last_saving = 1   # start save from 2M
        # different replay buffer for each env; avoid using too much memory if there are too many envs
        replay_buffer = dict()
        if num_envs_train > args.rb_max // args.rb_per_env:
            for name in envs_train_names:
                replay_buffer[name] = utils.ReplayBuffer(
                    max_size=args.rb_max // num_envs_train
                )
        else:
            for name in envs_train_names:
                replay_buffer[name] = utils.ReplayBuffer(args.rb_per_env)
    # Initialize training variables ================================================
    s = time.time()
    this_training_timesteps = 0
    collect_done = True
    episode_timesteps_list = [0 for i in range(num_envs_train)]
    done_list = [True for i in range(num_envs_train)]

    # best_checkpoint_cache = None
    timestep_to_next_saving = args.max_timesteps // 20 # every 0.5M
    # timestep_to_next_saving_log = args.max_timesteps // 100 # every 0.1M
    
    count_last_saving_log = 1
    best_return = -1000

    # Start training ===========================================================
    stats = {'timestep': [], 'episode_return': []}
    for i in range(num_envs_train):
        stats[f"{envs_train_names[i]}_episode_reward"] = []
        stats[f"{envs_train_names[i]}_episode_len"] = []

    while total_timesteps < args.max_timesteps:
        # train and log after one episode for each env
        if collect_done:
            # log updates and train policy
            if this_training_timesteps != 0:
                policy.train(
                    replay_buffer,
                    episode_timesteps_list,
                    args.batch_size,
                    args.discount,
                    args.tau,
                    args.policy_noise,
                    args.noise_clip,
                    args.policy_freq,
                    graph_dicts=args.graph_dicts,
                    envs_train_names=envs_train_names[:num_envs_train]
                )
                
                # add to log & display
                stat = {
                    'timestep': total_timesteps,
                    'episode_return': np.mean(episode_reward_list),
                }
                for i in range(num_envs_train):
                    stat[f"{envs_train_names[i]}_episode_reward"] = float(episode_reward_list[i])
                    stat[f"{envs_train_names[i]}_episode_len"] = float(episode_timesteps_list[i])
                for key in stats:
                  stats[key].append(stat[key])
                if count_last_saving_log * timestep_to_next_saving_log < total_timesteps:
                    count_last_saving_log = total_timesteps // timestep_to_next_saving_log + 1
                    with open(os.path.join(exp_path, 'stat.json'), 'w') as f:
                        json.dump(stats, f, indent=2)
                    
                # save model
                if count_last_saving * timestep_to_next_saving < total_timesteps:
                    model_saved_path = cp.save_model(
                        exp_path,
                        policy,
                        total_timesteps,
                        episode_num,
                        num_samples,
                        replay_buffer,
                        envs_train_names,
                        args,
                        model_name=f"model_{count_last_saving}.pyth",
                    )
                    print(f"***{count_last_saving}-th checkpoint saved at {exp_path}/model_{count_last_saving}.pyth***")
                    count_last_saving = total_timesteps // timestep_to_next_saving + 1

            # reset training variables
            obs_list = envs_train.reset()
            done_list = [False for i in range(num_envs_train)]
            episode_reward_list = [0 for i in range(num_envs_train)]
            episode_timesteps_list = [0 for i in range(num_envs_train)]
            episode_num += num_envs_train
            # create reward buffer to store reward for one sub-env when it is not done
            episode_reward_list_buffer = [0 for i in range(num_envs_train)]

        # start sampling ===========================================================
        # sample action randomly for sometime and then according to the policy
        if total_timesteps < args.start_timesteps * num_envs_train:
            action_list = [
                np.random.uniform(
                    low=envs_train.action_space.low[0],
                    high=envs_train.action_space.high[0],
                    size=max_num_limbs,
                )
                for i in range(num_envs_train)
            ]
        else:
            action_list = []
            for i in range(num_envs_train):
                # dynamically change the graph structure of the modular policy
                policy.change_morphology(args.graph_dicts[envs_train_names[i]])

                # remove 0 padding of obs before feeding into the policy (trick for vectorized env)
                obs = np.array(
                    obs_list[i][
                        : args.limb_obs_size * len(args.graphs[envs_train_names[i]])
                    ]
                )
                policy_action = policy.select_action(obs)
                if args.expl_noise != 0:
                    policy_action = (
                        policy_action
                        + np.random.normal(0, args.expl_noise, size=policy_action.size)
                    ).clip(
                        envs_train.action_space.low[0], envs_train.action_space.high[0]
                    )
                # add 0-padding to ensure that size is the same for all envs
                policy_action = np.append(
                    policy_action,
                    np.array([0 for i in range(max_num_limbs - policy_action.size)]),
                )
                action_list.append(policy_action)

        # perform action in the environment
        new_obs_list, reward_list, curr_done_list, _ = envs_train.step(action_list)

        # record if each env has ever been 'done'
        done_list = [done_list[i] or curr_done_list[i] for i in range(num_envs_train)]

        for i in range(num_envs_train):
            # add the instant reward to the cumulative buffer
            # if any sub-env is done at the momoent, set the episode reward list to be the value in the buffer
            episode_reward_list_buffer[i] += reward_list[i]
            if curr_done_list[i] and episode_reward_list[i] == 0:
                episode_reward_list[i] = episode_reward_list_buffer[i]
                episode_reward_list_buffer[i] = 0
            done_bool = float(curr_done_list[i])
            if episode_timesteps_list[i] + 1 == args.max_episode_steps:
                done_bool = 0
                done_list[i] = True
            # remove 0 padding before storing in the replay buffer (trick for vectorized env)
            num_limbs = len(args.graphs[envs_train_names[i]])
            obs = np.array(obs_list[i][: args.limb_obs_size * num_limbs])
            new_obs = np.array(new_obs_list[i][: args.limb_obs_size * num_limbs])
            action = np.array(action_list[i][:num_limbs])
            # insert transition in the replay buffer
            replay_buffer[envs_train_names[i]].add(
                (obs, new_obs, action, reward_list[i], done_bool)
            )
            num_samples += 1
            # do not increment episode_timesteps if the sub-env has been 'done'
            if not done_list[i]:
                episode_timesteps_list[i] += 1
                total_timesteps += 1
                this_training_timesteps += 1

        obs_list = new_obs_list
        collect_done = all(done_list)

    # save checkpoint after training ===========================================================
    model_saved_path = cp.save_model(
        exp_path,
        policy,
        total_timesteps,
        episode_num,
        num_samples,
        replay_buffer,
        envs_train_names,
        args,
    )
    with open(os.path.join(exp_path, 'stat.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print("*** training finished and model saved to {} ***".format(model_saved_path))


if __name__ == "__main__":
    args = get_args()
    args.run_name = f'{args.label}_{args.seed}'
    train(args)
