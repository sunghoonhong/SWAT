from __future__ import print_function
import os
import torch
import utils
import numpy as np
from copy import deepcopy


def has_checkpoint(checkpoint_path, rb_path):
    """check if a checkpoint exists"""
    if not (os.path.exists(checkpoint_path) and os.path.exists(rb_path)):
        return False
    if len(os.listdir(rb_path)) == 0:
        return False
    return True

def cache_model(
    policy,
    total_timesteps,
    episode_num,
    num_samples,
    args,
    ):
    actor_state_dict = policy.actor.state_dict()
    critic_state_dict = policy.critic.state_dict()
    # for k, v in policy.actor.state_dict().items(): actor_state_dict[k] = v.cpu()
    # for k, v in policy.critic.state_dict().items(): critic_state_dict[k] = v.cpu()
    checkpoint = {
        "actor_state": deepcopy(actor_state_dict),
        "critic_state": deepcopy(critic_state_dict),
        "total_timesteps": total_timesteps,
        "episode_num": episode_num,
        "num_samples": num_samples,
        # "args": args,
    }
    return checkpoint

def save_model_from_cache(cache, checkpoint_path, model_name):
    fpath = os.path.join(checkpoint_path, model_name)
    # (over)write the checkpoint
    if not os.path.exists(fpath):
        torch.save(cache, fpath)
    return fpath

def save_model(
    checkpoint_path,
    policy,
    total_timesteps,
    episode_num,
    num_samples,
    replay_buffer,
    env_names,
    args,
    model_name="model.pyth",
):
    # change to default graph before saving
    policy.change_morphology(utils.getGraphDict([-1]))
    # Record the state
    checkpoint = {
        "actor_state": policy.actor.state_dict(),
        "critic_state": policy.critic.state_dict(),
        "actor_target_state": policy.actor_target.state_dict(),
        "critic_target_state": policy.critic_target.state_dict(),
        "actor_optimizer_state": policy.actor_optimizer.state_dict(),
        "critic_optimizer_state": policy.critic_optimizer.state_dict(),
        "total_timesteps": total_timesteps,
        "episode_num": episode_num,
        "num_samples": num_samples,
        # "args": args,
        # "rb_max": {name: replay_buffer[name].max_size for name in replay_buffer},
        # "rb_ptr": {name: replay_buffer[name].ptr for name in replay_buffer},
        # "rb_slicing_size": {
        #     name: replay_buffer[name].slicing_size for name in replay_buffer
        # },
    }
    fpath = os.path.join(checkpoint_path, model_name)
    # (over)write the checkpoint
    if not os.path.exists(fpath):
        torch.save(checkpoint, fpath)
    return fpath

def save_model_full(
    checkpoint_path,
    policy,
    total_timesteps,
    episode_num,
    num_samples,
    replay_buffer,
    env_names,
    args,
    model_name="model.pyth",
):
    # change to default graph before saving
    policy.change_morphology(utils.getGraphDict([-1]))
    # Record the state
    checkpoint = {
        "actor_state": policy.actor.state_dict(),
        "critic_state": policy.critic.state_dict(),
        "actor_target_state": policy.actor_target.state_dict(),
        "critic_target_state": policy.critic_target.state_dict(),
        "actor_optimizer_state": policy.actor_optimizer.state_dict(),
        "critic_optimizer_state": policy.critic_optimizer.state_dict(),
        "total_timesteps": total_timesteps,
        "episode_num": episode_num,
        "num_samples": num_samples,
        "args": args,
        "rb_max": {name: replay_buffer[name].max_size for name in replay_buffer},
        "rb_ptr": {name: replay_buffer[name].ptr for name in replay_buffer},
        "rb_slicing_size": {
            name: replay_buffer[name].slicing_size for name in replay_buffer
        },
    }
    fpath = os.path.join(checkpoint_path, model_name)
    # (over)write the checkpoint
    if not os.path.exists(fpath):
        torch.save(checkpoint, fpath)
    return fpath


def save_replay_buffer(rb_path, replay_buffer):
    # save replay buffer
    for name in replay_buffer:
        np.save(
            os.path.join(rb_path, "{}.npy".format(name)),
            np.array(replay_buffer[name].storage),
            allow_pickle=False,
        )
    return rb_path


def load_checkpoint(checkpoint_path, rb_path, policy, args):
    # load latest existing model checkpoint
    for cp_index in reversed(range(1, 21)):
        float_point = '' if cp_index == 1 else '.0'
        model_name = f'model_{cp_index}{float_point}.pyth' if cp_index < 20 else f'model.pyth'
        fpath = os.path.join(checkpoint_path, model_name)
        if os.path.exists(fpath):
            break
    
    checkpoint = torch.load(fpath, map_location="cpu")
    # change to default graph before loading
    policy.change_morphology(utils.getGraphDict([-1]))
    # load and return checkpoint
    policy.actor.load_state_dict(checkpoint["actor_state"])
    policy.critic.load_state_dict(checkpoint["critic_state"])
    policy.actor_target.load_state_dict(checkpoint["actor_target_state"])
    policy.critic_target.load_state_dict(checkpoint["critic_target_state"])
    policy.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state"])
    policy.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state"])
    # load replay buffer
    all_rb_files = [f[:-4] for f in os.listdir(rb_path) if ".npy" in f]
    all_rb_files.sort()
    replay_buffer_new = dict()
    for name in all_rb_files:
        if len(all_rb_files) > args.rb_max // 1e6:
            replay_buffer_new[name] = utils.ReplayBuffer(
                max_size=args.rb_max // len(all_rb_files)
            )
        else:
            replay_buffer_new[name] = utils.ReplayBuffer()
        replay_buffer_new[name].max_size = int(checkpoint["rb_max"][name])
        replay_buffer_new[name].ptr = int(checkpoint["rb_ptr"][name])
        replay_buffer_new[name].slicing_size = checkpoint["rb_slicing_size"][name]
        replay_buffer_new[name].storage = list(
            np.load(os.path.join(rb_path, "{}.npy".format(name)))
        )
    count_last_saving = float(cp_index + 1)
    return (
        checkpoint["total_timesteps"],
        checkpoint["episode_num"],
        replay_buffer_new,
        checkpoint["num_samples"],
        fpath,
        count_last_saving,
    )


def load_model_only(model_path, policy):
    if not os.path.exists(model_path):
        print(f'no model file found ... {model_path}')
        return
        # raise FileNotFoundError("no model file found... " + model_path)
    print("*** using model {} ***".format(model_path))
    # checkpoint = torch.load(model_path, map_location="cpu")
    checkpoint = torch.load(model_path)
    # change to default graph before loading
    policy.change_morphology(utils.getGraphDict([-1]))
    # load and return checkpoint
    policy.actor.load_state_dict(checkpoint["actor_state"])
    policy.critic.load_state_dict(checkpoint["critic_state"])
    policy.actor_target.load_state_dict(checkpoint["actor_state"])
    policy.critic_target.load_state_dict(checkpoint["critic_state"])
