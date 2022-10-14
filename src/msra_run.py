import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import json

# some modules 
from modules.latent_models import REGISTRY as model_REGISTRY
from modules.state_encoders import REGISTRY as state_enc_REGISTRY

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

import numpy as np

def msra_run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    # unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    remark_str = getattr(args, "remark", "NoRemark")
    
    if _config["env"] == 'lbf':
        players = _config["env_args"]["players"]
        max_player_level = _config["env_args"]["max_player_level"]
        field_size = _config["env_args"]["field_size"]
        max_food = _config["env_args"]["max_food"]
        sight = _config["env_args"]["sight"]
        force_coop = "-coop" if _config["env_args"]["force_coop"] else ""
        _config["env_args"]["key"] = "Foraging-{}x{}-{}p-{}f-s{}{}".format(
            str(field_size),
            str(field_size),
            str(players),
            str(max_food),
            str(sight),
            str(force_coop)
        )

    if _config["env"] == 'traffic_junction':
        nagents = _config["env_args"]["nagents"]
        dim = _config["env_args"]["dim"]
        vision = _config["env_args"]["vision"]
        difficulty = _config["env_args"]["difficulty"]
        _config["env_args"]["map_name"] = "traffic_junction-{}p-{}d-{}v-{}".format(
            str(nagents),
            str(dim),
            str(vision),
            difficulty,
        )

    if _config["env"] == 'stag_hunt':
        nagents = _config["env_args"]["n_agents"]
        nstags = _config["env_args"]["n_stags"]
        world_shape = _config["env_args"]["world_shape"]
        sight = _config["env_args"]["agent_obs"][0]
        _config["env_args"]["map_name"] = "stag_hunt-{}x{}-{}p-{}s-v{}".format(
            str(world_shape[0]),
            str(world_shape[1]),
            str(nagents),
            str(nstags),
            str(sight),
        )

    if _config["env"] == 'hallway':
        nagents = _config["env_args"]["n_agents"]
        state_numbers = _config["env_args"]["state_numbers"]
        state_str = "x".join([str(state_num) for state_num in state_numbers])
        _config["env_args"]["key"] = f"hallway-{state_str}-{nagents}p"

    if _config["env"] == "hallway_group":
        nagents = _config["env_args"]["n_agents"]
        state_numbers = _config["env_args"]["state_numbers"]
        group_ids = _config["env_args"]["group_ids"]
        state_str = "x".join([str(state_num) for state_num in state_numbers])
        group_str = "x".join([str(group_id) for group_id in group_ids])
        _config["env_args"]["key"] = f"hallway-{state_str}-{group_str}-{nagents}p"

    try:
        map_name = _config["env_args"]["map_name"]
    except:
        map_name = _config["env_args"]["key"]
    unique_token = f"{_config['name']}_seed{_config['seed']}_{map_name}_{datetime.datetime.now()}"

    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "tb_logs", args.env, map_name, f"{_config['name']}_{remark_str}",
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

        # write config file
        config_str = json.dumps(vars(args), indent=4)
        with open(os.path.join(tb_exp_direc, "config.json"), "w") as f:
            f.write(config_str)

    if args.test_encoder:
        result_save_direc = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "recons_logs", args.env, map_name, f"{_config['name']}_{remark_str}",
        )
        os.makedirs(result_save_direc, exist_ok=True)
        args.encoder_result_direc = result_save_direc

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    # os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        print(args.test_nepisode)
        assert 0
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def evaluate_encoder(args, runner, buffer, learner):
    
    for _ in range(args.test_nepisode):
        episode_batch = runner.run(test_mode=True)
        buffer.insert_episode_batch(episode_batch)
    
    # Get samples from buffer    
    episode_sample = buffer.sample(args.test_nepisode)
    
    max_ep_t = episode_sample.max_t_filled()
    episode_sample = episode_sample[:, :max_ep_t]
    
    if episode_sample.device != args.device:
        episode_sample.to(args.device)
    
    learner.test_encoder(episode_sample)

    runner.close_env()
        


def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    if "unit_dim" in env_info:
        args.unit_dim = env_info["unit_dim"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Some other modules
    # State encoder model
    if args.name == "msra":
        state_dim = int(np.prod(args.state_shape))
        state_encoder = state_enc_REGISTRY[args.state_encoder](input_shape=state_dim, latent_dim=args.state_repre_dim)

        # Give mac the state_encoder
        mac.setup_encoder(encoder=state_encoder)
    
    # Latent model
    latent_model = model_REGISTRY[args.latent_model](args)

    # Learner
    if args.name == "msra":
        learner = le_REGISTRY[args.learner](mac, state_encoder, latent_model, buffer.scheme, logger, args)
    else:
        # msra_alpha or msra_beta
        learner = le_REGISTRY[args.learner](mac, latent_model, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            runner.log_train_stats_t = runner.t_env
            if args.test_encoder:
                evaluate_encoder(args, runner, buffer, learner)
                logger.console_logger.info("Finished Encoder Evaluation")
                return 
            else: 
                evaluate_sequential(args, runner)
                logger.log_stat("episode", runner.t_env, runner.t_env)
                logger.print_recent_stats()
                logger.console_logger.info("Finished Evaluation")
                return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)
            # print(episode_sample['filled'].shape)
            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()

            last_test_T = runner.t_env
            # normal test run
            for _ in range(n_test_runs):
                runner.run(test_mode=True, teacher_forcing=False)

        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            remark_str = getattr(args, "remark", "NoRemark")
            try:
                map_name = args.env_args["map_name"]
            except:
                map_name = args.env_args["key"]
            save_path = os.path.join(
                args.local_results_path, "models", args.env, map_name, f"{args.name}_{remark_str}", args.unique_token, str(runner.t_env)
            )
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
