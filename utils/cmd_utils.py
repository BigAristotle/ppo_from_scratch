import argparse


def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    argparse.ArgumentParser()
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def common_arg_parser():
    """
    Create an argparse.ArgumentParser for main.py.
    """
    parser = arg_parser()
    parser.add_argument("--policy", default="PPO")  # Policy name (PPO or PPO_with_GRU)
    parser.add_argument("--env", default="BipedalWalker-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25000, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5000, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1000000, type=int)  # Max time steps to run environment
    parser.add_argument("--batch_size", default=64, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    # parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    # parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--hidden_space", default=[256, 256], type=int)
    args = parser.parse_args()

    return args
