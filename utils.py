import json
import datetime
import os
import torch
import torch
import random
import numpy as np
import os
import logging
import hashlib


def load_json(file: str):
    with open(file, 'r') as f:
        return json.loads(f.read())

def load_jsonl(file: str):
    load_data = []
    with open(file, 'r') as f:
        for line in f.readlines():
            load_line = json.loads(line)
            load_data.append(load_line)
        return load_data

def ensure_dir(dir_path):

    os.makedirs(dir_path, exist_ok=True)

def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"

def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur

def get_seqs_len(seqs):
    seq_len = torch.tensor([len(seq) for seq in seqs])
    return seq_len

def dict2str(result_dict):
    r"""convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    return "    ".join(
        [str(metric) + " : " + str(value) for metric, value in result_dict.items()]
    )

def safe_load(model, file, verbose=True):
    state_dict = torch.load(file, map_location=lambda storage, loc: storage)['state_dict']
    model_state_dict_keys = list(model.state_dict().keys())
    new_state_dict_keys = list(state_dict.keys())
    new_keys_in_new = [k for k in new_state_dict_keys if k not in model_state_dict_keys]
    no_match_keys_of_model = [k for k in model_state_dict_keys if k not in new_state_dict_keys]
    if verbose:
        print('##', model._get_name(), '# new keys in file:', new_keys_in_new, '# no match keys:', no_match_keys_of_model)
    model.load_state_dict(state_dict, strict=False)

def config_for_log(config: dict) -> dict:
    config = config.copy()
    config.pop('device', None)
    config.pop('accelerator', None)
    for k, v in config.items():
        if isinstance(v, list):
            config[k] = str(v)
    return config


def init_logger(config, filename=None):
    LOGROOT = config['log_dir']
    os.makedirs(LOGROOT, exist_ok=True)
    dataset_name = os.path.join(LOGROOT, config["dataset"])
    os.makedirs(dataset_name, exist_ok=True)

    logfilename = get_file_name(config, suffix='.log') if filename is None else filename
    logfilepath = os.path.join(LOGROOT, config["dataset"], logfilename)

    filefmt = "%(asctime)-15s %(levelname)s  %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    fh = logging.FileHandler(logfilepath)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fileformatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    logging.basicConfig(level=logging.INFO, handlers=[sh, fh])


def init_seed(seed, reproducibility):
    r"""init random seed for random functions in numpy, torch, cuda and cudnn
        This function is taken from https://github.com/RUCAIBox/RecBole/blob/2b6e209372a1a666fe7207e6c2a96c7c3d49b427/recbole/utils/utils.py#L188-L205

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """

    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def get_file_name(config: dict, suffix: str = ''):
    config_str = "".join([str(value) for key, value in config.items() if (key != 'accelerator' and key != 'device') ])
    md5 = hashlib.md5(config_str.encode(encoding="utf-8")).hexdigest()[:6]
    logfilename = "{}-{}{}".format(config['run_local_time'], md5, suffix)
    return logfilename


def convert_config_dict(config: dict) -> dict:
    """
    Convert the values in a dictionary to their appropriate types.

    Args:
        config (dict): The dictionary containing the configuration values.

    Returns:
        dict: The dictionary with the converted values.

    """
    for key in config:
        v = config[key]
        if not isinstance(v, str):
            continue
        try:
            new_v = eval(v)
            if new_v is not None and not isinstance(
                new_v, (str, int, float, bool, list, dict, tuple)
            ):
                new_v = v
        except (NameError, SyntaxError, TypeError):
            if isinstance(v, str) and v.lower() in ['true', 'false']:
                new_v = (v.lower() == 'true')
            else:
                new_v = v
        config[key] = new_v
    return config


def init_device():
    """
    Set the visible devices for training. Supports multiple GPUs.

    Returns:
        torch.device: The device to use for training.

    """
    import torch
    use_ddp = True if os.environ.get("WORLD_SIZE") else False  # Check if DDP is enabled
    if torch.cuda.is_available():
        return torch.device('cuda'), use_ddp
    else:
        return torch.device('cpu'), use_ddp


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M")
    return cur


def parse_command_line_args(unparsed):

    args = {}
    for text_arg in unparsed:
        if '=' not in text_arg:
            raise ValueError(f"Invalid command line argument: {text_arg}, please add '=' to separate key and value.")
        key, value = text_arg.split('=')
        key = key[len('--'):]
        try:
            value = eval(value)
        except:
            pass
        args[key] = value
    return args


def log(message, logger, accelerator=None, level='info'):
    if accelerator is None or accelerator.is_main_process:
        if level == 'info':
            logger.info(message)
        elif level == 'error':
            logger.error(message)
        elif level == 'warning':
            logger.warning(message)
        elif level == 'debug':
            logger.debug(message)
        else:
            raise ValueError(f'Invalid log level: {level}')


def combine_index(x, y):
    final = []
    assert len(x) == len(y)
    for i in range(len(x)):
        final.append(x[i] + y[i])
    return final