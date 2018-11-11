import os
import configargparse

def parse_bool(v):
    """ Parse-able bool type.

    Bool type to set in add in parser.add_argument to fix not parsing of False. See:
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    Args:
        v (str): a string that indicates true ('yes', 'true', 't', 'y', '1') or false ('no', 'false', 'f', 'n', '0').
    """
    import configargparse
    if v.lower().strip() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower().strip() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


class ArchLayer:
    def __init__(self, layer_str):
        l = [int(e) for e in layer_str.split(",")]
        assert len(l)==2, "Each layer should have two ints seperatated by a komma."
        self.caps = l[0]
        self.len = l[1]


class Architecture:
    """ Used Architecture

    Class to specify the architecture in the config and parse the config string to a object.


    Args:
        arch_str (str): String to specify the architecture. The layers are separated by a ';'. Each layer consists of
        two numbers seperated by ','. The first specifies the number of capsules, the second the length. In the first
        layer the number of capsules is multiplied by the size of the grid.
    """

    def __init__(self, arch_str):

        arch = arch_str.split(";")
        assert 2 <= len(arch), "Architecture should have at least a primary and final layer."

        self.prim = ArchLayer(arch[0])
        self.final = ArchLayer(arch[-1])

        self.all_but_prim = []
        for i in arch[1:]:
            self.all_but_prim.append(ArchLayer(i))


def get_conf(path_root="."):
    """ Get configuration

    Args:
        path_root: path root to main project

    Returns:
        conf: (Configuration) Object with all configurations
        parser: (Parser) Object with the used parser
    """

    p = configargparse.get_argument_parser()

    p.add('--conf', is_config_file=True, default=f"{path_root}/configurations/default.conf",
          help='configurations file path')

    # required arguments: specified in configurations file or in
    p.add_argument('--trained_model_path', required=True, type=str, help='Path of checkpoints.')
    p.add_argument('--batch_size', type=int, required=True, help='Batch size.')
    p.add_argument('--epochs', type=int, required=True, help='Number of epochs')
    p.add_argument('--seed', type=int, required=True, help="Torch and numpy random seed. To ensure repeatability.")
    p.add_argument('--save_trained', type=parse_bool, required=True, help='Save fully trained model for inference.')
    p.add_argument('--debug', type=parse_bool, required=True, help="debug mode: break early")
    p.add_argument('--print_time', type=parse_bool, required=True, help="print train time per sample")
    p.add_argument('--load_name', type=str, required=True, help="Name of the model to load")
    p.add_argument('--load_model', type=parse_bool, required=True, help="Load model yes/no")
    p.add_argument("--log_file_name", type=str, required=True, help="log file to log output to")
    p.add_argument("--log_file", type=parse_bool, required=True, help="log file to log output to")
    p.add_argument("--drop_last", type=parse_bool, required=True, help="drop last incomplete batch")
    p.add_argument('--shuffle', type=parse_bool, required=True, help='Shuffle dataset')
    p.add_argument('--n_saved', type=int, required=True, help='Models are save every epoch. N_saved is length of this '
                                                              'history')
    p.add_argument('--learning_rate', type=float, required=True, help='Learning rate of optimizer')
    p.add_argument('--early_stop', type=parse_bool, required=True, help='Early stopping on validation loss')
    p.add_argument('--cudnn_benchmark', type=parse_bool, required=True,
                   help='Bool for cudnn benchmarking. Faster for large')
    p.add_argument('--valid_size', type=float, required=True, help='Size of the validation set (between 0.0 and 1.0)')
    p.add_argument('--score_file_name', type=str, required=True,
                   help='File name of the best scores over all epochs. Save must be True. ')
    p.add_argument('--save_best', type=str, required=True, help='Save best score yes/no.')
    p.add_argument('--exp_name', type=str, required=True, help="Name of the experiment.")
    p.add_argument('--model_name', type=str, required=True, help='Name of the model.')
    p.add_argument('--alpha', type=float, required=True, help="Alpha of CapsuleLoss")
    p.add_argument('--m_plus', type=float, required=True, help="m_plus of margin loss")
    p.add_argument('--m_min', type=float, required=True, help="m_min of margin loss")
    p.add_argument('--routing_iters', type=int, required=True,
                        help="Number of iterations in the routing algo.")
    p.add_argument('--dataset', type=str, required=True, help="Either mnist or cifar10")
    p.add_argument('--stdev_W', type=float, required=True, help="stddev of W of capsule layer")
    p.add_argument('--bias_routing', type=parse_bool, required=True, help="whether to use bias in routing")
    p.add_argument('--architecture', type=Architecture, required=True,
                        help="Architecture of the capsule network. Notation: Example: 32,8;10,16")
    p.add_argument('--use_recon', type=parse_bool, required=True,
                        help="Use reconstruction in the total loss yes/no")
    p.add_argument('--use_visdom', type=parse_bool, required=True, help="Use visdom yes/no.")
    p.add_argument('--start_visdom', type=parse_bool, required=True, help="Start visdom.")

    try:
        conf = p.parse_args()
    except:
        print(p.format_help())
        raise ValueError("Could not parse config.")

    # combined configs
    conf.model_checkpoint_path = "{}/{}{}".format(conf.trained_model_path, conf.model_name,
                                                  "_debug" if conf.debug else "")
    conf.model_load_path = f"./experiments/{conf.exp_name}/{conf.trained_model_path}/{conf.load_name}"
    conf.exp_path = f"./experiments/{conf.exp_name}"

    if not os.path.exists(conf.exp_path):
        os.makedirs(conf.exp_path)

    with open(f"{conf.exp_path}/used_conf.txt", "w") as f:
        f.write(p.format_values())

    return conf, p
