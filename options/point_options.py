import argparse


class Point_Options():
    """This classification defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset classification and model classification.
    """

    def __init__(self):
        """Reset the classification; indicates the classification hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--data_root', type=str, default='/home/isalab206/Downloads/dataset', help='your path to dataroot')
        parser.add_argument('--dataset', type=str, default='potsdam', help='[potsdam|vaihingen]')
        parser.add_argument('--experiment_name', type=str, default='Point', help='name of the experiment. It decides where to load datafiles, store samples and models')
        parser.add_argument('--save_path', type=str, default='/home/isalab206/LJX/HSLabeling-master/al_seg_train_20000_HSLabeling_loveda/save', help='models are saved here al_gpv1_50000')
        parser.add_argument('--data_label', type=str, default='seg_train_20000_HSLabeling_loveda', help='labels, weights are saved here gpv1_50000')
        parser.add_argument('--data_inform_path', type=str, default='/home/isalab206/LJX/HSLabeling-master/datafiles', help='path to files about the datafiles information')

        # model parameter
        parser.add_argument('--base_model', type=str, default='Deeplab', help='[Deeplab|FCN|UNet]')
        parser.add_argument('--backbone', type=str, default='resnet18', help='which resnet')
        parser.add_argument('--out_stride', type=int, default=32, help='out_stride')
        parser.add_argument('--num_classes', type=int, default=5, help='classes')

        # train parameters
        parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
        parser.add_argument('--pin', type=bool, default=True, help='pin_memory or not')

        parser.add_argument('--num_workers', type=int, default=10, help='number of workers')
        parser.add_argument('--img_size', type=int, default=256, help='image size')
        parser.add_argument('--in_channels', type=int, default=3, help='input channels')

        parser.add_argument('--num_epochs', type=int, default=100, help='num of epochs')
        parser.add_argument('--base_lr', type=float, default=1e-3, help='base learning rate')
        parser.add_argument('--decay', type=float, default=5e-4, help='decay')
        parser.add_argument('--log_interval', type=int, default=60, help='how long to log, set yo 100 batch')
        parser.add_argument('--resume', type=bool, default=False, help='resume the saved checkpoint or not')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        self.opt = opt

        return self.opt


if __name__ == '__main__':
    opt = Point_Options().parse()
    print(opt)