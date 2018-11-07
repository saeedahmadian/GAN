import tensorflow as tf
from config import args


from model import CyberGAN
from data import CyberData


def main(_):

    data = CyberData(args)
    model = CyberGAN(args)

    if args.mode == 'train':
        model.train(data)


if __name__ == '__main__':
    tf.app.run()



a=1