from BigGAN_512 import BigGAN_512
from BigGAN_256 import BigGAN_256
from BigGAN_128 import BigGAN_128
import argparse
from utils import *
# NPU库
from npu_bridge.npu_init import *
# Moxing库
import moxing as mox

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of BigGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--dataset', type=str, default='CAT_00', help='[mnist / cifar10 / custom_dataset]')

    # epoch
    # 作业1用的是5
    # parser.add_argument('--epoch', type=int, default=5, help='The number of epochs to run')
    parser.add_argument('--epoch', type=int, default=5, help='The number of epochs to run')

    # iteration ModelArts用 5
    parser.add_argument('--iteration', type=int, default=5, help='The number of training iterations')

    # batch_size ModelArts用 32
    parser.add_argument('--batch_size', type=int, default=32, help='The size of batch per gpu')
    parser.add_argument('--ch', type=int, default=96, help='base channel number per layer')

    # SAGAN
    # batch_size = 256
    # base channel = 64
    # epoch = 100 (1M iterations)

    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freqy')
    parser.add_argument('--save_freq', type=int, default=1000, help='The number of ckpt_save_freq')

    parser.add_argument('--g_lr', type=float, default=0.00005, help='learning rate for generator')
    parser.add_argument('--d_lr', type=float, default=0.0002, help='learning rate for discriminator')

    # if lower batch size
    # g_lr = 0.0001
    # d_lr = 0.0004

    # if larger batch size
    # g_lr = 0.00005
    # d_lr = 0.0002

    parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for Adam optimizer')
    parser.add_argument('--moving_decay', type=float, default=0.9999, help='moving average decay for generator')

    parser.add_argument('--z_dim', type=int, default=128, help='Dimension of noise vector')
    parser.add_argument('--sn', type=str2bool, default=True, help='using spectral norm')

    parser.add_argument('--gan_type', type=str, default='hinge', help='[gan / lsgan / wgan-gp / wgan-lp / dragan / hinge]')
    parser.add_argument('--ld', type=float, default=10.0, help='The gradient penalty lambda')

    parser.add_argument('--n_critic', type=int, default=2, help='The number of critic')

    # img_size ModelArts用 128
    parser.add_argument('--img_size', type=int, default=128, help='The size of image')
    parser.add_argument('--sample_num', type=int, default=64, help='The number of sample images')

    parser.add_argument('--test_num', type=int, default=10, help='The number of images generated by the test')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    # 补充两个PyCharm提交训练作业需要的字段
    parser.add_argument('--data_url', type=str, default='',
                        help='Directory OBS')
    parser.add_argument('--train_url', type=str, default='',
                        help='Directory output')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # 将数据集从OBS拷贝到ModelArts
    mox.file.copy_parallel(src_url="obs://cann-biggan/dataset/CAT_00/", dst_url="dataset/CAT_00")


    # 创建Profiling所需的目录
    os.mkdir("/cache/profiling")

    # # 获取当前目录
    # cwd = os.getcwd()
    # print("1cwd is", cwd)
    # listdir = os.listdir(cwd)
    # print("1listdir is",  listdir)
    #
    # # 切换到dataset目录下看看
    # os.chdir("./dataset")
    # cwd = os.getcwd()
    # print("2cwd is", cwd)
    # listdir = os.listdir(cwd)
    # print("2listdir is",  listdir)
    #
    # # 切换到CAT_01目录下看看
    # os.chdir("./CAT_00")
    # cwd = os.getcwd()
    # print("3cwd is", cwd)
    # listdir = os.listdir(cwd)
    # print("3listdir is",  listdir)

    # os.chdir("/cache/user-job-dir/workspace/device0")

    # open session
    # config = tf.ConfigProto(allow_soft_placement=True)
    config = tf.ConfigProto(allow_soft_placement=True)
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"

    custom_op.parameter_map["use_off_line"].b = True
    # 打开Profiling数据采集
    custom_op.parameter_map["profiling_mode"].b = True
    custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(
        '{"output":"/cache/profiling","task_trace":"on","aicpu":"on"}')

    # https://gitee.com/ascend/modelzoo/wikis/Modelarts%E9%87%87%E9%9B%86Profiling%E6%80%A7%E8%83%BD%E6%95%B0%E6%8D%AE?sort_id=3646848
    # 参考以上链接

    # 调优尝试
    # 混合精度开关
    # custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")

    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭remap

    # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    with tf.Session(config=config) as sess:
        # default gan = BigGAN_128

        if args.img_size == 512 :
            gan = BigGAN_512(sess, args)
        elif args.img_size == 256 :
            gan = BigGAN_256(sess, args)
        else :
            gan = BigGAN_128(sess, args)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        if args.phase == 'train' :
            # launch the graph in a session
            gan.train()

            # visualize learned generator
            gan.visualize_results(args.epoch - 1)

            print(" [*] Training finished!")

        if args.phase == 'test' :
            gan.test()
            print(" [*] Test finished!")

    # 将profiling数据拷贝回OBS
    mox.file.copy_parallel(src_url="/cache/profiling", dst_url="obs://cann-biggan/profiling/")

if __name__ == '__main__':
    main()