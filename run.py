import os
import shutil
import argparse
import numpy as np
import math
from core.data_provider import datasets_factory
from core.models.model_factory import Model
from core.utils import preprocess
import core.trainer as trainer
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import datetime

dt_now = datetime.datetime.now()
TIMESTAMP = dt_now.strftime('%Y%m%d%H%M')

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PredRNN')
# training/test
parser.add_argument('--is_training', type=bool, default=True)
parser.add_argument('--device', type=str)
# data
parser.add_argument('--dataset', type=str)
parser.add_argument('--config', type=str)
parser.add_argument('--data_train_path', type=str)
parser.add_argument('--data_test_path', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--gen_frm_dir', type=str)
parser.add_argument('--input_length', type=int)
parser.add_argument('--total_length', type=int)
parser.add_argument('--img_width', type=int)
parser.add_argument('--img_height', type=int)
parser.add_argument('--img_channel', type=int)
# model
parser.add_argument('--model_name', type=str)
parser.add_argument('--pretrained_model', type=str)
parser.add_argument('--num_hidden', type=str)
parser.add_argument('--filter_size', type=int)
parser.add_argument('--stride', type=int)
parser.add_argument('--patch_size', type=int)
parser.add_argument('--layer_norm', type=int)
parser.add_argument('--decouple_beta', type=float)
# reverse scheduled sampling
parser.add_argument('--reverse_scheduled_sampling', type=bool)
parser.add_argument('--r_sampling_step_1', type=float)
parser.add_argument('--r_sampling_step_2', type=int)
parser.add_argument('--r_exp_alpha', type=int)
# scheduled sampling
parser.add_argument('--scheduled_sampling', type=bool)
parser.add_argument('--sampling_stop_iter', type=int)
parser.add_argument('--sampling_start_value', type=float)
parser.add_argument('--sampling_changing_rate', type=float)
# optimization
parser.add_argument('--lr', type=float)
parser.add_argument('--reverse_input', type=bool)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--max_epoches', type=int)
parser.add_argument('--num_save_samples', type=int)
parser.add_argument('--num_valid_samples',      type=int)
parser.add_argument('--n_gpu', type=int)
# visualization of memory decoupling
parser.add_argument('--visual', type=int)
parser.add_argument('--visual_path', type=str)
# action-based predrnn
parser.add_argument('--injection_action', type=str)
parser.add_argument('--conv_on_input', type=int)
parser.add_argument('--res_on_conv', type=int)
parser.add_argument('--num_action_ch', type=int)
args = parser.parse_args()

if args.config == 'mnist':
    from configs.mnist_configs import configs
elif args.config == 'kitti':
    from configs.kitti_configs import configs
elif args.config == 'town':
    from configs.town_configs import configs
elif args.config == 'aia211':
    from configs.aia211_configs import configs
args = configs(args)

print('---------------------------------------------')
print('Dataset       :', args.dataset)
print('Configuration :', args.config)
print('---------------------------------------------')

def reserve_schedule_sampling_exp(itr):
    if itr < args.r_sampling_step_1:
        r_eta = 0.5
    elif itr < args.r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - args.r_sampling_step_1) / args.r_exp_alpha)
    else:
        r_eta = 1.0

    if itr < args.r_sampling_step_1:
        eta = 0.5
    elif itr < args.r_sampling_step_2:
        eta = 0.5 - (0.5 / (args.r_sampling_step_2 - args.r_sampling_step_1)) * (itr - args.r_sampling_step_1)
    else:
        eta = 0.0

    r_random_flip = np.random.random_sample(
        (args.batch_size, args.input_length - 1))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)

    ones = np.ones((args.img_height // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))

    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - 2):
            if j < args.input_length - 1:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
            else:
                if true_token[i, j - (args.input_length - 1)]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length - 2,
                                  args.img_height // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * args.img_channel))
    return real_input_flag


def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))

    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0

    random_flip = np.random.random_sample((args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_height // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length - args.input_length - 1,
                                  args.img_height // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * args.img_channel))
    return eta, real_input_flag


def plot_loss(indices, finish_time):
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3,3)
    ax = []
    ax.append(fig.add_subplot(gs[0, 0:2],fc='gray', xlim=(0,10),ylim=(0,10)))
    ax.append(fig.add_subplot(gs[1, 0]))
    ax.append(fig.add_subplot(gs[1, 1]))
    ax.append(fig.add_subplot(gs[2, 1]))
    ax.append(fig.add_subplot(gs[1, 2]))
    ax.append(fig.add_subplot(gs[2, 2]))

    ax[1].plot(indices[0].flatten(), color='r', lw=0.75, label='train loss')
    ax[3].plot(indices[1].flatten(), color='g', lw=0.75, label='valid mse')
    ax[4].plot(indices[2].flatten(), color='y', lw=0.75, label='valid psnr')
    ax[5].plot(indices[3].flatten(), color='m', lw=0.75, label='valid ssim')
    ax[6].plot(indices[4].flatten(), color='c', lw=0.75, label='valid lpips')

    for i in range(len(ax)):
        ax[i].grid()
        ax[i].legend()

    ax[0].xaxis.set_major_locator(mpl.ticker.NullLocator())
    ax[0].yaxis.set_major_locator(mpl.ticker.NullLocator())
    ax[0].text(1,9,"PredRNN " + str(TIMESTAMP))
    ax[0].text(1,8,"---------------------")
    ax[0].text(1,7,"Dataset " + str(args.dataset))
    ax[0].text(1,6,"Batch size: " + str(args.batch_size))
    ax[0].text(1,5,"Epoch: " + str(args.max_epoches))
    time = '- ' if finish_time == 0 else str(finish_time)
    ax[0].text(1,4,"Time: " + time + 'h')

    fig.patch.set_alpha(0)
    fig.tight_layout()
    fig_path = os.path.join(args.gen_frm_dir, TIMESTAMP, 'losses.png')
    fig.savefig(fig_path, format="png", dpi=200)


def train_wrapper(model):
    if args.pretrained_model:
        model.load(args.pretrained_model)

    train_input_handle = datasets_factory.data_provider(configs=args,
                                                        data_train_path=args.data_train_path,
                                                        dataset=args.dataset,
                                                        data_test_path=args.data_test_path,
                                                        batch_size=args.batch_size,
                                                        is_training=True,
                                                        is_shuffle=True)
    test_input_handle = datasets_factory.data_provider(configs=args,
                                                      data_train_path=args.data_train_path,
                                                      dataset=args.dataset,
                                                      data_test_path=args.data_test_path,
                                                      batch_size=args.batch_size,
                                                      is_training=False,
                                                      is_shuffle=False)

    eta = args.sampling_start_value
    itr = 0
    indices = np.zeros((5,args.max_epoches))
    time_train_start = time.time() 

    for epoch in range(1, args.max_epoches + 1):
        print("------------- epoch: " + str(epoch) + " / " + str(args.max_epoches) + " ----------------")
        print("Train with " + str(len(train_input_handle)) + " data")
        time_epoch_start = time.time() 

        for ims in train_input_handle:
            time_itr_start = time.time() 
            batch_size = ims.shape[0]

            if args.reverse_scheduled_sampling:
                real_input_flag = reserve_schedule_sampling_exp(itr)
            else:
                eta, real_input_flag = schedule_sampling(eta, itr)
            loss = trainer.train(model, ims, real_input_flag, args, itr)

            time_itr = round(time.time() - time_itr_start, 3)
            print('\ritr:' + str(itr) + ' ' + str(time_itr).ljust(5,'0') + 's | Loss: ' + str(loss), end='')
            itr += 1

        test_indices = trainer.test(model, test_input_handle, args, itr, TIMESTAMP, True)
        indices[:, epoch-1] = tuple([loss]) + test_indices
        plot_loss(indices,0)
        model.save(TIMESTAMP,itr)
        time_epoch = round((time.time() - time_epoch_start) / 60, 3)
        pred_finish_time = time_epoch * (args.max_epoches - epoch) / 60
        print(str(time_epoch) + 'm/epoch | ETA: ' + str(round(pred_finish_time,2)) + 'h')

    train_finish_time = round((time.time() - time_train_start) / 3600,2)
    trainer.test(model, test_input_handle, args, itr, TIMESTAMP, False)
    plot_loss(indices,train_finish_time)


def test_wrapper(model):
    model.load(args.pretrained_model)
    test_input_handle = datasets_factory.data_provider(configs=args,
                                                       data_train_path=args.data_train_path,
                                                       dataset=args.dataset,
                                                       data_test_path=args.data_test_path,
                                                       batch_size=args.batch_size,
                                                       is_training=False,
                                                       is_shuffle=False)
    trainer.test(model, test_input_handle, args, itr, TIMESTAMP, False)


if __name__ == '__main__':
    print(args.model_name)
    print('Initializing models')
    model = Model(args)

    gen_path = os.path.join(args.gen_frm_dir, TIMESTAMP)
    if not os.path.exists(gen_path): os.mkdir(gen_path)

    if args.is_training:
        save_path = os.path.join(args.save_dir, TIMESTAMP)
        if not os.path.exists(save_path): os.mkdir(save_path)
        print('save results : ' + str(TIMESTAMP))
        train_wrapper(model)
    else:
        test_wrapper(model)
