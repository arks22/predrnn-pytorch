import os.path
import datetime
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from core.utils import preprocess, metrics
import lpips
import torch
import time
import math
import datetime


def train(model, ims, real_input_flag, configs, itr):
    loss = model.train(ims, real_input_flag)

    if configs.reverse_input:
        ims = ims.to('cpu').detach().numpy().copy()
        ims_rev = np.flip(ims, axis=1).copy()
        loss += model.train(ims_rev, real_input_flag)
        loss = loss / 2
    return loss


def test(model, test_input_handle, configs, itr, timestamp,is_valid):
    if is_valid:
        print('\nValid with ' + str(configs.num_valid_samples) + ' data')
    else:
        print('\nTest with ' + str(len(test_input_handle)) + ' data')

    loss_fn_alex = lpips.LPIPS(net='alex')
    res_path = os.path.join(configs.gen_frm_dir, timestamp, str(itr))
    if not os.path.exists(res_path): os.mkdir(res_path)

    avg_mse = 0
    batch_id = 0
    img_mse, ssim, psnr = [], [], []
    lp = []

    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)
        lp.append(0)

    # reverse schedule sampling
    if configs.reverse_scheduled_sampling == 1:
        mask_input = 1
    else:
        mask_input = configs.input_length

    real_input_flag = np.zeros(
        (configs.batch_size,
         configs.total_length - mask_input - 1,
         configs.img_width // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))

    if configs.reverse_scheduled_sampling == 1:
        real_input_flag[:, :configs.input_length - 1, :, :] = 1.0

    for data in test_input_handle:
        if is_valid and configs.num_valid_samples < batch_id: break;
        print('\ritr:' + str(batch_id),end='')

        batch_id = batch_id + 1
        batch_size = data.shape[0]
        real_input_flag = np.zeros(
            (configs.batch_size,
            configs.total_length - mask_input - 1,
            configs.img_width // configs.patch_size,
            configs.img_width // configs.patch_size,
            configs.patch_size ** 2 * configs.img_channel))
        img_gen = model.test(data, real_input_flag)
        #############################
        img_gen = img_gen.transpose(0, 1, 3, 4, 2)  # * 0.5 + 0.5
        test_ims = data.detach().cpu().numpy().transpose(0, 1, 3, 4, 2)  # * 0.5 + 0.5
        output_length = configs.total_length - configs.input_length
        output_length = min(output_length, configs.total_length - 1)
        test_ims = preprocess.reshape_patch_back(test_ims, configs.patch_size)
        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        img_out = img_gen[:, -output_length:, :]

        # MSE per frame
        for i in range(output_length):
            x = test_ims[:, i + configs.input_length, :, :, :]
            gx = img_out[:, i, :, :, :]
            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse
            # cal lpips
            img_x = np.zeros([configs.batch_size, 3, configs.img_width, configs.img_width])
            if configs.img_channel == 3:
                img_x[:, 0, :, :] = x[:, :, :, 0]
                img_x[:, 1, :, :] = x[:, :, :, 1]
                img_x[:, 2, :, :] = x[:, :, :, 2]
            else:
                img_x[:, 0, :, :] = x[:, :, :, 0]
                img_x[:, 1, :, :] = x[:, :, :, 0]
                img_x[:, 2, :, :] = x[:, :, :, 0]
            img_x = torch.FloatTensor(img_x)
            img_gx = np.zeros([configs.batch_size, 3, configs.img_width, configs.img_width])
            if configs.img_channel == 3:
                img_gx[:, 0, :, :] = gx[:, :, :, 0]
                img_gx[:, 1, :, :] = gx[:, :, :, 1]
                img_gx[:, 2, :, :] = gx[:, :, :, 2]
            else:
                img_gx[:, 0, :, :] = gx[:, :, :, 0]
                img_gx[:, 1, :, :] = gx[:, :, :, 0]
                img_gx[:, 2, :, :] = gx[:, :, :, 0]
            img_gx = torch.FloatTensor(img_gx)
            lp_loss = loss_fn_alex(img_x, img_gx)
            lp[i] += torch.mean(lp_loss).item()

            real_frm = np.uint8(x * 255)
            pred_frm = np.uint8(gx * 255)

            #psnr[i] += metrics.batch_psnr(pred_frm, real_frm)
            p = 0
            for sample_id in range(batch_size):
                mse_tmp = np.square(x[sample_id, :] - gx[sample_id, :]).mean()
                p += 10 * np.log10(1 / (mse_tmp + 1e-5))
            p /= (batch_size)
            psnr[i] += p

            for b in range(configs.batch_size):
                score, _ = compare_ssim(pred_frm[b], real_frm[b], full=True, channel_axis=-1)
                ssim[i] += score

        #Save as many prediction samples as 'num_save_samples'
        if batch_id <= configs.num_save_samples:
            res_width = configs.img_width
            res_height = configs.img_height
            img = np.ones((2 * res_height, configs.total_length * res_width, configs.img_channel))
            img_name = os.path.join(res_path, str(batch_id) + '.png')

            vid_arr = np.ones((res_height, res_width*2, configs.img_channel, configs.total_length))
            vid_name = os.path.join(res_path, str(batch_id) + '.mp4')
            fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
            video  = cv2.VideoWriter(vid_name, fourcc, 1.00, (res_width*2+1, res_height+1))
            #なぜかFPS2.00が選べない。なぜ？
            for i in range(configs.total_length):
                img[:res_height, i * res_width:(i + 1) * res_width, :] = test_ims[0, i, :,:,:]
                vid_arr[:res_height, :res_width, :, i] = test_ims[0, i, :,:,:]

            for i in range(output_length):
                img[res_height:, (configs.input_length + i) * res_width:(configs.input_length + i + 1) * res_width, :] = img_out[0, -output_length + i, :]
                vid_arr[:res_height, res_width:, : ,configs.input_length + i] = img_out[0, -output_length + i, :, :, :]

            for i in range(configs.total_length):
                frame = vid_arr[:,:,:,i]
                frame = np.maximum(frame, 0)
                frame = np.minimum(frame, 1)
                frame = np.repeat(frame, 3).reshape(res_height,res_width*2,3)
                video.write((frame * 255).astype(np.uint8))
            video.release()

            img = np.maximum(img, 0)
            img = np.minimum(img, 1)
            cv2.imwrite(img_name, (img * 255).astype(np.uint8))

    print('')
    avg_mse = avg_mse / (batch_id * configs.batch_size)
    avg_mse_per_frame = avg_mse / (configs.total_length - configs.input_length)
    print('----------------------------------------------------------------------------------------------------')
    print('|    1    |    2    |    3    |    4    |    5    |    6    |    7    |    8    |    9    |    10   |')
    print('| -- *MSE  per frame: ' + str(avg_mse_per_frame) + ' ---------------------------------------------')
    for i in range(configs.total_length - configs.input_length):
        stage_mse = img_mse[i] / (batch_id * configs.batch_size)
        digits = math.floor(math.log10(stage_mse))
        print('|  ' + str(round(stage_mse, 4 - digits)).ljust(6,'0') + ' ', end='')
    print('|')

    avg_psnr = np.mean(psnr)
    psnr = np.asarray(psnr, dtype=np.float32) / batch_id
    print('| -- *PSNR  per frame: ' + str(avg_psnr) + ' ----------------------------------------------------------')
    for i in range(configs.total_length - configs.input_length):
        digits = math.floor(math.log10(psnr[i]))
        print('|  ' + str(round(psnr[i], 4 - digits)).ljust(6,'0') + ' ', end='')
    print('|')

    ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * batch_id)
    avg_ssim = np.mean(ssim)
    print('| -- *SSIM per frame: ' + str(avg_ssim) + ' -------------------------------------------------------------')
    for i in range(configs.total_length - configs.input_length):
       print('| ' + str(round(ssim[i], 5)).ljust(7,'0') + ' ', end='')
    print('|')

    lp = np.asarray(lp, dtype=np.float32) / batch_id
    avg_lp = np.mean(lp)
    print('| -- *LPIPS per frame: ' + str(avg_lp) + ' ---------------------------------------------------------')
    for i in range(configs.total_length - configs.input_length):
        stage_lp = lp[i] / batch_id
        digits = math.floor(math.log10(stage_lp))
        print('| ' + str(round(stage_lp, 3 - digits)).ljust(6,'0') + ' ', end='')
    print('|')
    print('----------------------------------------------------------------------------------------------------')

    return avg_mse_per_frame, avg_psnr, avg_ssim, avg_lp
