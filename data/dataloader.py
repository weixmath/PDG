import torch
import torch.utils.data as data
import os
import numpy as np
from torchvision import transforms
import glob
from data.data_utils import *


class PaddingData(data.Dataset):
    def __init__(self, pc_root, aug=False, status='train', pc_input_num=2048, density=0, drop=0, p_scan=0,
                 swapax=False):
        super(PaddingData, self).__init__()

        self.status = status

        self.density = density
        self.drop = drop
        self.p_scan = p_scan

        self.aug = aug
        self.pc_list = []
        self.lbl_list = []
        self.transforms = transforms.Compose(
            [
                PointcloudToTensor(),
                PointcloudScale(),
                PointcloudRotate(),
                PointcloudRotatePerturbation(),
                PointcloudTranslate(),
                PointcloudJitter(),
            ]
        )
        self.pc_input_num = pc_input_num

        categorys = glob.glob(os.path.join(pc_root, '*'))
        categorys = [c.split(os.path.sep)[-1] for c in categorys]
        categorys = sorted(categorys)
        print(categorys)
        if self.density > 0:
            rand_points = np.random.uniform(-1, 1, 40000)
            x1 = rand_points[:20000]
            x2 = rand_points[20000:]
            power_sum = x1 ** 2 + x2 ** 2
            p_filter = power_sum < 1
            power_sum = power_sum[p_filter]
            sqrt_sum = np.sqrt(1 - power_sum)
            x1 = x1[p_filter]
            x2 = x2[p_filter]
            x = (2 * x1 * sqrt_sum).reshape(-1, 1)
            y = (2 * x2 * sqrt_sum).reshape(-1, 1)
            z = (1 - 2 * power_sum).reshape(-1, 1)
            self.density_points = np.hstack([x, y, z])
        if status == 'train':
            npy_list = glob.glob(os.path.join(pc_root, '*', 'train', '*.npy'))
        else:
            npy_list = glob.glob(os.path.join(pc_root, '*', 'test', '*.npy'))

        for idx, _dir in enumerate(npy_list):
            print("\r%d/%d" % (idx, len(npy_list)), end="")
            pc = np.load(_dir).astype(np.float32)
            if swapax:
                pc[:, 1] = pc[:, 2] + pc[:, 1]
                pc[:, 2] = pc[:, 1] - pc[:, 2]
                pc[:, 1] = pc[:, 1] - pc[:, 2]
            self.pc_list.append(pc)
            self.lbl_list.append(categorys.index(_dir.split('/')[-3]))
        print()

        print(f'{status} data num: {len(self.pc_list)}')
        self.pc_list = np.array(self.pc_list)
        self.lbl_list = np.array(self.lbl_list)
        self.num_examples = len(self.pc_list)
        self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int)
        np.random.shuffle(self.train_ind)
        self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int)
        np.random.shuffle(self.val_ind)
    def __getitem__(self, idx):
        lbl = self.lbl_list[idx]
        pc = self.pc_list[idx]
        pc = normal_pc(pc)
        pn = min(pc.shape[0], self.pc_input_num)
        if self.aug and idx not in self.val_ind:
            pc = self.transforms(pc)
            pc = pc.numpy()
        if pn < self.pc_input_num:
            pc = np.append(pc, np.zeros((self.pc_input_num - pc.shape[0], 3)), axis=0)
        pc = pc[:self.pc_input_num]
        pc = np.expand_dims(pc.transpose(), axis=2)
        return idx,torch.from_numpy(pc).type(torch.FloatTensor), pn, lbl

    def __len__(self):
        return len(self.pc_list)

class myBatchPaddingData_DG(PaddingData):
    def __init__(self, pc_root, aug=False, status='train', pc_input_num=2048, swapax=False, batch_size=32,
                 sample_num=5,):
        super(myBatchPaddingData_DG, self).__init__(pc_root, aug, status, pc_input_num, swapax=swapax)
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.batch_count = int(len(self.pc_list) / batch_size * 3)
        self.pc_list = np.array(self.pc_list)
        self.lbl_list = np.array(self.lbl_list)

        rand_points = np.random.uniform(-1, 1, 40000)
        x1 = rand_points[:20000]
        x2 = rand_points[20000:]
        power_sum = x1 ** 2 + x2 ** 2
        p_filter = power_sum < 1
        power_sum = power_sum[p_filter]
        sqrt_sum = np.sqrt(1 - power_sum)
        x1 = x1[p_filter]
        x2 = x2[p_filter]
        x = (2 * x1 * sqrt_sum).reshape(-1, 1)
        y = (2 * x2 * sqrt_sum).reshape(-1, 1)
        z = (1 - 2 * power_sum).reshape(-1, 1)
        self.density_points = np.hstack([x, y, z])
        self.fn = [
            lambda pc: drop_hole(pc, p=0.24),
            lambda pc: drop_hole(pc, p=0.36),
            lambda pc: drop_hole(pc, p=0.45),
            lambda pc: p_scan_DG(pc, pixel_size=0.017),
            lambda pc: p_scan_DG(pc, pixel_size=0.022),
            lambda pc: p_scan_DG(pc, pixel_size=0.035),
            lambda pc: density(pc, self.density_points[np.random.choice(self.density_points.shape[0])], 1.3),
            lambda pc: density(pc, self.density_points[np.random.choice(self.density_points.shape[0])], 1.4),
            lambda pc: density(pc, self.density_points[np.random.choice(self.density_points.shape[0])], 1.6),
            lambda pc: pc.copy(),
        ]
        self.task_num = len(self.fn)
        self.task_p = [1 / self.task_num] * self.task_num
        self.task_index = list(range(self.task_num))

    def __getitem__(self, idx):
        lbl = self.lbl_list[idx]
        pc = self.pc_list[idx]
        pc = normal_pc(pc)
        pc_ori = np.copy(pc)
        self.selected_tasks_ori = np.random.choice(self.task_index, 1)
        self.selected_tasks = np.random.choice(self.task_index,1)
        pc_ori = self.fn[self.selected_tasks_ori[0]](pc_ori)
        tmp_pc = self.fn[self.selected_tasks[0]](pc)
        if self.aug:
            tmp_pc = self.transforms(tmp_pc)
            tmp_pc = tmp_pc.numpy()
            pc_ori = self.transforms(pc_ori)
            pc_ori = pc_ori.numpy()
        pn = min(tmp_pc.shape[0], self.pc_input_num)
        if pn < self.pc_input_num:
            tmp_pc = np.append(tmp_pc, np.zeros((self.pc_input_num - tmp_pc.shape[0], 3)), axis=0)
        tmp_pc = tmp_pc[:self.pc_input_num]
        pn_ori = min(pc_ori.shape[0], self.pc_input_num)
        if pn_ori < self.pc_input_num:
            pc_ori = np.append(pc_ori, np.zeros((self.pc_input_num - pc_ori.shape[0], 3)), axis=0)
        pc_ori = pc_ori[:self.pc_input_num]

        return idx,torch.from_numpy(tmp_pc).type(torch.FloatTensor), pn, lbl, torch.from_numpy(pc_ori).type(
            torch.FloatTensor), pn_ori

    def __len__(self):
        return len(self.pc_list)

class myBatchPaddingData_DA(PaddingData):
    def __init__(self, pc_root, aug=False, status='train', pc_input_num=2048, swapax=False, batch_size=32,
                 sample_num=5,):
        super(myBatchPaddingData_DA, self).__init__(pc_root, aug, status, pc_input_num, swapax=swapax)
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.batch_count = int(len(self.pc_list) / batch_size * 3)
        self.pc_list = np.array(self.pc_list)
        self.lbl_list = np.array(self.lbl_list)

        rand_points = np.random.uniform(-1, 1, 40000)
        x1 = rand_points[:20000]
        x2 = rand_points[20000:]
        power_sum = x1 ** 2 + x2 ** 2
        p_filter = power_sum < 1
        power_sum = power_sum[p_filter]
        sqrt_sum = np.sqrt(1 - power_sum)
        x1 = x1[p_filter]
        x2 = x2[p_filter]
        x = (2 * x1 * sqrt_sum).reshape(-1, 1)
        y = (2 * x2 * sqrt_sum).reshape(-1, 1)
        z = (1 - 2 * power_sum).reshape(-1, 1)
        self.density_points = np.hstack([x, y, z])
        self.fn = [
            lambda pc: drop_hole(pc, p=0.24),
            lambda pc: drop_hole(pc, p=0.36),
            lambda pc: drop_hole(pc, p=0.45),
            lambda pc: p_scan_DA(pc, pixel_size=0.017),
            lambda pc: p_scan_DA(pc, pixel_size=0.022),
            lambda pc: p_scan_DA(pc, pixel_size=0.035),
            lambda pc: density(pc, self.density_points[np.random.choice(self.density_points.shape[0])], 1.3),
            lambda pc: density(pc, self.density_points[np.random.choice(self.density_points.shape[0])], 1.4),
            lambda pc: density(pc, self.density_points[np.random.choice(self.density_points.shape[0])], 1.6),
            lambda pc: pc.copy(),
        ]
        self.task_num = len(self.fn)
        self.task_p = [1 / self.task_num] * self.task_num
        self.task_index = list(range(self.task_num))

    def __getitem__(self, idx):
        lbl = self.lbl_list[idx]
        pc = self.pc_list[idx]
        pc = normal_pc(pc)
        pc_ori = np.copy(pc)
        self.selected_tasks_ori = np.random.choice(self.task_index, 1)
        self.selected_tasks = np.random.choice(self.task_index,1)
        pc_ori = self.fn[self.selected_tasks_ori[0]](pc_ori)
        tmp_pc = self.fn[self.selected_tasks[0]](pc)
        if self.aug:
            tmp_pc = self.transforms(tmp_pc)
            tmp_pc = tmp_pc.numpy()
            pc_ori = self.transforms(pc_ori)
            pc_ori = pc_ori.numpy()
        pn = min(tmp_pc.shape[0], self.pc_input_num)
        if pn < self.pc_input_num:
            tmp_pc = np.append(tmp_pc, np.zeros((self.pc_input_num - tmp_pc.shape[0], 3)), axis=0)
        tmp_pc = tmp_pc[:self.pc_input_num]
        pn_ori = min(pc_ori.shape[0], self.pc_input_num)
        if pn_ori < self.pc_input_num:
            pc_ori = np.append(pc_ori, np.zeros((self.pc_input_num - pc_ori.shape[0], 3)), axis=0)
        pc_ori = pc_ori[:self.pc_input_num]

        return idx,torch.from_numpy(tmp_pc).type(torch.FloatTensor), pn, lbl, torch.from_numpy(pc_ori).type(
            torch.FloatTensor), pn_ori

    def __len__(self):
        return len(self.pc_list)