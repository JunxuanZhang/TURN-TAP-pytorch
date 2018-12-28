import torch.utils.data as data
import numpy as np
import random
import os


class turnTestDataset(data.Dataset):
    def __init__(self, ctx_num, feat_dir, test_clip_path, batch_size,
                 unit_size, unit_feature_dim, data_preparation=None):
        self.ctx_num = ctx_num
        self.feat_dir = feat_dir
        self.test_clip_path = test_clip_path
        self.batch_size = batch_size
        self.unit_size = unit_size
        self.unit_feature_dim = unit_feature_dim
        self.data_preparation = data_preparation

        print "Reading testing sliding window list from: " + test_clip_path + '\n'
        self.test_samples = list()
        with open(test_clip_path) as f:
            for l in f:
                video_name = l.rstrip().split(" ")[0]
                clip_start = float(l.rstrip().split(" ")[1])
                clip_end = float(l.rstrip().split(" ")[2])
                self.test_samples.append((video_name, clip_start, clip_end))
        self.num_samples = len(self.test_samples)
        print "total test clips number is: " + str(len(self.test_samples)) + '\n'

    def __getitem__(self, index):
        video_name = self.test_samples[index][0]
        clip_start = self.test_samples[index][1]
        clip_end = self.test_samples[index][2]

        prop_feat = self.get_centeric_proposal_feature(self.feat_dir, video_name, clip_start, clip_end)
        left_feat = self.get_left_context_feature(self.feat_dir, video_name, clip_start, clip_end)
        right_feat = self.get_right_context_feature(self.feat_dir, video_name, clip_start, clip_end)
        feat = np.hstack((left_feat, prop_feat, right_feat))
        feat = self.data_preparation(feat)
        return video_name, feat, clip_start, clip_end


    def get_right_context_feature(self, feat_dir, movie_name, start, end):
        swin_step = self.unit_size
        all_feat = np.zeros([0, self.unit_feature_dim], dtype=np.float32)
        count = 0
        current_pos = end
        context_ext = False
        while count < self.ctx_num:
            swin_start = current_pos
            swin_end = current_pos + swin_step
            if os.path.exists(feat_dir + movie_name + ".mp4" + "_" + str(swin_start) + "_" + str(swin_end) + ".npy"):
                feat = np.load(feat_dir + movie_name + ".mp4" + "_" + str(swin_start) + "_" + str(swin_end) + ".npy")
                all_feat = np.vstack((all_feat, feat))
                context_ext = True
            current_pos += swin_step
            count += 1
        if context_ext:
            pool_feat = np.mean(all_feat, axis=0)
        else:
            pool_feat = np.zeros([self.unit_feature_dim], dtype=np.float32)
        return pool_feat

    def get_left_context_feature(self, feat_dir, movie_name, start, end):
        swin_step = self.unit_size
        all_feat = np.zeros([0, self.unit_feature_dim], dtype=np.float32)
        count = 0
        current_pos = start
        context_ext = False
        while count < self.ctx_num:
            swin_start = current_pos - swin_step
            swin_end = current_pos
            if os.path.exists(feat_dir + movie_name + ".mp4" + "_" + str(swin_start) + "_" + str(swin_end) + ".npy"):
                feat = np.load(feat_dir + movie_name + ".mp4" + "_" + str(swin_start) + "_" + str(swin_end) + ".npy")
                all_feat = np.vstack((all_feat, feat))
                context_ext = True
            current_pos -= swin_step
            count += 1
        if context_ext:
            pool_feat = np.mean(all_feat, axis=0)
        else:
            pool_feat = np.zeros([self.unit_feature_dim], dtype=np.float32)
        return pool_feat

    def get_centeric_proposal_feature(self, feat_dir, movie_name, start, end):

        swin_step = self.unit_size
        all_feat = np.zeros([0, self.unit_feature_dim], dtype=np.float32)
        current_pos = start
        while current_pos < end:
            swin_start = current_pos
            swin_end = swin_start + swin_step
            feat = np.load(feat_dir + movie_name + ".mp4" + "_" + str(swin_start) + "_" + str(swin_end) + ".npy")
            all_feat = np.vstack((all_feat, feat))
            current_pos += swin_step
        pool_feat = np.mean(all_feat, axis=0)
        return pool_feat

    def __len__(self):
        return len(self.test_samples)




class turnTrainDataset(data.Dataset):
    def __init__(self, ctx_num, unit_feature_dim, unit_size,
                 batch_size, video_length_info, feat_dir,
                 clip_gt_path, background_path, epoch_multiplier=1,
                 data_preparation=None):
        self.ctx_num = ctx_num
        self.unit_feature_dim = unit_feature_dim
        self.unit_size = unit_size
        self.batch_size = batch_size
        self.video_length_info = video_length_info
        self.visual_feature_dim = self.unit_feature_dim * 3
        self.feat_dir = feat_dir
        self.epoch_multiplier = epoch_multiplier
        self.data_preparation = data_preparation

        # prepare the foreground training sample list
        print "Reading foreground training sample list from: " + clip_gt_path + '\n'
        self.training_samples = list()
        with open(clip_gt_path) as f:
            for l in f:
                movie_name = l.rstrip().split(" ")[0]
                clip_start = float(l.rstrip().split(" ")[1])
                clip_end = float(l.rstrip().split(" ")[2])
                gt_start = float(l.rstrip().split(" ")[3])
                gt_end = float(l.rstrip().split(" ")[4])
                round_gt_start = np.round(gt_start / unit_size) * self.unit_size + 1
                round_gt_end = np.round(gt_end / unit_size) * self.unit_size + 1
                self.training_samples.append(
                    (movie_name, clip_start, clip_end, gt_start, gt_end, round_gt_start, round_gt_end, 1))
        print str(len(self.training_samples)) + " foreground training samples are read" + '\n'

        # prepare the background training sample list
        print "Reading background training sample list from: " + background_path + '\n'
        positive_num = len(self.training_samples) * 1.0
        with open(background_path) as f:
            for l in f:
                # control the ratio between background samples and positive samples to be 10:1
                if random.random() > 10.0 * positive_num / 270000: continue
                movie_name = l.rstrip().split(" ")[0]
                clip_start = float(l.rstrip().split(" ")[1])
                clip_end = float(l.rstrip().split(" ")[2])
                self.training_samples.append((movie_name, clip_start, clip_end, 0, 0, 0, 0, 0))
        self.num_samples = len(self.training_samples)
        print str(len(self.training_samples)) + " background training samples are read" + '\n'

    def __getitem__(self, index):
        real_index = index % len(self.training_samples)
        video_name = self.training_samples[real_index][0]
        clip_start = self.training_samples[real_index][1]
        clip_end = self.training_samples[real_index][2]
        round_gt_start = self.training_samples[real_index][5]
        round_gt_end = self.training_samples[real_index][6]
        label = self.training_samples[real_index][7]

        start_offset, end_offset = self.calculate_regoffset(clip_start, clip_end, round_gt_start, round_gt_end)
        prop_feat = self.get_centeric_proposal_feature(self.feat_dir, video_name, clip_start, clip_end)
        left_feat = self.get_left_context_feature(self.feat_dir, video_name, clip_start, clip_end)
        right_feat = self.get_right_context_feature(self.feat_dir, video_name, clip_start, clip_end)
        feat = np.hstack((left_feat, prop_feat, right_feat))
        feat = self.data_preparation(feat)
        return feat, label, start_offset, end_offset

    def calculate_regoffset(self, clip_start, clip_end, round_gt_start, round_gt_end):
        start_offset = (round_gt_start - clip_start) / self.unit_size
        end_offset = (round_gt_end - clip_end) / self.unit_size
        return start_offset, end_offset

    def get_right_context_feature(self,feat_dir,movie_name,start,end):
        swin_step = self.unit_size
        all_feat = np.zeros([0, self.unit_feature_dim], dtype=np.float32)
        count = 0
        current_pos = end
        context_ext = False
        while count < self.ctx_num:
            swin_start = current_pos
            swin_end = current_pos + swin_step
            if os.path.exists(feat_dir + movie_name + ".mp4" + "_" + str(swin_start) + "_" + str(swin_end) + ".npy"):
                feat = np.load(feat_dir + movie_name + ".mp4" + "_" + str(swin_start) + "_" + str(swin_end) + ".npy")
                all_feat = np.vstack((all_feat, feat))
                context_ext = True
            current_pos += swin_step
            count += 1
        if context_ext:
            pool_feat = np.mean(all_feat, axis=0)
        else:
            pool_feat = np.zeros([self.unit_feature_dim], dtype=np.float32)
        return pool_feat

    def get_left_context_feature(self,feat_dir,movie_name,start,end):
        swin_step = self.unit_size
        all_feat = np.zeros([0, self.unit_feature_dim], dtype=np.float32)
        count = 0
        current_pos = start
        context_ext = False
        while count < self.ctx_num:
            swin_start = current_pos - swin_step
            swin_end = current_pos
            if os.path.exists(feat_dir + movie_name + ".mp4" + "_" + str(swin_start) + "_" + str(swin_end) + ".npy"):
                feat = np.load(feat_dir + movie_name + ".mp4" + "_" + str(swin_start) + "_" + str(swin_end) + ".npy")
                all_feat = np.vstack((all_feat, feat))
                context_ext = True
            current_pos -= swin_step
            count += 1
        if context_ext:
            pool_feat = np.mean(all_feat, axis=0)
        else:
            pool_feat = np.zeros([self.unit_feature_dim], dtype=np.float32)
        return pool_feat

    def get_centeric_proposal_feature(self,feat_dir,movie_name,start,end):
        swin_step = self.unit_size
        all_feat = np.zeros([0, self.unit_feature_dim], dtype=np.float32)
        current_pos = start
        while current_pos < end:
            swin_start = current_pos
            swin_end = swin_start + swin_step
            feat = np.load(feat_dir + movie_name + ".mp4" + "_" + str(swin_start) + "_" + str(swin_end) + ".npy")
            all_feat = np.vstack((all_feat, feat))
            current_pos += swin_step
        pool_feat = np.mean(all_feat, axis=0)
        return pool_feat

    def __len__(self):
        return len(self.training_samples) * self.epoch_multiplier

