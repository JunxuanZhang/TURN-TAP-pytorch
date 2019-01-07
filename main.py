import io
import os
import requests
import turn_model
import pickle
import torch
import numpy as np
from turn_opts import *
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm
from turn_dataset import turnTrainDataset
from turn_dataset import turnTestDataset
from operator import itemgetter
import pandas as pd
from progressbar import *


def main():
    global args
    args = parser.parse_args()

    train_video_length_info = {}
    with open(args.val_video_length) as f:
        for l in f:
            train_video_length_info[l.rstrip().split(" ")[0]] = int(l.rstrip().split(" ")[2])

    test_video_length_info = {}
    with open(args.test_video_length) as f:
        for l in f:
            test_video_length_info[l.rstrip().split(" ")[0]] = int(l.rstrip().split(" ")[2])

    model = turn_model.TURN(tr_batch_size=args.tr_batch_size, ts_batch_size=args.ts_batch_size,
                            lambda_reg=args.lambda_reg, unit_feature_dim=args.unit_feature_dim,
                            middle_layer_dim=args.middle_layer_dim, dropout=args.dropout)

    policies = model.get_optim_policies()
    data_preparation = model.data_preparation()

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    cudnn.benchmark = True

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint (epoch {})"
                   .format(checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))


    train_loader = torch.utils.data.DataLoader(
        turnTrainDataset(ctx_num=args.ctx_number, unit_feature_dim=args.unit_feature_dim,
                         unit_size=args.unit_size, batch_size=args.tr_batch_size,
                         video_length_info=train_video_length_info,
                         feat_dir=args.train_featmap_dir, clip_gt_path=args.train_clip_path,
                         background_path=args.background_path,
                         data_preparation=data_preparation),
        batch_size=args.tr_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        turnTestDataset(ctx_num=args.ctx_number, feat_dir=args.test_featmap_dir,
                        test_clip_path=args.test_clip_path, batch_size=args.ts_batch_size,
                        unit_feature_dim=args.unit_feature_dim, unit_size=args.unit_size,
                        data_preparation=data_preparation),
        batch_size=args.ts_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)

    criterion = [torch.nn.CrossEntropyLoss(), RegressionLoss().cuda()]

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, weight_decay_mult: {} \n'.format(
            group['name'], len(group['params']), group['lr_mult'], group['weight_decay_mult'])))

    optimizer = torch.optim.SGD(policies, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # load necessary data for evaluation
    clip_length_file = "./train_val_files/val_training_samples.txt"
    clip_prob = compute_prob_dist(clip_length_file)
    frm_nums = pickle.load(open("./train_val_files/frm_num.pkl"))

    ground_truth_url = ('https://gist.githubusercontent.com/cabaf/'
                        'ed34a35ee4443b435c36de42c4547bd7/raw/'
                        '952f17b9cdc6aa4e6d696315ba75091224f5de97/'
                        'thumos14_test_groundtruth.csv')
    s = requests.get(ground_truth_url).content
    ground_truth = pd.read_csv(io.StringIO(s.decode('utf-8')), sep=' ')
    
    if args.evaluate:
        evaluate(val_loader, model, clip_prob, frm_nums, ground_truth)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation list
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            AR, AN = evaluate(val_loader, model, clip_prob, frm_nums, ground_truth)

            # save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'AR': AR,
                'AN': AN
            }, epoch, AR)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    cls_losses = AverageMeter()
    res_losses = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    optimizer.zero_grad()

    for i, (feats, labels, start_offsets, end_offsets) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input_feats = torch.autograd.Variable(feats).cuda()
        input_labels = torch.autograd.Variable(labels).cuda()
        start_offsets = torch.autograd.Variable(start_offsets).cuda().float()
        end_offsets = torch.autograd.Variable(end_offsets).cuda().float()
        pred_labels = model(input_feats)

        cls_loss = criterion[0](pred_labels[:, :2], input_labels)
        res_loss = criterion[1](pred_labels[:, 2:], input_labels.float(), start_offsets, end_offsets)
        cls_losses.update(cls_loss.data[0], feats.size(0))
        res_losses.update(res_loss.data[0], torch.sum(labels))
        loss = cls_loss + args.lambda_reg * res_loss
        losses.update(loss.data[0], feats.size(0))

        # compute gradient and do SGD step
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print('Clipping gradient: {} with coef {}'.format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\n'
                  'Classification Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                  'Regression Loss {res_loss.val:.4f} ({res_loss.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\n'
                  .format(epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, cls_loss=cls_losses,
                          res_loss=res_losses, lr=optimizer.param_groups[0]['lr'])
                  )


def evaluate(val_loader, model, clip_prob, frm_nums, ground_truth):

    model.eval()

    print('Begin to evaluate the sliding window proposals in test set ...\n')
    max_num = len(val_loader)
    widgets = ['Processing testing proposals' + ': ', Percentage(), ' ', Bar('>'), ' ', Timer()]
    pbar = ProgressBar(widgets=widgets, maxval=max_num).start()

    result_dict = dict()
    for i, (video_names, feats, starts, ends) in enumerate(val_loader):
        input_feats = torch.autograd.Variable(feats, volatile=True).cuda()
        pred_labels = model(input_feats).cpu().data.numpy()

        round_reg_start = starts.numpy() + np.round(pred_labels[:, 2]) * args.unit_size
        round_reg_end = ends.numpy() + np.round(pred_labels[:, 3]) * args.unit_size
        softmax_score = softmax(pred_labels[:, 0:2])
        action_score = softmax_score[:, 1]

        reg_start = round_reg_start / 30.0
        reg_end = round_reg_end / 30.0

        for ind, video_name in enumerate(video_names):
            if video_name not in result_dict:
                result_dict[video_name] = [[reg_start[ind], reg_end[ind], action_score[ind]]]
            else:
                result_dict[video_name].append([reg_start[ind], reg_end[ind], action_score[ind]])

        pbar.update(i)

    pbar.finish()

    print('Finish the process of prediction, now begin to evaluate ...\n')

    for _key in result_dict:
        result_dict[_key] = sorted(result_dict[_key], key=itemgetter(2))[::-1]
        result_dict[_key] = np.array(result_dict[_key])
        x1 = result_dict[_key][:, 0]
        x2 = result_dict[_key][:, 1]
        s = result_dict[_key][:, 2]
        for k in range(x1.shape[0]):
            clip_length_index = [16, 32, 64, 128, 256, 512].index(
                min([16, 32, 64, 128, 256, 512], key=lambda x: abs(x - int(x2[k] * 30 - x1[k] * 30))))
            s[k] = s[k] * clip_prob[clip_length_index]
        new_ind = np.argsort(s)[::-1]
        result_dict[_key] = result_dict[_key][new_ind, :]

    rows = pkl2dataframe(frm_nums, result_dict)
    daps_results = pd.DataFrame(rows, columns=['f-end', 'f-init', 'score', 'video-frames', 'video-name'])

    video_lst = daps_results['video-name'].unique()

    # For each video, computes tiou scores among the retrieved proposals.
    score_lst = []
    total_gt_num = 0.
    total_pr_num = 0.
    for videoid in video_lst:
        # Get proposals for this video.
        prop_idx = daps_results['video-name'] == videoid
        this_video_proposals = daps_results[prop_idx][['f-init',
                                                    'f-end']].values
        # Sort proposals by score.
        sort_idx = daps_results[prop_idx]['score'].argsort()[::-1]
        this_video_proposals = this_video_proposals[sort_idx, :]

        # Get ground-truth instances associated to this video.
        gt_idx = ground_truth['video-name'] == videoid
        this_video_ground_truth = ground_truth[gt_idx][['f-init', 'f-end']].values

        # Compute tiou scores.
        tiou, gt_num, pr_num = segment_tiou(this_video_ground_truth, this_video_proposals)
        score_lst.append(tiou)
        total_gt_num += gt_num
        total_pr_num += pr_num

    tiou_thresholds = np.linspace(0.5, 1.0, 11)
    AN_list = [50, 100, 200]
    ave_pr_per_video = total_pr_num / video_lst.shape[0]
    pcn_lst = [AN / ave_pr_per_video for AN in AN_list]
    matches = np.empty((video_lst.shape[0], len(pcn_lst)))
    recall = np.empty((tiou_thresholds.shape[0], len(pcn_lst)))
    # Iterates over each tiou threshold.
    for ridx, tiou in enumerate(tiou_thresholds):

        # Inspect positives retrieved per video at different
        # number of proposals (percentage of the total retrieved).
        for i, score in enumerate(score_lst):

            for j, pcn in enumerate(pcn_lst):
                # Get number of proposals as a percentage of total retrieved.
                nr_proposals = int(np.ceil(score.shape[1] * pcn))
                # Find proposals that satisfies minimum tiou threhold.
                matches[i, j] = ((score[:, :nr_proposals] >= tiou).sum(axis=1) > 0).sum()

        # Computes recall given the set of matches per video.
        recall[ridx, :] = matches.sum(axis=0) / total_gt_num

    # Recall is averaged.
    recall = recall.mean(axis=0)

    # Get the average number of proposals per video.
    proposals_per_video = [pcn * (float(daps_results.shape[0]) / video_lst.shape[0]) for pcn in pcn_lst]

    for i in range(len(AN_list)):
        print('AR@AN={}: {}'.format(AN_list[i], recall[i]))

    return recall, proposals_per_video



def segment_tiou(target_segments, test_segments):
    """Compute intersection over union btw segments
    Parameters
    ----------
    target_segments : ndarray
        2-dim array in format [m x 2:=[init, end]]
    test_segments : ndarray
        2-dim array in format [n x 2:=[init, end]]
    Outputs
    -------
    tiou : ndarray
        2-dim array [m x n] with IOU ratio.
    Note: It assumes that target-segments are more scarce that test-segments
    """
    if target_segments.ndim != 2 or test_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    m, n = target_segments.shape[0], test_segments.shape[0]
    tiou = np.empty((m, n))
    for i in xrange(m):
        tt1 = np.maximum(target_segments[i, 0], test_segments[:, 0])
        tt2 = np.minimum(target_segments[i, 1], test_segments[:, 1])

        # Non-negative overlap score
        intersection = (tt2 - tt1 + 1.0).clip(0)
        union = ((test_segments[:, 1] - test_segments[:, 0] + 1) +
                 (target_segments[i, 1] - target_segments[i, 0] + 1) -
                 intersection)
        # Compute overlap as the ratio of the intersection
        # over union of two segments at the frame level.
        tiou[i, :] = intersection / union
    return tiou, m, n

def pkl2dataframe(frm_nums, result_dict):
    data_frame = []
    movie_fps = pickle.load(open("./train_val_files/movie_fps.pkl"))
    for _key in result_dict:
        fps = movie_fps[_key]
        frm_num = frm_nums[_key]
        for line in result_dict[_key]:
            start = int(line[0]*30)
            end = int(line[1]*30)
            score = float(line[2])
            data_frame.append([end, start, score, frm_num, _key])
    return data_frame


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)


def save_checkpoint(state, epoch, AR, filename='checkpoint.pth.tar'):
    filename = 'turn' + '_'.join((args.snapshot_pref, args.dataset, args.modality.lower(),
                                  'epoch_{:02d}_AR@AN={AR1:.4f}_{AR2:.4f}_{AR3:.4f}'
                                  .format(epoch, AR1=AR[0], AR2=AR[1], AR3=AR[2]), filename))
    torch.save(state, 'results/' + filename)


def adjust_learning_rate(optimizer, epoch, lr_steps):
    # Set the learning rate to the initial LR decayed by 10 every 30 epoches
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['weight_decay_mult']


def compute_prob_dist(clip_length_file):

    length_dist = {}
    for _key in [16,32,64,128,256,512]:
        length_dist[_key] = 0
    with open(clip_length_file) as f:
        for line in f:
            clip_length = int(line.split(" ")[2])-int(line.split(" ")[1])
            length_dist[clip_length] += 1
    sample_sum = sum([length_dist[_key] for _key in length_dist])
    prob = [float(length_dist[_key])/sample_sum for _key in [16,32,64,128,256,512]]
    return prob



def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"


class _Loss(torch.nn.Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average


class RegressionLoss(_Loss):
    def forward(self, preds, labels, s_offsets, e_offsets):
        _assert_no_grad(s_offsets)
        _assert_no_grad(e_offsets)
        loss = torch.mean((torch.abs(preds[:, 0] - s_offsets) + torch.abs(preds[:, 1] - e_offsets)) * labels)
        return loss


class AverageMeter(object):
    # Computes and stores the average and current value
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()



