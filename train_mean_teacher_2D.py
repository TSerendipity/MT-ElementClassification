import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

# from dataloaders import utils
# from dataloaders.dataset import (BaseDataSets, RandomGenerator,
#                                  TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume
# import torch
# from torch import nn
import datetime
import numpy
from typing import Dict, Any, Tuple
from pathlib import Path
from torch.nn.modules.loss import _Loss as Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler, DistributedSampler
# from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, PreTrainedTokenizer
from sketch_model.configs import SketchModelConfig, config_with_arg, ModelConfig
from sketch_model.datasets import build_dataset
from sketch_model.utils import misc as utils, f1score, r2score, accuracy_simple, creat_confusion_matrix, precision, recall, confusion_matrix
from sketch_model.utils import NestedTensor
# from sketch_model.utils.sampler import TwoBatchSampler
from sketch_model.utils.utils import flatten_input_for_model
from sketch_model.model import build, SketchLayerClassifierModel


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../../EGFE/sketch_dataset/data', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='EGFE/Mean_Teacher', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='EGFE', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=3000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=10,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
# parser.add_argument('--patch_size', type=list,  default=[256, 256],
#                     help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=3,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=4,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=20,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.9, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()


def init_config(
        config: SketchModelConfig) -> Tuple[SketchModelConfig, Dict[str, Any]]:
    '''
    fix the seed for reproducibility
    if resume, loading config checkpoint
    '''

    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    checkpoint = None
    if config.resume:
        checkpoint = torch.load(config.resume, map_location='cpu')
        saved_config: SketchModelConfig = checkpoint['config']
        config.start_epoch = checkpoint['epoch'] + 1
        for field in fields(ModelConfig):
            # override the current config by using saved config
            config.__setattr__(field.name,
                               saved_config.__getattribute__(field.name))
    return config, checkpoint

def init_model(
    config: SketchModelConfig, checkpoint: Dict[str, Any], device: torch.device, ema=False
) -> Tuple[PreTrainedTokenizer, SketchLayerClassifierModel, Loss, optim.Optimizer,
           _LRScheduler]:
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name)
    tokenizer.model_max_length = config.max_name_length
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id
    model, criterion = build(config)
    model.to(device)

    model_without_ddp = model
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[config.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters(
        ) if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": config.lr_backbone,
        },
    ]
    optimizer = optim.AdamW(param_dicts,
                                  lr=args.base_lr,
                                  weight_decay=config.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, config.lr_drop)

    if checkpoint is not None:
        print("Loading Checkpoint...")
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
    if ema:
            for param in model.parameters():
                param.detach_()
                
    return (tokenizer, model, model_without_ddp, criterion, optimizer, lr_scheduler)



def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 0.0001
    # return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        

def train(args, snapshot_path, config: SketchModelConfig):
    utils.init_distributed_mode(config)
    config, checkpoint = init_config(config)
    device = torch.device(config.device)
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    
    def worker_init_fn(worker_id):
            random.seed(args.seed + worker_id)
    
    tokenizer, model, model_without_ddp, criterion, optimizer, lr_scheduler = init_model(
        config, checkpoint, device)
    ema_tokenizer, ema_model, ema_model_without_ddp, ema_criterion, ema_optimizer, ema_lr_scheduler = init_model(
        config, checkpoint, device, ema=True)
    n_parameters = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print('number of params:', n_parameters)

    db_train = build_dataset(config.train_index_json,
                                  Path(config.train_index_json).parent.__str__(),
                                  tokenizer,
                                  cache_dir=config.cache_dir,
                                  use_cache=config.use_cache,
                                  remove_text=config.remove_text)
    sampler_train = RandomSampler(db_train)
    batch_sampler_train = BatchSampler(sampler_train,
                                       config.batch_size,
                                       drop_last=True)
    trainloader = DataLoader(db_train,
                                   batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn,
                                   num_workers=config.num_workers)
    
    db_val = build_dataset(config.test_index_json,
                                Path(config.test_index_json).parent.__str__(),
                                tokenizer,
                                cache_dir=config.cache_dir,
                                use_cache=config.use_cache,
                                remove_text=config.remove_text)
    if config.distributed:
        sampler_val = DistributedSampler(db_val, shuffle=False)
    else:
        sampler_val = SequentialSampler(db_val)
    valloader = DataLoader(db_val,
                                 batch_size,
                                 sampler=sampler_val,
                                 drop_last=False,
                                 collate_fn=utils.collate_fn,
                                 num_workers=config.num_workers)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_train_acc, best_train_f1, best_test_acc, best_test_f1, best_test_precision, best_test_recall = [0] * 6
    iterator = tqdm(range(max_epoch), ncols=70)
    
    print("Start training")
    start_time = time.time()
    for epoch in iterator:
        model.train()
        criterion.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter(
            'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter(
            'acc', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter(
            'f1', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter(
            'precision', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter(
            'recall', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter(
            'r2', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = f'Task:{config.task_name} Epoch: [{epoch}]'
        print_freq = 10
        y_pred, y_true = [], []
        
        for (imgs_batch, names_batch, bboxes_batch, colors_batch, classes_batch), targets in metric_logger.log_every(
                 trainloader, print_freq, header):
            model.train()
            criterion.train()
            imgs_batch: NestedTensor = imgs_batch.to(device)  # [8, 176, 3, 64, 64]
            names_batch: NestedTensor = names_batch.to(device)  # [8, 176, 32]
            bboxes_batch: NestedTensor = bboxes_batch.to(device)  # [8, 176, 4]
            colors_batch: NestedTensor = colors_batch.to(device)  # [8, 176, 4]
            classes_batch: NestedTensor = classes_batch.to(device)  # [8, 176]

            imgs_tensor, batch_sizes = imgs_batch.decompose()
            noise = torch.clamp(torch.randn_like(
                imgs_tensor) * 0.1, -0.2, 0.2)
            ema_imgs_tensor = imgs_tensor + noise
            ema_imgs_batch = NestedTensor(ema_imgs_tensor, batch_sizes)

            outputs = model(flatten_input_for_model(
                imgs_batch, names_batch, bboxes_batch, colors_batch, classes_batch))
            outputs_soft = outputs
            with torch.no_grad():
                ema_output = ema_model(flatten_input_for_model(
                    # imgs_batch, names_batch, bboxes_batch, colors_batch, classes_batch))
                    ema_imgs_batch, names_batch, bboxes_batch, colors_batch, classes_batch))
                    # ema_imgs_inputs, unlabeled_names_batch, unlabeled_bboxes_batch, unlabeled_colors_batch, unlabeled_classes_batch))
                # ema_output_soft = torch.softmax(ema_output, dim=1)
                ema_output_soft = ema_output

            targets = [t.to(device) for t in targets]
            batch_ce_loss = torch.tensor(0.0, device=device)
            acc, f1, r2, precision_score, recall_score = 0, 0, 0, 0, 0
            for i in range(args.labeled_bs):
                packed = outputs[i][:len(targets[i])]  # [120, 4]
                ce_loss = criterion(packed, targets[i])
                batch_ce_loss += ce_loss
                pred = packed.max(-1)[1]
                # for confusion matrix
                y_pred.extend(pred.cpu().numpy())
                y_true.extend(targets[i].cpu().numpy())
                # for logger
                acc += accuracy_simple(pred, targets[i])
                f1 += f1score(pred, targets[i])
                r2 += r2score(pred, targets[i])
                precision_score += precision(pred, targets[i])
                recall_score += recall(pred, targets[i])
            acc, f1, r2, precision_score, recall_score = numpy.array(
                [acc, f1, r2, precision_score, recall_score]) / args.labeled_bs
            # 计算平均损失
            batch_ce_loss /= args.labeled_bs
            supervised_loss = batch_ce_loss
            consistency_weight = get_current_consistency_weight(iter_num//8)

            if iter_num < 50:
                consistency_loss = 0.0
            else:
                consistency_loss = torch.mean(
                    (outputs_soft[args.labeled_bs:]-ema_output_soft[args.labeled_bs:])**2)
            loss = supervised_loss + consistency_weight * consistency_loss
            # loss = supervised_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            metric_logger.update(acc=acc,
                             f1=f1,
                             r2=r2,
                             precision=precision_score,
                             recall=recall_score)
            metric_logger.update(loss=batch_ce_loss.detach())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            
        metric_logger.synchronize_between_processes()
        train_stats = {k: meter.global_avg
             for k, meter in metric_logger.meters.items()}
        print("[Train] Averaged stats:", train_stats)
        lr_scheduler.step()
            
        update_ema_variables(model, ema_model, args.ema_decay, iter_num)
        # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr_
                        
        iter_num = iter_num + 1
        # writer.add_scalar('info/lr', lr_, iter_num)
        writer.add_scalar('info/total_loss', loss, iter_num)
        writer.add_scalar('info/batch_ce_loss', batch_ce_loss, iter_num)
        writer.add_scalar('info/consistency_loss',
                          consistency_loss, iter_num)
        writer.add_scalar('info/consistency_weight',
                          consistency_weight, iter_num)

        logging.info(
            'iteration %d : loss : %f, loss_ce: %f' %
            (iter_num, loss.item(), batch_ce_loss.item()))

        # if iter_num % 20 == 0:
        #     image = imgs_batch[1, 0:1, :, :]
        #     writer.add_image('train/Image', image, iter_num)
        #     outputs = torch.argmax(torch.softmax(
        #         outputs, dim=1), dim=1, keepdim=True)
        #     writer.add_image('train/Prediction',
        #                      outputs[1, ...] * 50, iter_num)
        #     labs = labels_batch[1, ...].unsqueeze(0) * 50
        #     writer.add_image('train/GroundTruth', labs, iter_num)

        #ceshi
        test_stats = evaluate(config, model, criterion, valloader,device)
        y_pred, y_true = test_stats[0]
        test_stats = test_stats[1]
        log_stats = {
            **{f'train_{k}': v
               for k, v in train_stats.items()},
            **{f'test_{k}': v
               for k, v in test_stats.items()}, 'epoch': epoch,
            'n_parameters': n_parameters
        }

        best_train_acc, best_train_f1, best_test_acc, best_test_f1,\
            best_test_precision, best_test_recall = np.maximum(
                (best_train_acc, best_train_f1, best_test_acc,
                 best_test_f1, best_test_precision, best_test_recall),
                (train_stats['acc'], train_stats['f1'], test_stats['acc'],
                 test_stats['f1'], test_stats['precision'], test_stats['recall']))
        if utils.is_main_process():
            writer.add_figure("test/Confusion matrix",
                              creat_confusion_matrix(y_true, y_pred), epoch)
            writer.add_scalar('train/loss', train_stats['loss'], epoch)
            writer.add_scalar('train/acc', train_stats['acc'], epoch)
            writer.add_scalar('train/f1', train_stats['f1'], epoch)
            writer.add_scalar('train/r2', train_stats['r2'], epoch)
            writer.add_scalar('test/loss', test_stats['loss'], epoch)
            writer.add_scalar('test/acc', test_stats['acc'], epoch)
            writer.add_scalar('test/f1', test_stats['f1'], epoch)
            writer.add_scalar('test/r2', test_stats['r2'], epoch)
            writer.add_scalar('test/precision', test_stats['precision'], epoch)
            writer.add_scalar('test/recall', test_stats['recall'], epoch)
                # writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'],
                #                   epoch)

#             if train_stats['acc'] == best_train_acc:
#                 save_model_path = os.path.join(snapshot_path,
#                                               'iter_{}_dice_{}.pth'.format(
#                                                   iter_num, round(best_train_acc, 4)))
#                 save_best = os.path.join(snapshot_path,
#                                          '{}_best_model.pth'.format(args.model))
#                 torch.save(model.state_dict(), save_model_path)
#                 torch.save(model.state_dict(), save_best)

#             if iter_num % 300 == 0:
#                 save_model_path = os.path.join(
#                     snapshot_path, 'iter_' + str(iter_num) + '.pth')
#                 torch.save(model.state_dict(), save_model_path)
#                 logging.info("save model to {}".format(save_model_path))
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.flush()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('Best test acc: {}'.format(best_test_acc))
    print('Best test f1: {}'.format(best_test_f1))
    print('Best test precision: {}'.format(best_test_precision))
    print('Best test recall: {}'.format(best_test_recall))
    print('Best train acc: {}'.format(best_train_acc))
    print('Best train f1: {}'.format(best_train_f1))
    return "Training Finished!"

@torch.no_grad()
def evaluate(config: SketchModelConfig,
             model: nn.Module,
             criterion: nn.Module,
             dataloader: DataLoader,
             device: torch.device,
             eval_model: str = "macro"):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'acc', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'f1', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'r2', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'precision', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'recall', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Task:{config.task_name} Test:'
    print_freq = 10
    y_pred, y_true = [], []
    cal_tiny: bool = False
    # comparison for tiny objects
    if cal_tiny:
        tiny_pred, tiny_true = [], []
    for (batch_img, batch_name, batch_bbox, batch_color,
         batch_class), targets in metric_logger.log_every(
             dataloader, print_freq, header):
        batch_img = batch_img.to(device)
        batch_name = batch_name.to(device)
        batch_bbox = batch_bbox.to(device)
        batch_color = batch_color.to(device)
        batch_class = batch_class.to(device)
        targets = [t.to(device) for t in targets]

        outputs = model(
            flatten_input_for_model(batch_img, batch_name, batch_bbox,
                                    batch_color, batch_class))
        batch_ce_loss = torch.tensor(0.0, device=device)
        acc, f1, r2, precision_score, recall_score = 0, 0, 0, 0, 0
        for i in range(len(targets)):
            if cal_tiny:
                bbox = batch_bbox.tensors[i][:len(targets[i])]
                bbox_areas = [(bbox[bindex][2] - bbox[bindex][0]) * (bbox[bindex]
                                                                     [3] - bbox[bindex][1]) for bindex in range(bbox.shape[0])]
                bbox_sizes = [False if (bbox[bindex][2] - bbox[bindex][0] > 32 or bbox[bindex]
                                        [3] - bbox[bindex][1] > 32) else True for bindex in range(bbox.shape[0])]
                area_check = [True if area <= 32 *
                              32 else False for area in bbox_areas]
                area_check = [a and b for a, b in zip(bbox_sizes, area_check)]
            packed = outputs[i][:len(targets[i])]
            ce_loss = criterion(packed, targets[i])
            batch_ce_loss += ce_loss
            pred = packed.max(-1)[1]
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(targets[i].cpu().numpy())
            acc += accuracy_simple(pred, targets[i])
            r2 += r2score(pred, targets[i])
            precision_score += precision(pred, targets[i], average=eval_model)
            recall_score += recall(pred, targets[i], average=eval_model)
            if cal_tiny:
                tiny_pred.extend(pred[area_check].cpu().numpy())
                tiny_true.extend(targets[i][area_check].cpu().numpy())
                precision_score += precision(pred[area_check], targets[i][area_check], average=eval_model)
                recall_score += recall(pred[area_check], targets[i][area_check], average=eval_model)
        acc, r2, precision_score, recall_score = numpy.array(
            [acc, r2, precision_score, recall_score]) / len(targets)
        batch_ce_loss /= len(targets)
        f1 = 2 * precision_score * recall_score / (precision_score +
                                                   recall_score)
        metric_logger.update(acc=acc,
                             f1=f1,
                             r2=r2,
                             precision=precision_score,
                             recall=recall_score)
        metric_logger.update(loss=batch_ce_loss.detach())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg
             for k, meter in metric_logger.meters.items()}
    print("[TEST] Averaged stats:", stats)
    return [(y_pred, y_true),
            stats]




if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path, config_with_arg())
