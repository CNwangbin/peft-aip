import yaml
import argparse
import logging
import os
import wandb
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import time
from collections import OrderedDict
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import torch.distributed as dist
import pytorch_warmup as warmup

from dataset import EsmDataset
from models import FineTuneESM, PeftESM
from utils import set_seed, get_train_step, save_checkpoint, load_model_checkpoint, update_summary, print_trainable_parameters
from metrics import AverageMeter, compute_metrics

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='esm-finetune model Training Config',
                                                 add_help=False)
parser.add_argument('-c',
                    '--config',
                    default='',
                    type=str,
                    metavar='FILE',
                    help='YAML config file specifying default arguments')
parser = argparse.ArgumentParser(
    description='ESM linear head finetune classification trainning config')

parser.add_argument('-peft',
                    '--is-peft',
                    default=False,
                    type=bool,
                    help='whether to use peft model')
parser.add_argument('-nl',
                    '--num-layers',
                    default=0,
                    type=int,
                    help='number of layers used in finetuning of esm model')
parser.add_argument('-rmt',
                    '--remove-top-layers',
                    default=0,
                    type=int,
                    help='number of top layers to remove')
parser.add_argument('-pm',
                    '--pretrain-model',
                    default='esm2_t33_650M_UR50D',
                    type=str,
                    choices=('esm2_t6_8M_UR50D',
                            'esm2_t12_35M_UR50D',
                            'esm2_t30_150M_UR50D',
                            'esm2_t33_650M_UR50D',
                            'esm2_t36_3B_UR50D',
                            'esm2_t48_15B_UR50D',))
parser.add_argument('-nlora',
                    '--num-end-lora-layers',
                    default=0,
                    type=int,
                    help='number of layers used in lora finetuning')
parser.add_argument('-nr',
                    '--num-lora-r',
                    default=8,
                    type=int,
                    help='number of lora r')
parser.add_argument('-na',
                    '--num-lora-alpha',
                    default=8,
                    type=int,
                    help='number of lora alpha')
parser.add_argument('-im',
                    '--inference-mode',
                    default=False,
                    type=bool,
                    help='inference mode')
parser.add_argument('-ld',
                    '--lora-dropout',
                    default='0.0',
                    type=str,
                    help='lora dropout')
parser.add_argument('-bias',
                    '--bias',
                    default='none',
                    type=str,
                    help='bias')
parser.add_argument('-train',
                    '--train-path',
                    default='data/data_AIP-MDL/train_df.pkl',
                    type=str,
                    help='path of train data')
parser.add_argument('-valid',
                    '--valid-path',
                    default='data/data_AIP-MDL/valid_df.pkl',
                    type=str,
                    help='path of valid data')
parser.add_argument('-test',
                    '--test-path',
                    default='data/data_AIP-MDL/test_df.pkl',
                    type=str,
                    help='path of test data')
parser.add_argument('--resume',
                    default=None,
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs',
                    default=5,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-j',
                    '--workers',
                    type=int,
                    default=4,
                    metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('-b',
                    '--batch-size',
                    default=32,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 32) per gpu')
parser.add_argument('-lr',
                    '--learning-rate',
                    default='3e-5',
                    type=str,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('-mlr',
                    '--min-learning-rate',
                    default=1e-6,
                    type=float,
                    metavar='MLR',
                    help='minimal learning rate',)
parser.add_argument(
    '--lr-schedule',
    default='cosine',
    type=str,
    metavar='SCHEDULE',
    choices=[None, 'cosine', 'exponential'],
    help='Type of LR schedule: {}, {}'.format(None, 'exponential'))
parser.add_argument('--gamma',
                    default=0.6,
                    type=float,
                    help='exponential learning rate decay factor')
parser.add_argument('--optimizer',
                    default='adamw',
                    type=str,
                    choices=('sgd', 'adam', 'adamw'))
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('-wd',
                    '--weight-decay',
                    default=0.01,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 0.0)',
                    dest='weight_decay')
parser.add_argument('--warmup_period',
                    default=100,
                    type=int,
                    help='warmup period, if > 0, use linear warmup')
parser.add_argument(
    '--no-checkpoints',
    action='store_false',
    dest='save_checkpoints',
    help='do not store any checkpoints, useful for benchmarking',
)
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--output-dir',
                    default='./work_dirs',
                    type=str,
                    help='output directory for model and log')
parser.add_argument('--log-wandb',
                    action='store_true',
                    help='while to use wandb log systerm')
parser.add_argument('--experiment', default='peft-aip', type=str)
parser.add_argument('--patience',
                    default=2,
                    type=int,
                    help='patience of evaluation Accuracy metric')
parser.add_argument('--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--training-only',
                    action='store_true',
                    help='do not evaluate')
parser.add_argument(
    '--amp',
    action='store_true',
    default=False,
    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--native-amp',
                    action='store_true',
                    default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument(
    '--early-stopping-patience',
    default=-1,
    type=int,
    metavar='N',
    help='early stopping after N epochs without improving',
)
parser.add_argument(
    '--gradient_accumulation_steps',
    default=1,
    type=int,
    metavar='N',
    help='=To run gradient descent after N steps',
)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument(
    '--static-loss-scale',
    type=float,
    default=1,
    help='Static loss scale',
)
parser.add_argument(
    '--dynamic-loss-scale',
    action='store_true',
    help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
    '--static-loss-scale.',
)

def main(args):

    print('log to wandb')
    wandb.init(project=args.experiment, config=args, entity='cnwangbi')
    set_seed(args.seed)

    if args.is_peft:
        logger.info('Using Parameter Efficient Fine-Tuning ESM model')
    else:
        logger.info('Using directly finetune ESM model')

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.local_rank = 0

    args.gpu = 0
    args.world_size = 1
    args.rank = 0

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()

        logger.info(
            'Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
            % (args.rank, args.world_size))
    else:
        logger.info('Training with a single process on %s .' % args.gpu)



    # Dataset and DataLoader
    train_dataset = EsmDataset(pd.read_pickle(args.train_path))
    valid_dataset = EsmDataset(pd.read_pickle(args.valid_path))
    test_dataset = EsmDataset(pd.read_pickle(args.test_path))

    # dataloder
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=valid_dataset.collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=test_dataset.collate_fn,
    )

    # PEFT-SP using LoRA
    if args.num_end_lora_layers > 0:
        logger.info('Using PeftESM model')
        model = PeftESM(num_end_lora_layers = args.num_end_lora_layers,
                        num_lora_r = args.num_lora_r,
                        num_lora_alpha = args.num_lora_alpha,
                        inference_mode = args.inference_mode,
                        lora_dropout = args.lora_dropout,
                        bias = args.bias,
                        pretrained_model_name = args.pretrain_model,
                        remove_top_layers = args.remove_top_layers)

    else:
        # firectly finetune esm model 
        logger.info('Using FineTuneESM model')
        model = FineTuneESM( num_layers=args.num_layers, 
                            pretrained_model_name=args.pretrain_model, 
                            remove_top_layers=args.remove_top_layers)

    logger.info(print_trainable_parameters(model))
    logger.info(model)

    logger.info("**********************************")
    logger.info("Trainbel parameters:")
    total_trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info("%s, %s", name, str(param.data.shape))
            total_trainable_params += np.prod(param.data.shape)
    logger.info("**********************************")
    logger.info("Total trainable parameters: %d", total_trainable_params)

    # optimizer and lr policy
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
            lr=args.lr,
            weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
            lr=args.lr,
            weight_decay=args.weight_decay)
    else:
        print(f'optimizer {args.optimizer} is not implemented.')

    if args.lr_schedule == None:
        lr_schedule = None
    elif args.lr_schedule == 'cosine':
        lr_schedule = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=args.min_learning_rate)
    elif args.lr_schedule == 'exponential':
        lr_schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    else:
        print(f'lr_schedule {args.lr_schedule} is not implemented.')

    if args.warmup_period > 0:
        warmup_period = 100
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period)
    else:
        warmup_scheduler = None

    model.cuda()

    if args.resume is not None:
        if args.local_rank == 0:
            model_state, _ = load_model_checkpoint(args.resume)
            model.load_state_dict(model_state)

    scaler = torch.cuda.amp.GradScaler(
        init_scale=args.static_loss_scale,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=100 if args.dynamic_loss_scale else 1000000000,
        enabled=args.amp,
    )

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.gpu],
                output_device=args.gpu,
                find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(
                model, output_device=0, find_unused_parameters=True)
    else:
        model.cuda()

    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    if lr_schedule is not None and start_epoch > 0:
        lr_schedule.step(start_epoch)

    if args.local_rank == 0:
        logger.info('Scheduled epochs: {}'.format(args.epochs))

    gradient_accumulation_steps = args.gradient_accumulation_steps

    train_loop(model,
               optimizer,
               lr_schedule,
               scaler,
               gradient_accumulation_steps,
               train_loader,
               valid_loader,
               use_amp=args.amp,
               logger=logger,
               start_epoch=start_epoch,
               end_epoch=args.epochs,
               early_stopping_patience=args.early_stopping_patience,
               skip_training=args.evaluate,
               skip_validation=args.training_only,
               save_checkpoints=True,
               output_dir=args.output_dir,
               log_wandb=True,
               log_interval=args.log_interval,
               warmup_scheduler=warmup_scheduler,
               warmup_period=warmup_period)
    logger.info('Training ended, start testing...')

    model_state, _ = load_model_checkpoint(os.path.join(args.output_dir, 'model_best.pth.tar'))
    model.load_state_dict(model_state)
    model = model.cuda().eval()

    test_metrics = evaluate(model, test_loader, args.amp, logger, args.log_interval)
    logger.info('Test: %s' % (test_metrics))
    logger.info('Experiment ended.')

def train(model,
          loader,
          optimizer,
          scaler,
          gradient_accumulation_steps,
          use_amp,
          epoch,
          logger,
          log_interval=1,
          warmup_scheduler=None,
          warmup_period=0,
          lr_scheduler=None):
    batch_time_m = AverageMeter('Time', ':6.3f')
    data_time_m = AverageMeter('Data', ':6.3f')
    losses_m = AverageMeter('Loss', ':.4e')

    step = get_train_step(model, optimizer, scaler,
                          gradient_accumulation_steps, use_amp)

    model.train()
    optimizer.zero_grad()
    steps_per_epoch = len(loader)
    end = time.time()
    for idx, batch in enumerate(loader):
        # Add batch to GPU
        batch = {key: val.cuda() for key, val in batch.items()}
        data_time = time.time() - end
        optimizer_step = ((idx + 1) % gradient_accumulation_steps) == 0
        loss = step(batch, optimizer_step)
        batch_size = batch['labels'].shape[0]

        it_time = time.time() - end
        batch_time_m.update(it_time)
        data_time_m.update(data_time)
        losses_m.update(loss.item(), batch_size)

        if lr_scheduler is not None:
            if warmup_scheduler is not None:
                with warmup_scheduler.dampening():
                    if warmup_scheduler.last_step + 1 >= warmup_period:
                        lr_scheduler.step()
        end = time.time()
        if (idx % log_interval == 0) or (idx == steps_per_epoch - 1):
            if not torch.distributed.is_initialized(
            ) or torch.distributed.get_rank() == 0:
                learning_rate = optimizer.param_groups[0]['lr']
                log_name = 'Train-log'
                logger.info(
                    '{0}: [epoch:{1:>2d}] [{2:>2d}/{3}] '
                    'DataTime: {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'BatchTime: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f}) '
                    'lr: {lr:>4.6f} '.format(log_name,
                                             epoch + 1,
                                             idx,
                                             steps_per_epoch,
                                             data_time=data_time_m,
                                             batch_time=batch_time_m,
                                             loss=losses_m,
                                             lr=learning_rate))
    return OrderedDict([('loss', losses_m.avg)])


def evaluate(model, loader, use_amp, logger, log_interval=10):
    batch_time_m = AverageMeter('Time', ':6.3f')
    data_time_m = AverageMeter('Data', ':6.3f')
    losses_m = AverageMeter('Loss', ':.4e')

    model.eval()
    steps_per_epoch = len(loader)
    end = time.time()
    # Variables to gather full output
    true_labels= []
    pred_labels = []
    for idx, batch in enumerate(loader):
        batch = {key: val.cuda() for key, val in batch.items()}
        labels = batch['labels']
        labels = labels.float()
        data_time = time.time() - end
        with torch.no_grad(), autocast(enabled=use_amp):
            outputs = model(**batch)
            loss = outputs[0]
            logits = outputs[1]
            preds = torch.sigmoid(logits)
            preds = preds.detach().cpu()

        torch.cuda.synchronize()

        preds = preds.numpy()
        labels = labels.to('cpu').numpy()
        true_labels.append(labels)
        pred_labels.append(preds)

        batch_size = labels.shape[0]
        data_time = time.time() - end
        it_time = time.time() - end
        end = time.time()

        batch_time_m.update(it_time)
        data_time_m.update(data_time)
        losses_m.update(loss.item(), batch_size)
        if (idx % log_interval == 0) or (idx == steps_per_epoch - 1):
            if not torch.distributed.is_initialized(
            ) or torch.distributed.get_rank() == 0:
                logger_name = 'Test-log'
                logger.info(
                    '{0}: [{1:>2d}/{2}] '
                    'DataTime: {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f}) '.format(
                        logger_name,
                        idx,
                        steps_per_epoch,
                        data_time=data_time_m,
                        batch_time=batch_time_m,
                        loss=losses_m))
    # Flatten outputs
    true_labels = np.concatenate(true_labels, axis=0)
    pred_labels = np.concatenate(pred_labels, axis=0)
    # avg_auc
    metric_dict = compute_metrics(true_labels, pred_labels)
    # {'acc': acc, 'mcc': mcc, 'roc_auc': roc_auc, 'aupr': aupr, 'f1':f1, 'spec': specificity, 'sens': sensitivity, 'p': precision, 'r': recall}
    metrics = OrderedDict([('loss', losses_m.avg), 
                           ('acc', metric_dict['acc']), 
                           ('mcc', metric_dict['mcc']), 
                           ('auc', metric_dict['roc_auc']), 
                           ('aupr', metric_dict['aupr']), 
                           ('f1', metric_dict['f1']),
                           ('spec', metric_dict['spec']),
                           ('sens', metric_dict['sens']),
                           ('p', metric_dict['p']),
                           ('r', metric_dict['r'])])
    return metrics

def train_loop(model,
               optimizer,
               lr_scheduler,
               scaler,
               gradient_accumulation_steps,
               train_loader,
               val_loader,
               use_amp,
               logger,
               start_epoch=0,
               end_epoch=0,
               early_stopping_patience=-1,
               skip_training=False,
               skip_validation=True,
               save_checkpoints=True,
               output_dir='./',
               log_wandb=True,
               log_interval=10,
               warmup_scheduler=None,
               warmup_period=0):
    if early_stopping_patience > 0:
        epochs_since_improvement = 0

    best_metric = 0
    logger.info('Evaluate validation set before start training')
    eval_metrics = evaluate(model, val_loader, use_amp, logger, log_interval)
    logger.info('Evaluation: %s' % (eval_metrics))
    logger.info(f'RUNNING EPOCHS FROM {start_epoch} TO {end_epoch}')
    for epoch in range(start_epoch, end_epoch):
        is_best = False
        if not skip_training:
            train_metrics = train(model, train_loader, optimizer, scaler,
                                  gradient_accumulation_steps, use_amp, epoch,
                                  logger, log_interval, warmup_scheduler, warmup_period=warmup_period, lr_scheduler=lr_scheduler)

            logger.info('[Epoch %d] training: %s' % (epoch + 1, train_metrics))

        if not skip_validation:
            eval_metrics = evaluate(model, val_loader, use_amp, logger,
                                    log_interval)

            logger.info('[Epoch %d] Evaluation: %s' %
                        (epoch + 1, eval_metrics))

        if eval_metrics['acc'] > best_metric:
            is_best = True
            best_metric = eval_metrics['acc']
        if log_wandb and (not torch.distributed.is_initialized()
                          or torch.distributed.get_rank() == 0):
            update_summary(epoch,
                           train_metrics,
                           eval_metrics,
                           os.path.join(output_dir, 'summary.csv'),
                           write_header=best_metric is None,
                           log_wandb=log_wandb)

        if save_checkpoints and (not torch.distributed.is_initialized()
                                 or torch.distributed.get_rank() == 0):
            checkpoint_state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_metric': eval_metrics['acc'],
                'optimizer': optimizer.state_dict(),
            }
            logger.info('[*] Saving model epoch %d...' % (epoch + 1))
            save_checkpoint(checkpoint_state,
                            epoch,
                            is_best,
                            checkpoint_dir=output_dir)

        if early_stopping_patience > 0:
            if not is_best:
                epochs_since_improvement += 1
            else:
                epochs_since_improvement = 0
            if epochs_since_improvement >= early_stopping_patience:
                break

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


if __name__ == '__main__':
    args, args_text = _parse_args()
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    model_short = args.pretrain_model.split('_')[-2]
    if args.is_peft:
        task_name = 'esm_lora' + '_' + model_short + '_' + f'lr{args.learning_rate}' + '_' + f'rmt{args.remove_top_layers}' + '_' + f'r{args.num_lora_r}a{args.num_lora_alpha}l{args.num_end_lora_layers}d{args.lora_dropout}'
    else:
        task_name = 'esm_finetune' + '_' + model_short + '_' + f'lr{args.learning_rate}' + '_' + f'rmt{args.remove_top_layers}' + '_' + f'f{args.num_layers}'
    
    args.learning_rate = float(args.learning_rate)
    args.lora_dropout = float(args.lora_dropout)
    args.output_dir = os.path.join(args.output_dir, task_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)

    logger = logging.getLogger('')
    filehandler = logging.FileHandler(
        os.path.join(args.output_dir, 'summary.log'))
    streamhandler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    main(args)