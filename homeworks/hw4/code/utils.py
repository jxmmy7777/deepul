from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

def train(model, train_loader, optimizer, epoch, quiet, grad_clip=None,scheduler=None):
    model.train()

    if not quiet:
        pbar = tqdm(total=len(train_loader.dataset))
    losses = OrderedDict()
    for batch in train_loader:
        if len(batch) == 2:
            x, y = batch
            y = y.cuda()  # Move labels to GPU if present
            x = x.cuda()
            out = model.loss(x,y)
        else:
            x = batch[0]
            x = x.cuda()
            out = model.loss(x)
        optimizer.zero_grad()
        out['loss'].backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        desc = f'Epoch {epoch}'
        for k, v in out.items():
            if k not in losses:
                losses[k] = []
            losses[k].append(v.item())
            avg_loss = np.mean(losses[k][-50:])
            desc += f', {k} {avg_loss:.4f}'

        if not quiet:
            pbar.set_description(desc)
            pbar.update(x.shape[0])
    if not quiet:
        pbar.close()
    return losses


def eval_loss(model, data_loader, quiet):
    model.eval()
    total_losses = OrderedDict()
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 2:
                x, y = batch
                y = y.cuda()  # Move labels to GPU if present
                x = x.cuda()
                out = model.loss(x,y)
            else:
                x = batch[0]
                x = x.cuda()
                out = model.loss(x)
            for k, v in out.items():
                total_losses[k] = total_losses.get(k, 0) + v.item() * x.shape[0]

        desc = 'Test '
        for k in total_losses.keys():
            total_losses[k] /= len(data_loader.dataset)
            desc += f', {k} {total_losses[k]:.4f}'
        if not quiet:
            print(desc)
    return total_losses
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineDecayLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.cosine_decay_steps = total_steps - warmup_steps
        super(WarmupCosineDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_factor = self.last_epoch / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        # Cosine annealing
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / self.cosine_decay_steps))
        decayed_lr = self.min_lr + (self.base_lrs[0] - self.min_lr) * cosine_decay
        return [decayed_lr for _ in self.base_lrs]

def train_epochs(model, train_loader, test_loader, train_args, quiet=False, checkpoint=None):
    epochs, lr = train_args['epochs'], train_args['lr']
    grad_clip = train_args.get('grad_clip', None)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # if scheduler
    if "scheduler" in train_args:
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_decay_lr)
        # T_max = train_args["scheduler"]["Total_steps"] - train_args["scheduler"]["Warmup_steps"]
        # scheduler =  optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0)
        # scheduler = WarmupCosineDecayLR(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=0)
        # from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
        #https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
        # scheduler = CosineAnnealingWarmupRestarts(optimizer,
        #                                   first_cycle_steps=200,
        #                                   cycle_mult=1.0,
        #                                   max_lr=0.1,
        #                                   min_lr=0.001,
        #                                   warmup_steps=train_args["scheduler"]["Warmup_steps"],
        #                                   gamma=1.0)
        # from functools import partial
        # warmup_cosine_decay_lr_args = partial(warmup_cosine_decay_lr, warmup_steps=train_args["scheduler"]["Warmup_steps"], total_steps=train_args["scheduler"]["Total_steps"], initial_lr=train_args["lr"]) 
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_decay_lr)
        scheduler = WarmupCosineDecayScheduler(optimizer, warmup_steps=train_args["scheduler"]["Warmup_steps"], total_steps=train_args["scheduler"]["Total_steps"], initial_lr=train_args["lr"])
    else:
        scheduler = None

    train_losses, test_losses = OrderedDict(), OrderedDict()
    for epoch in range(epochs):
        model.train()
        train_loss = train(model, train_loader, optimizer, epoch, quiet, grad_clip, scheduler)
        test_loss = eval_loss(model, test_loader, quiet)
        # if scheduler is not None:
        #     scheduler.step()
        for k in train_loss.keys():
            if k not in train_losses:
                train_losses[k] = []
                test_losses[k] = []
            train_losses[k].extend(train_loss[k])
            test_losses[k].append(test_loss[k])
        
        if checkpoint is not None and (epoch+1) % 10 == 0:
            #save all information to checkpoint path
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses,
                'test_loss': test_losses
            }, checkpoint)
            print(f"Checkpoint saved at {checkpoint}")
    return train_losses, test_losses

def warmup_cosine_decay_lr(steps, warmup_steps=100, total_steps=10000, initial_lr=1e-3):
    if steps <= warmup_steps:
        # Linear warmup
        lr = initial_lr * (steps / warmup_steps)
    else:
        # Cosine decay
        decay_steps = steps - warmup_steps
        decay_total = total_steps - warmup_steps
        cosine_decay = 0.5 * (1 + np.cos(torch.pi * decay_steps / decay_total))
        lr = initial_lr * cosine_decay
    return lr


class WarmupCosineDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, initial_lr=1e-3, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.initial_lr = initial_lr
        self.verbose = verbose
        super(WarmupCosineDecayScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_steps:
            lr = self.initial_lr * self._step_count / self.warmup_steps
        else:
            decay_steps = self._step_count - self.warmup_steps
            decay_total = self.total_steps - self.warmup_steps
            cosine_decay = 0.5 * (1 + np.cos(np.pi * decay_steps / decay_total))
            lr = self.initial_lr * cosine_decay
        return [lr for _ in self.optimizer.param_groups]
    
class MLP(nn.Module):
    def __init__(self, input_shape, output_shape, hiddens=[]):
        super().__init__()

        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        if isinstance(output_shape, int):
            output_shape = (output_shape,)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hiddens = hiddens

        model = []
        prev_h = np.prod(input_shape)
        for h in hiddens + [np.prod(output_shape)]:
            model.append(nn.Linear(prev_h, h))
            model.append(nn.ReLU())
            prev_h = h
        model.pop()
        self.net = nn.Sequential(*model)

    def forward(self, x):
        b = x.shape[0]
        x = x.view(b, -1)
        return self.net(x).view(b, *self.output_shape)