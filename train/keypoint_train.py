import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import time
import copy
from tqdm import tqdm


def rmse_score(true, pred):
    return torch.sqrt(torch.mean((true - pred) ** 2))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


WARMUP_EPOCH = 1
def train_one_epoch(model, opt, dataloader, loss_fn, device, epoch, max_batch_size):
    model.train()
    losses = {"mse": 0}

    lr_scheduler = None
    if epoch < WARMUP_EPOCH:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(dataloader) - 1)
        lr_scheduler = warmup_lr_scheduler(opt, warmup_iters, warmup_factor)

    for imgs, keypoints in tqdm(dataloader):
        imgs = imgs.float().to(device)[: max_batch_size]
        keypoints = torch.from_numpy(keypoints).float().to(device)[: max_batch_size]
        keypoints = torch.cat([keypoints, keypoints, keypoints])

        opt.zero_grad()
        pred = model(imgs)
        pred = torch.cat(pred)
        if loss_fn is not None:
            loss = loss_fn(pred, keypoints)
        else:
            loss = F.mse_loss(pred, keypoints)
        loss.backward()
        opt.step()

        losses["mse"] += loss.item()

        if lr_scheduler is not None:
            lr_scheduler.step()
    losses["mse"] /= len(dataloader)
    return losses


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device, max_batch_size):
    model.train()
    losses = {"mse": 0}

    for imgs, keypoints in tqdm(dataloader):
        imgs = imgs.float().to(device)[: max_batch_size]
        keypoints = torch.from_numpy(keypoints).float().to(device)[: max_batch_size]
        keypoints = torch.cat([keypoints, keypoints, keypoints])

        pred = model(imgs)
        pred = torch.cat(pred)
        if loss_fn is not None:
            loss = loss_fn(pred, keypoints)
        else:
            loss = F.mse_loss(pred, keypoints)

        losses["mse"] += loss.item()

    losses["mse"] /= len(dataloader)
    return losses


SAVE_INTERVAL = 5
def train_model(model, opt, scheduler, loss_fn, device, save_dir, start_epoch, end_epoch,
                train_dataloader, valid_dataloader, max_batch_size):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100.

    train_losses = {"mse": []}
    valid_losses = {"mse": []}

    #cudnn.benchmark = True
    for epoch in range(start_epoch, end_epoch + 1):
        print("\n" + "=" * 40)
        print("Epoch {}/{}".format(epoch, end_epoch))

        train_loss = train_one_epoch(model, opt, train_dataloader, loss_fn, device, epoch, max_batch_size)
        print("\nTrain Loss")
        print("\t mse: {:.6f}".format(train_loss["mse"]))
        for key in train_losses:
            train_losses[key].append(train_loss[key])

        if scheduler is not None:
            scheduler.step()

        valid_loss = evaluate(model, valid_dataloader, loss_fn, device, max_batch_size)
        print("\nValid Loss")
        print("\t mse: {:.6f}".format(valid_loss["mse"]))
        for key in valid_losses:
            valid_losses[key].append(valid_loss[key])

        if valid_loss["mse"] < best_loss:
            best_loss = valid_loss["mse"]
            best_model_wts = copy.deepcopy(model.state_dict())
            if epoch % SAVE_INTERVAL == 0:
                torch.save(best_model_wts, save_dir)

    time_elapsed = time.time() - since
    print("\nTraining complete in {}m {:0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val loss: {:.4f}".format(best_loss))

    torch.save(best_model_wts, save_dir)
    return train_losses, valid_losses
