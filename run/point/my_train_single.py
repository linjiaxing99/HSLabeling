import torch
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict
import sys
# print(sys.path)
sys.path.append('/home/isalab206/LJX/HSLabeling-master')
# from options import *
from utils import *
from dataset import *

print("PyTorch Version: ", torch.__version__)
print('cuda', torch.version.cuda)
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = torch.device("cuda:0")
print('Device:', device)


def main():
    opt = Point_Options().parse()
    log_path, checkpoint_path, _, _, _ = create_save_path(opt)
    train_txt_path, val_txt_path, _ = create_data_path(opt)

    # Define logger
    logger, tensorboard_log_dir = create_logger(log_path)
    logger.info(opt)
    # Define Transformation
    train_transform = transforms.Compose([
        trans.RandomHorizontalFlip(),
        trans.RandomVerticleFlip(),
        trans.RandomRotate90(),
    ])

    # load train and val dataset
    train_dataset = Dataset_point(opt, train_txt_path, flag='train', transform=train_transform)
    val_dataset = Dataset_point(opt, val_txt_path, flag='val', transform=None)

    # Create training and validation dataloaders
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.num_workers, pin_memory=opt.pin, drop_last=True)

    model = Seg_Net(opt)
    if opt.resume:
        checkpoint = torch.load(checkpoint_path+'/model_best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        print('resume success')

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=opt.base_lr, amsgrad=True)
    best_metric = 1e16

    for epoch in range(opt.num_epochs):
        time_start = time.time()

        train(opt, epoch, model, train_loader, optimizer, logger)
        metric = validate(opt, model, val_dataset, logger)

        logger.info('best_val_metric:%.4f current_val_metric:%.4f' % (best_metric, metric))
        if (int(epoch) % 10 == 9):
            torch.save({'state_dict': model.state_dict()},
                       os.path.join(checkpoint_path, 'model_latest_' + str(epoch) +'.pth'))
        if metric < best_metric:
            logger.info('epoch:%d Save to model_best' % (epoch))
            torch.save({'state_dict': model.state_dict()},
                       os.path.join(checkpoint_path, 'model_best.pth'))
            best_metric = metric

        end_time = time.time()
        time_cost = end_time - time_start
        logger.info("Epoch %d Time %d ----------------------" % (epoch, time_cost))
        logger.info('\n')


def train(opt, epoch, model, loader, optimizer, logger):
    model.train()

    loss_m = AverageMeter()
    seg_loss_m = AverageMeter()

    if len(loader) >= 2:
        last_idx = len(loader) - 1
    else:
        last_idx = 1

    for batch_idx, batch in enumerate(loader):
        step = epoch * len(loader) + batch_idx
        adjust_learning_rate(opt.base_lr, optimizer, step, len(loader), num_epochs=30)
        lr = get_lr(optimizer)

        last_batch = batch_idx == last_idx
        img, label, cls_label, _ = batch
        input, label, cls_label = img.cuda(non_blocking=True), label.cuda(non_blocking=True), \
                                  cls_label.cuda(non_blocking=True)
        output, _ = model(input)
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        # class_weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 5.0]).to(device)
        # class_weight = torch.tensor([1.0, 1.5, 1.5, 1.0, 1.5, 1.0, 1.0]).to(device)
        # criterion = nn.CrossEntropyLoss(weight=class_weight, ignore_index=255)
        seg_loss = criterion(output, label)
        loss = seg_loss

        seg_loss_m.update(seg_loss.item(), input.size(0))
        loss_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        if len(loader) >= 3:
            log_interval = len(loader) // 3
        else:
            log_interval = 1
        if last_batch or batch_idx % log_interval == 0:
            logger.info('Train:{} [{:>4d}/{} ({:>3.0f}%)] '
                    'Loss:({loss.avg:>6.4f}) '
                    'segloss:({seg_loss.avg:>6.4f}) '
                    'LR:{lr:.3e} '.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=loss_m,
                        seg_loss=seg_loss_m,
                        lr=lr))

    return OrderedDict([('loss', loss_m.avg)])


def validate(opt, model, val_dataset, logger):
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=opt.pin)
    model.eval()
    seg_loss_m = AverageMeter()

    for batch_idx, batch in enumerate(val_loader):
        img, label, cls_label, _ = batch
        with torch.no_grad():
            input, label, cls_label = img.cuda(non_blocking=True), label.cuda(non_blocking=True), \
                                      cls_label.cuda(non_blocking=True)
            out, _ = model.forward(input)
            criterion = nn.CrossEntropyLoss(ignore_index=255)
            seg_loss = criterion(out, label)
            seg_loss_m.update(seg_loss.item(), input.size(0))

    logger.info('VAL:''segloss:({seg_loss.avg:>6.4f}) '.format(seg_loss=seg_loss_m,))

    return seg_loss_m.avg


if __name__ == '__main__':
   main()