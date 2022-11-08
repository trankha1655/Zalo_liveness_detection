import os, sys
import torch
import torch.nn as nn
import time
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils import data
import torch.distributed as dist
# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_train, build_dataset_test


from utils.utils import setup_seed, netParams
from utils.plot_log import draw_log
from utils.record_log import record_log
from utils.earlyStopping import EarlyStopping
from utils.scheduler.lr_scheduler import PolyLR

from utils.metrics import Classify_Metrics
from tools.train_val_tools import train, val
import warnings

warnings.filterwarnings('ignore')

sys.setrecursionlimit(1000000)  # solve problem 'maximum recursion depth exceeded'
GLOBAL_SEED = 88

def parse_args():
    parser = ArgumentParser(description='Liveness Detection with PyTorch')
    # model and dataset
    parser.add_argument('--model', type=str, default="MobileNetv2",
                            choices=['Inception_Resnetv2', 'MobileNetv2', 'MobileNetv3_small', 'MobileNetv3_large'], 
                            help="model name")
    
    parser.add_argument('--test_mode', type=int, default=False, help="int number is a lenght dataset to testing speed")
    parser.add_argument('--root', type=str, default="", help="path of datasets")
    parser.add_argument('--base_size', type=int, default=720, help="input size of image")
    parser.add_argument('--crop_size', type=int, default=299, help="crop size of image")
    parser.add_argument('--num_workers', type=int, default=1, help=" the number of parallel threads")
    # training hyper params
    parser.add_argument('--max_epochs', type=int, default=300, help="the number of epochs: 300 for train")
    parser.add_argument('--batch_size', type=int, default=4, help="the batch size is set to 16 GPU")
    parser.add_argument('--val_epochs', type=int, default=10, help="the number of epochs: 100 for val set")
    parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--optim', type=str.lower, default='adam', choices=['sgd', 'adam', 'adamw'],
                        help="select optimizer")
    parser.add_argument('--predict_type', default="validation", choices=["validation", "predict"],
                        help="Defalut use validation type")
    
    # cuda setting
    parser.add_argument('--cuda', type=bool, default=True, help="running on CPU or GPU")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--gpus_id', type=str, default="1", help="default GPU devices 1")
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help="use this file to load last checkpoint for continuing training")

    parser.add_argument('--weight', type =str, help="path direct to weight")                    
    parser.add_argument('--savedir', default="./checkpoint/", help="directory to save the model snapshot")
    parser.add_argument('--logFile', default="log.txt", help="storing the training and validation logs")
    args = parser.parse_args()

    return args


def main(args):
    """
    args:
       args: global arguments
    """
    
    setup_seed(GLOBAL_SEED)
    
    traindataset = build_dataset_train(args.root, 
                                        args.base_size, 
                                        args.crop_size,
                                        args.test_mode
                                        )
    # load the test set, if want set cityscapes test dataset change none_gt=False
    testdataset = build_dataset_test(args.root, 
                                     args.crop_size,
                                     True,
                                     args.test_mode
                                     )
    # Default size image of MobileNet model
    if 'Mobile' in args.model:
        args.crop_size = 224

    model = build_model(args.model, 
                        2,
                        args.weight)

    # define loss function, respectively
    criterion = nn.CrossEntropyLoss()
    
    # move model and criterion on cuda
    if args.cuda:
        
        device = torch.device("cuda", args.local_rank)

        trainLoader = data.DataLoader(traindataset, batch_size=args.batch_size, 
                 num_workers=args.num_workers)

        
        model = torch.nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
        testLoader = data.DataLoader(testdataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.num_workers)

        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    # define optimization strategy
    # parameters = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
    #             {'params': model.get_10x_lr_params(), 'lr': args.lr}]
    parameters = model.parameters()

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=False)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(parameters,lr=args.lr, weight_decay=5e-4)
    elif args.optim == 'adamw':
        optimizer = torch.optim.AdamW(parameters,lr=args.lr, weight_decay=5e-4)

    # initial log file val output save
    args.savedir = (args.savedir + '/' + args.model + '/')
    if not os.path.exists(args.savedir) and args.local_rank == 0:
        os.makedirs(args.savedir)

    # save_seg_dir
    args.save_seg_dir = args.savedir
    if not os.path.exists(args.save_seg_dir) and args.local_rank == 0:
        os.makedirs(args.save_seg_dir)

    
    recorder = record_log(args)
    if args.resume == None and args.local_rank == 0:
        recorder.record_args(datas, str(netParams(model) / 1e6) + ' M', GLOBAL_SEED)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=300)
    start_epoch = 1
    if args.local_rank == 0:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"
              ">>>>>>>>>>>  beginning training   >>>>>>>>>>>\n"
              ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    

    epoch_list = []
    lossTr_list = []
    Acc_list = []
    lossVal_list = []
    accuracy = 0
    Best_Acc = 0
    last_model_file_name = args.savedir + '/last_model.pth'
    # continue training
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optimizer'])
            check_list = [i for i in checkpoint['model'].items()]
            # Read weights with multiple cards, and continue training with a single card this time
            if 'module.' in check_list[0][0]:
                new_stat_dict = {}
                for k, v in checkpoint['model'].items():
                    new_stat_dict[k[:]] = v
                model.load_state_dict(new_stat_dict, strict=True)
            # Read the training weight of a single card, and continue training with a single card this time
            else:
                check_list = [i for i, v in model.state_dict().items()]
                if 'module.' in check_list[0]:
                    new_stat_dict = {}
                    for k, v in checkpoint['model'].items():
                        new_stat_dict['module.'+ k[:]] = v

                    model.load_state_dict(new_stat_dict, strict=True)
                else:

                    model.load_state_dict(checkpoint['model'])
            if args.local_rank == 0:
                print("loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if args.local_rank == 0:
                print("no checkpoint found at '{}'".format(args.resume))

        logger, lines = recorder.resume_logfile()
        for index, line in enumerate(lines):
            lossTr_list.append(float(line.strip().split()[2]))
            if len(line.strip().split()) != 3:
                epoch_list.append(int(line.strip().split()[0]))
                lossVal_list.append(float(line.strip().split()[3]))
                Acc_list.append(float(line.strip().split()[4]))

                if Best_Acc < Acc_list[-1]:
                    Best_Acc = Acc_list[-1]
            
            if index == start_epoch:
                break
    else:
        logger = recorder.initial_logfile()
        logger.flush()
    
        

    for epoch in range(start_epoch, args.max_epochs + 1):
        start_time = time.time()
        # training
        train_start = time.time()
        #WRITE LOG FILE BEFORE TRAINING: PREVENT BUG AFTER TRAINED
        with open(last_model_file_name[:-4]+'.txt', 'w') as f:
                f.write('Last model from epoch: ' + epoch)

        lossTr, lr = train(args= args, 
                            train_loader= trainLoader, 
                            model= model, 
                            criterion= criterion, 
                            optimizer= optimizer, 
                            epoch= epoch, 
                            device= device)
        if args.local_rank == 0:
            lossTr_list.append(lossTr)

        train_end = time.time()
        train_per_epoch_seconds = train_end - train_start
        validation_per_epoch_seconds = 60  # init validation time
        # validation if mode==validation, predict with label; elif mode==predict, predict without label.

        if epoch % args.val_epochs == 0 or epoch == 1 or args.max_epochs - 20 < epoch <= args.max_epochs:
            validation_start = time.time()

            metrics = Classify_Metrics(args, 2)
            loss, accuracy, precision, recall, f1 = val(
                                                         val_loader= testLoader,
                                                         model= model, 
                                                         criterion= criterion, 
                                                         device= device, 
                                                         metrics= metrics)
        
            torch.cuda.empty_cache()
            # record trainVal information
            if args.local_rank == 0:
                epoch_list.append(epoch)
                Acc_list.append(accuracy)
                lossVal_list.append(loss)
                
                recorder.record_trainVal_log(logger = logger, 
                                            epoch = epoch, 
                                            lr= lr, 
                                            lossTr= lossTr, 
                                            val_loss= loss, 
                                            Acc= accuracy, 
                                            Pre= precision, 
                                            Rec= recall, 
                                            F1= f1)

                torch.cuda.empty_cache()
                validation_end = time.time()
                validation_per_epoch_seconds = validation_end - validation_start
        else:
            # record train information
            if args.local_rank == 0:
                recorder.record_train_log(logger, epoch, lr, lossTr)

            # # Update lr_scheduler. In pytorch 1.1.0 and later, should call 'optimizer.step()' before 'lr_scheduler.step()'
            # lr_scheduler.step()
        if args.local_rank == 0:
            # draw log fig
            draw_log(args, epoch, epoch_list, lossTr_list, Acc_list, lossVal_list)
            
            # save the model
            model_file_name = args.savedir + '/best_model.pth'
            last_model_file_name = args.savedir + '/last_model.pth'
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            if accuracy > Best_Acc:
                Best_Acc = accuracy
                torch.save(state, model_file_name)
                recorder.record_best_epoch(epoch= epoch,
                                            Acc = accuracy, 
                                            Pre = precision,
                                            Rec = recall, 
                                            F1 = f1) 
            #Save last epoch. 
            torch.save(state, last_model_file_name)
            

            # early_stopping monitor
            early_stopping.monitor(monitor=accuracy)
            if early_stopping.early_stop:
                print("Early stopping and Save checkpoint")
                # if not os.path.exists(last_model_file_name):
                #     torch.save(state, last_model_file_name)
                torch.cuda.empty_cache()  # empty_cache
                loss, accuracy, precision, recall, f1 = val(
                                                        val_loader= testLoader,
                                                        model= model, 
                                                        criterion= criterion, 
                                                        device= device, 
                                                        metrics= metrics)
                print("Epoch {}  lr= {:.6f}  Train Loss={:.4f}  Val Loss={:.4f}  Acc={:.4f}  Precision={:.4f}  Recall={:.4f} F1_Score={:.4f}\n"
                        .format(epoch, lr, lossTr, loss, accuracy, precision, recall, f1))
                break

            total_second = start_time + (args.max_epochs - epoch) * train_per_epoch_seconds + \
                           ((args.max_epochs - epoch) / args.val_epochs + 10) * validation_per_epoch_seconds + 43200
            print('Best Validation Accuracy:{}'.format(Best_Acc))
            print('Training deadline is: {}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_second))))



if __name__ == '__main__':
    start = time.time()
    args = parse_args()
    
    

    main(args)

    end = time.time()
    hour = 1.0 * (end - start) / 3600
    minute = (hour - int(hour)) * 60
    if args.local_rank == 0:
        print("training time: %d hour %d minutes" % (int(hour), int(minute)))
