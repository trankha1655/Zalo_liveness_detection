
# _*_ coding: utf-8 _*_
"""
Time:     2020/11/30 17:02
Author:   Ding Cheng(Deeachain)
File:     predict.py
Describe: Write during my study in Nanjing University of Information and Secience Technology
Github:   https://github.com/Deeachain
"""
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils import data
from argparse import ArgumentParser
from prettytable import PrettyTable
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test, build_dataset_mp4
#from builders.loss_builder import build_loss
from tools.train_val_tools import predict, val
from dataset.liveness import LivenessTestVideo
from utils.write_result import WriteResult
def main(args):
    """
     main function for testing
     param args: global arguments
     return: None
    """
    t = PrettyTable(['args_name', 'args_value'])
    for k in list(vars(args).keys()):
        t.add_row([k, vars(args)[k]])
    print(t.get_string(title="Predict Arguments"))

    # build the model
    model = build_model(args.model, 
                        2,
                        None)

    # load the test set
    if args.predict_type == 'validation':
        
        testdataset = build_dataset_test(args.root, 
                                        args.crop_size,
                                        gt=True)

    elif args.predict_type == 'predict':
        testdataset = build_dataset_test(args.root, 
                                        args.crop_size,
                                        gt=False)
    else:
        testdataset = build_dataset_mp4(args.root, args.crop_size, args.frame_num)
        args.batch_size = 1

    DataLoader = data.DataLoader(testdataset, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    if args.cuda:
        #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        model = model.cuda()
        device = 'cuda'
        cudnn.benchmark = True
        if not torch.cuda.is_available():
            raise Exception("no GPU found or wrong gpu id, please run without --cuda")

    if not os.path.exists(args.save_seg_dir):
        os.makedirs(args.save_seg_dir)
    
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            checkpoint = torch.load(args.checkpoint)['model']
            check_list = [i for i in checkpoint.items()]
            # Read weights with multiple cards, and continue training with a single card this time
            if 'module.' in check_list[0][0]:  # 读取使用多卡训练权重,并且此次使用单卡预测
                new_stat_dict = {}
                for k, v in checkpoint.items():
                    new_stat_dict[k[7:]] = v
                model.load_state_dict(new_stat_dict, strict=True)
            # Read the training weight of a single card, and continue training with a single card this time
            else:
                model.load_state_dict(checkpoint)
            print('Loaded weight from ', args.checkpoint)
        else:
            print("no checkpoint found at '{}'".format(args.checkpoint))
            raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))
    

    # define loss function
    criterion = torch.nn.CrossEntropyLoss()
    write_results = WriteResult(args)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"
          ">>>>>>>>>>>  beginning testing   >>>>>>>>>>>>\n"
          ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    
    if 'val' in args.predict_type:
        pass
    else:

        predict(args, DataLoader, model, device, write_results)
        write_results.save_df()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="MobileNetv2", help="model name")
    parser.add_argument('--root', type=str, default="", help="path of datasets")
    parser.add_argument('--predict_type', default="validation", choices=["validation", "predict", "predict_mp4"],
                        help="Defalut use validation type")
    parser.add_argument('--num_workers', type=int, default=1, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=16,
                        help=" the batch_size is set to 1 when evaluating or testing NOTES:image size should fixed!")
    parser.add_argument('--crop_size', type=int, default=224, help="crop size of image")
    parser.add_argument('--frame_num', type=int, default=5, help="get frame after number of frames")
    parser.add_argument('--checkpoint', type=str, default='best_model.pth',
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--save_seg_dir', type=str, default="./outputs/",
                        help="saving path of prediction result")
    parser.add_argument('--loss', type=str, default="CrossEntropyLoss2d",
                        choices=['CrossEntropyLoss2d', 'ProbOhemCrossEntropy2d', 'CrossEntropyLoss2dLabelSmooth',
                                 'LovaszSoftmax', 'FocalLoss2d'], help="choice loss for train or val in list")
    parser.add_argument('--cuda', default=True, help="run on CPU or GPU")
    parser.add_argument("--gpus", default="1", type=str, help="gpu ids (default: 0)")
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    save_dirname = args.model

    args.save_seg_dir = os.path.join(args.save_seg_dir,  save_dirname)

    

    main(args)
