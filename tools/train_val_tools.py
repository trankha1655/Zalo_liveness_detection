from tqdm import tqdm
from utils.scheduler.lr_scheduler import PolyLR
import torch
import numpy as np
def train(args, train_loader, model, criterion, optimizer, epoch, device):
    """
    args:
       train_loader: loaded for training dataset
       model       : model
       criterion   : loss function
       optimizer   : optimization algorithm, such as ADAM or SGD
       epoch       : epoch number
       device      : cuda
    return: average loss, lr
    """

    model.train()
    epoch_loss = []
    lr = optimizer.param_groups[0]['lr']
    total_batches = len(train_loader)
    pbar = tqdm(iterable=enumerate(train_loader),
                total=total_batches,
                desc='Epoch {}/{}'.format(epoch, args.max_epochs)
                )
    
    for iteration, batch in pbar:
        
        max_iter = args.max_epochs * total_batches
        cur_iter = (epoch - 1) * total_batches + iteration
        scheduler = PolyLR(optimizer,
                           max_iter=max_iter,
                           cur_iter=cur_iter,
                           power=0.9)
        lr = optimizer.param_groups[0]['lr']

        scheduler.step()
        optimizer.zero_grad()

        images, labels, _, _ = batch
        #print(images.shape)
        images = images.to(device).float()
        #print(images.shape)
        labels = labels.to(device).float()
        output = model(images)

        #print(output.shape, labels.shape)

        loss = 0
        if type(output) is tuple:  # output = (main_loss, aux_loss1, axu_loss2***)
            length = len(output)
            for index, out in enumerate(output):
                loss_record = criterion(out, labels)
                if index == 0:
                    loss_record *= 0.6
                else:
                    loss_record *= 0.4 / (length - 1)
                loss += loss_record
        else:
            loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
        pbar.set_postfix({'loss': loss.item()} )

        
        

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    # torch.cuda.empty_cache()
    return average_epoch_loss_train, lr


def val(args, val_loader, model, criterion, optimizer, epoch, device, metrics):
    """
    args:
       train_loader: loaded for training dataset
       model       : model
       criterion   : loss function
       optimizer   : optimization algorithm, such as ADAM or SGD
       epoch       : epoch number
       device      : cuda
    return: average loss, lr
    """

    model.eval()
    epoch_loss = []
    # lr = optimizer.param_groups[0]['lr']
    total_batches = len(val_loader)
    pbar = tqdm(iterable=enumerate(val_loader),
                total=total_batches,
                desc='Predicting')  
        
    for iteration, batch in pbar:
        
        with torch.no_grad():

            images, labels, _, _ = batch
            #print(images.shape)
            images = images.to(device).float()
            #print(images.shape)
            labels = labels.to(device).float()
            
            output = model(images)

            #print(output.shape, labels.shape)

            loss = 0
            loss = criterion(output, labels)
            
            
            epoch_loss.append(loss.item())
            
        gt = torch.argmax(labels, 1).long()
        gt = np.asarray(gt.cpu(), dtype=np.uint8)
        output = torch.argmax(output, 1).long()
        output = np.asarray(output.cpu(), dtype=np.uint8)
        #get metrics of a batch then 
        acc = metrics.addBatch(target = gt, predict= output)

        pbar.set_postfix({'loss': loss.item(), 'accuracy': acc})

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    accuracy, precision, recall, f1 = metrics.get_metrics()
    

    # torch.cuda.empty_cache()
    return average_epoch_loss_train, accuracy, precision, recall, f1

def predict(args, val_loader, model, device):
    """
    args:
       train_loader: loaded for training dataset
       model       : model
       device      : cuda
    return: {'names': names, 'labels': predict_label}
    """

    model.eval()
    epoch_loss = []
    # lr = optimizer.param_groups[0]['lr']
    total_batches = len(train_loader)
    pbar = tqdm(iterable=enumerate(val_loader),
                total=total_batches,
                desc='Predicting')
                
    for iteration, batch in pbar:
        
        outputs = predict_batch(batch, model, device)
    


    
    # torch.cuda.empty_cache()
    return average_epoch_loss_train


def predict_batch(batch, model, device):
    """
    args:
       batch       : images, size, names    |if type predict is video, batch is number image is cutted from video      
       model       : model                  while normal predict just is a batch.
       device      : cuda
    return: {'names': names, 'labels': predict_label}:  B x str or 1 x str, B x 2
                                                       (B names of Batch of images or one name of one videos)
    """

    model.eval()
    
    # lr = optimizer.param_groups[0]['lr']
    
                
    
    with torch.no_grad():

        images, _, names = batch
        #print(images.shape)
        images = images.to(device).float()
        
        
        
        output = model(images)

        if len(names) > 1:
            output = torch.argmax(output, 1).long()
            output = np.asarray(output.cpu(), dtype=np.uint8) # (B x 2)
        else:
            output = torch.mean(output, 0)
            output = np.asarray(output.cpu(), dtype=np.uint8) 
    # torch.cuda.empty_cache()
    return {'names': names, 'labels': output}


