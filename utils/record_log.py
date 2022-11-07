from prettytable import PrettyTable


class record_log():
    def __init__(self, args):
        self.args = args

    def record_args(self, total_paramters, GLOBAL_SEED):
        with open(self.args.savedir + 'args.txt', 'w') as f:
            t = PrettyTable(['args_name', 'args_value'])
            for k in list(vars(self.args).keys()):
                t.add_row([k, vars(self.args)[k]])
            t.add_row(['seed', GLOBAL_SEED])
            t.add_row(['parameters', total_paramters])

            print(t.get_string(title="Train Arguments"))
            f.write(str(t))

    def record_best_epoch(self, epoch, Acc, Pre, Rec, F1):
        with open(self.args.savedir + 'args.txt', 'a+') as f:
            f.write('\nBest Validation Epoch {} Best_Acc is {:.4f} Precision is {:.4f} Recall is {:.4f} F1 is {:.4f} '.format(
                epoch, Acc, Pre, Rec, F1))

    def initial_logfile(self):
        logFileLoc = self.args.savedir + self.args.logFile
        logger = open(logFileLoc, 'w')
        logger.write(("{}\t{}\t\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
            'Epoch', '   lr', 'Loss(Tr)', 'Loss(Val)', 'Accuracy(Val)', 
            'Precicion(Val)',  'Recall(Val)', 'F1_Score(Val)')))
        return logger

    def resume_logfile(self):
        logFileLoc = self.args.savedir + self.args.logFile
        logger_recored = open(logFileLoc, 'r')
        next(logger_recored)
        lines = logger_recored.readlines()
        logger_recored.close()
        logger = open(logFileLoc, 'a+')
        return logger, lines

    def record_trainVal_log(self, logger, epoch, lr, lossTr, val_loss, Acc, Pre, Rec, F1):
        logger.write("{}\t{:.6f}\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\n".format(
            epoch, lr, lossTr, val_loss, Acc, Pre, Rec, F1 ))
        logger.flush()
        print("Epoch {}  lr={:.6f}  Train Loss={:.4f}  Val Loss={:.4f}".format(epoch, lr, lossTr, val_loss))


    def record_train_log(self, logger, epoch, lr, lossTr):
        logger.write("{}\t{:.6f}\t{:.4f}\n".format(epoch, lr, lossTr))
        logger.flush()
        print("Epoch {}  lr={:.6f}  Train Loss={:.4f}".format(epoch, lr, lossTr))
