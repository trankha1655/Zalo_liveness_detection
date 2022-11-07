import os
import shutil
import pandas
class WriteResult:
    def __init__(self, args):
        self.args = args
        self.root = args.root
        self.save_dir = args.save_seg_dir  #default: ../checkpoint/MODEL_NAME/
        self.columns = ["", ""]
        self.df = []
    
    def write_file(src, des):
        shutil.copy(src, des)


    def write_result(self, outputs, save_mp4 = False):
        names = outputs['names']
        classes = outputs['labels']

        if not 'mp4' in names:
            for file_name, label in zip(names, classes):
                #only get label, folder, name
                temp = file_name.split('/')[-3:]
                #change label = new label
                temp[0] = str(label) 
                #create sub folder if does not exist
                sub_name = os.path.join(self.save_dir, *temp[:2])
                if not os.path.exists(sub_name):
                    os.makedirs(sub_name)
                #full_name = dir/label/folder/name 
                full_name = os.path.join(sub_name, temp[-1])
                
                write_file(file_name, fullname)

        else:
            name = file_name.split('/')[-1]
            
            if save_mp4:
                
                sub_name = os.path.join(self.save_dir, str(classes))
                if not os.path.exists(sub_name):
                    os.makedirs(sub_name)
                full_name = os.path.join(sub_name, name )
                write_file(file_name, fullname)

            self.df.append([name, classes])

    def save_df(self):
        self.df = pandas.DataFrame(df, columns=['fname', 'liveness_score'])
        self.df.to_csv( os.path.join(root, "/Predict.csv"))
