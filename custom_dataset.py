import torch
from torch.utils.data import Dataset
import cv2

class RapDataset(Dataset):
    def __init__(self, txt_file):
        self.data = []
        self.img_dim = (227,227)
        with open(txt_file) as f:
            lines = f.read().splitlines()
            for i in lines:
                path = i.split(' ')[0]
                labels = i.split(' ')[1]
                labels_atr = labels[:][0:-2]
                labels_atr2 = []
                        
                for i in labels_atr.split(","):
                    #bad code
                    if(i != ''):
                        labels_atr2.append(int(i))
                labels_view = [int(labels[-2])]
                
                labels_atr2 = torch.tensor(labels_atr2)
                labels_view = torch.tensor(labels_view)
                #self.data.append([path,labels_atr2,labels_view])
                self.data.append([path,labels_atr2])
                        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        #img_path, labels_atr, labels_view = self.data[idx]
        img_path, labels_atr = self.data[idx]
        #print(img_path, labels_atr)
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        #return img_tensor, labels_atr, labels_view
        return img_tensor.float(), labels_atr.float(), img_path 
    
def get_loader(path):
    return RapDataset(path)
