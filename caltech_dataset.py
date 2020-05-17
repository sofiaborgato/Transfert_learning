from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        #self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        self.root = root
 
        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''
        z=0
        c=0
        
        dir = os.path.expanduser(root+'101_ObjectCategories/')
        l=[]
        self.samples=[]
        labels=[]
        n=[]
        
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]        
        classes.sort()
        print(len(classes),classes)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        del class_to_idx['BACKGROUND_Google']
        print(len(class_to_idx),class_to_idx)
        b = 0
        with open(root + split+'.txt','r') as f:
            for line in f:
                row = line.split("\n")[0]
                #empty = field[1]
                label = line.split("/")[0]
                if(label != 'BACKGROUND_Google'):
                    self.path = os.path.join(dir,row)
                    s = class_to_idx[label]
                    self.n=n
                    self.n.append(s)
                    self.samples.append((pil_loader(self.path),s))
                    c = c+1
                z+=1
        
        
        

        dim = self.__len__()
        f.close()
        
        print("iterations without BACKGROUND",c)
        print("iterations with BACKGROUND",z)
       
        print("Number of images of %s = %d "%(split, dim))
        print("Number of images of %s without BACKGROUND_Google Class = %d" %(split,c))      
        print("len",len(class_to_idx))


    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''   
        #image, label =  
                           # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        image,label = self.samples[index]        #print("index",index)
        #print("label",label)
        #image,label = self.path,index
        #image = pil_loader(self.path)
        #image = pil_loader(path)

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)
            #print(image)
            #label=self.transform(label)
        if label>4048:
            label-=4048
        return image, label-1

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        print(len(self.n))
         # Provide a way to get the length (number of elements) of the dataset
        return len(self.n)
