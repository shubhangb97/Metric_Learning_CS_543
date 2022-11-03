import torch
import torchvision
import numpy as np

class BaseDataset(torch.utils.data.Dataset):
    """
    Base dataset to be used by other datasets
    Following need to created by the __init__ of derived datates.
        self.nClasses -- number of classes
        self.fileNames -- locations of image files (nClasses+1, 1-indexed)
        self.indicesForClass -- {"train":[], "test":[]} for each class,
            of length nClasses+1, 1-indexed
        self.classNames -- names of the classes (nClasses+1 length)
        self.train -- indicate if dataset is in train or test mode
        self.outputShape -- output of dataset in this shape (tuple (h,w))
        self.transform -- transformation which convert output to above shape
    """

    def __len__(self):
        """Minimum of # of images in classes (subject to self.train)"""
        return min([self.getClassLen(i) for i in range(1, self.nClasses+1)])

    def __getitem__(self, index):
        """
        The data is drawn randomly from the train or test datasets 
        depending on the value of self.train
        Argument index is ignored
        Use the getClassImage function for getting all images in
        non-random manner
        Output:
            Tensor of size (200 x 3 x h x w ), one image for each
                of the 200 classes
        """
        imageList = torch.zeros((self.nClasses, 3,
                                self.outputShape[0], self.outputShape[1]))
        for i in range(self.nClasses):
            whichData = "train" if self.train else "test"
            index = int(torch.randint(high=self.getClassLen(i+1), size=(1,)))
            imageIdx = self.indicesForClass[i+1][whichData][index]
            fileName = self.fileNames[imageIdx]
            image = torchvision.io.read_image(fileName)
            image = self.transform(image)
            imageList[i] = image
        return imageList

    def setTrain(self, train):
        """Set the dataset in train or test mode -- train is bool"""
        self.train = train
    
    def getClassLen(self, classIdx):
        """
        Get # of images with label classIdx (subject to self.train)
        Note that classIdx indexed starts from 1.
        """
        whichData = "train" if self.train else "test"
        return len(self.indicesForClass[classIdx][whichData])
    
    def getClassName(self, classIdx):
        return self.classNames[classIdx]
    
    def getClassImage(self, classIdx, idx, applyTransform=True):
        """
        Get the idx-th image with label classIdx
        Args:
            classIdx (int): class label (indexing starts at 1)
            idx (int): index of image within class (starts at 0)
            applyTransform (bool, default=True): apply the image
                reshaping transforming before returning
        Output:
            Tensor of size (3 x h x w) -- h and w are fixed by
                self.outputShape if applyTransform is True
        """
        whichData = "train" if self.train else "test"
        imageIdx = self.indicesForClass[classIdx][whichData][idx]
        fileName = self.fileNames[imageIdx]
        image = torchvision.io.read_image(fileName)
        if applyTransform:
            image = self.transform(image)
        return image

    def getOutputShape(self):
        """Return self.outputShape"""
        return self.outputShape
    
    def getNClasses(self):
        """Return number of classes"""
        return self.nClasses

    def printStats(self):
        """Print some stats about the dataset"""
        print("\nComputing some stats about the dataset..")
        oldTrain = self.train

        self.train = True
        hList, wList, nImageList = [], [], []
        for i in range(1, self.nClasses+1):
            for j in range(self.getClassLen(i)):
                image = self.getClassImage(i, j, False)
                _, h, w = image.shape
                hList.append(h)
                wList.append(w)
            nImageList.append(self.getClassLen(i))
            print(f"\rDone computing over {i}/{self.nClasses} classes for train data..", end="")
        print(f"\rFor the train data:" + " "*50)
        mu, std = np.mean(hList), np.std(hList)
        print(f"    average image height = {mu:.3f}, std = {std:.3f}")
        mu, std = np.mean(wList), np.std(wList)
        print(f"    average image width  = {mu:.3f}, std = {std:.3f}")
        mu, std = np.mean(nImageList), np.std(nImageList)
        print(f"    average # of images per class = {mu:.3f}, std = {std:.3f}")
        print(f"    minimum # of images in a class = {min(nImageList)}")

        self.train = False
        hList, wList, nImageList = [], [], []
        for i in range(1, self.nClasses+1):
            for j in range(self.getClassLen(i)):
                image = self.getClassImage(i, j, False)
                _, h, w = image.shape
                hList.append(h)
                wList.append(w)
            nImageList.append(self.getClassLen(i))
            print(f"\rDone computing over {i}/{self.nClasses} classes for test data..", end="")
        print(f"\rFor the test data:" + " "*50)
        mu, std = np.mean(hList), np.std(hList)
        print(f"    average image height = {mu:.3f}, std = {std:.3f}")
        mu, std = np.mean(wList), np.std(wList)
        print(f"    average image width  = {mu:.3f}, std = {std:.3f}")
        mu, std = np.mean(nImageList), np.std(nImageList)
        print(f"    average # of images per class = {mu:.3f}, std = {std:.3f}")
        print(f"    minimum # of images in a class = {min(nImageList)}\n")

        self.train = oldTrain

