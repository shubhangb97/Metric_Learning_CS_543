import torch
import torchvision
import numpy as np
from pathlib import Path
import urllib.request
import shutil
from baseDataset import *

class CUBDataset(BaseDataset):
    """Dataset for the CUB data"""

    def __init__(self, dataFolder="data", outputShape=(100,100)):
        """
        Args:
            dataFolder (string): path to data folder -- will be created if
                it does not exist
            outputShape (tuple (h, w)): output data size -- will be resized
                using torchvision.transforms.Resize()
        """
        # Check if folder already exists, else download
        cubFolder = Path(dataFolder) / "CUB"
        if not cubFolder.is_dir():
            cubFolder.mkdir(parents=True, exist_ok=True)
            fileLocation = cubFolder / "cubdata.tgz"
            urlToDownload = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
            print("Downloading CUB data...")
            urllib.request.urlretrieve(urlToDownload, str(fileLocation))
            print("Decompressing CUB data...")
            shutil.unpack_archive(fileLocation, extract_dir=str(cubFolder))
            print("Decompressing done.")
        dataFolder = cubFolder / "CUB_200_2011"

        # Store the list of training and testing images
        trainTestFile = dataFolder / "train_test_split.txt"
        self.isTrain = [-1]
        with open(trainTestFile, "r") as trainTest:
            for line in trainTest:
                number, isTrain = map(int, line.strip().split(" "))
                self.isTrain.append(-1)
                self.isTrain[number] = isTrain

        # Store the filenames corresponding to indices
        self.fileNames = [""]*(len(self.isTrain)+1)
        imagesFileName = dataFolder / "images.txt"
        with open(imagesFileName, "r") as imagesFile:
            for line in imagesFile:
                number, fileName = line.strip().split(" ")
                self.fileNames[int(number)] = str(dataFolder / "images" / fileName)

        # Store class labels corresponding to each image
        self.indicesForClass = []
        for _ in range(201):
            self.indicesForClass.append({"train":[], "test":[]})
        imageToClassFileName = dataFolder / "image_class_labels.txt"
        with open(imageToClassFileName, "r") as imageToClassFile:
            for line in imageToClassFile:
                number, classIndex = map(int, line.strip().split(" "))
                whichData = "train" if self.isTrain[number] == 1 else "test"
                self.indicesForClass[classIndex][whichData].append(number)
        
        # Store the names of the classes
        self.nClasses = 200
        self.classNames = [""]*(self.nClasses+1)
        classNamesFileName = dataFolder / "classes.txt"
        with open(classNamesFileName, "r") as classNamesFile:
            for line in classNamesFile:
                number, name = line.strip().split(" ")
                self.classNames[int(number)] = name[4:]

        self.train = True
        self.outputShape = outputShape
        self.transform = torchvision.transforms.Resize(outputShape)

        print("Done creating CUB dataset.")

class SOPDataset(BaseDataset):
    """Dataset for the SOP data"""
    def __init__(self, dataFolder="data", outputShape=(100,100)):
        """
        Args:
            dataFolder (string): path to data folder -- will be created if
                it does not exist
            outputShape (tuple (h, w)): output data size -- will be resized
                using torchvision.transforms.Resize()
        """
        # Check if folder already exists, else download
        sopFolder = Path(dataFolder) / "SOP"
        if not sopFolder.is_dir():
            sopFolder.mkdir(parents=True, exist_ok=True)
            fileLocation = sopFolder / "sopdata.zip"
            urlToDownload = "ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip"
            print("Downloading SOP data...")
            urllib.request.urlretrieve(urlToDownload, str(fileLocation))
            print("Decompressing SOP data...")
            shutil.unpack_archive(fileLocation, extract_dir=str(sopFolder))
            print("Decompressing done.")
        dataFolder = sopFolder / "Stanford_Online_Products"

        # Number of classes is 12 from the dataset
        self.nClasses = 12

        # Store class labels corresponding to each image
        self.indicesForClass = []
        for _ in range(self.nClasses+1):
            self.indicesForClass.append({"train":[], "test":[]})
        trainImagesFileName = dataFolder / "Ebay_train.txt"
        with open(trainImagesFileName, "r") as trainImagesFile:
            for i,line in enumerate(trainImagesFile):
                if i == 0: continue
                imageIdx, _, classIdx, imagePath = line.strip().split(" ")
                self.indicesForClass[int(classIdx)]["train"].append(int(imageIdx))
        testImagesFileName = dataFolder / "Ebay_test.txt"
        with open(testImagesFileName, "r") as testImagesFile:
            for i,line in enumerate(testImagesFile):
                if i == 0: continue
                imageIdx, _, classIdx, imagePath = line.strip().split(" ")
                self.indicesForClass[int(classIdx)]["test"].append(int(imageIdx))

        # Store file names for each image
        self.fileNames = [""]
        imagesFileName = dataFolder / "Ebay_info.txt"
        with open(imagesFileName, "r") as imagesFile:
            for i,line in enumerate(imagesFile):
                if i == 0: continue
                imageIdx, _, classIdx, path = line.strip().split(" ")
                self.fileNames.append("")
                imagePath = dataFolder / path
                self.fileNames[int(imageIdx)] = str(imagePath)
        
        # Store the names of the classes
        self.classNames = [""]*(self.nClasses+1)
        for i in range(1, self.nClasses+1):
            oneImageIdx = self.indicesForClass[i]["train"][0]
            oneImagePath = self.fileNames[oneImageIdx]
            parentFolder = Path(oneImagePath).parent.stem
            self.classNames[i] = parentFolder[:-6]

        self.train = True
        self.outputShape = outputShape
        self.transform = torchvision.transforms.Resize(outputShape)

        print("Done creating SOP dataset.")

class CARDataset(BaseDataset):
    pass

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    datasetNameList = [CUBDataset, SOPDataset]
    params = {"batch_size":100, "num_workers":8, "shuffle":True}
    for datasetName in datasetNameList:
        dataset = datasetName()
        dataset.printStats()

        loader = torch.utils.data.DataLoader(dataset, **params)

        batch = next(iter(loader))
        print(f"Used batch_size = {params['batch_size']} and got shape {batch.shape}")

        classIdx = int(torch.randint(low=1, high=dataset.getNClasses()+1, size=(1,)))
        images = batch[:, classIdx-1]
        imageGrid = torchvision.utils.make_grid(images, nrow=5)
        imageGrid = imageGrid.numpy().transpose(1, 2, 0)
        plt.imshow(imageGrid/255.)
        plt.axis("off")
        plt.title(f"{dataset.getClassName(classIdx)}")
        plt.show()

