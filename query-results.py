import torch
import torchvision
from deep_net.googlenet import *
import eval_dataset
import numpy as np
import matplotlib.pyplot as plt
from evaluate import *

def get_embedding_images(model, dataloader):
    model.eval()
    total_images = len(dataloader.dataset)
    num_saved = 0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(dev)
            labels = labels.to(dev)
            embedding_batch = model(images)
            if(num_saved == 0):
                full_embedding = torch.zeros(total_images, embedding_batch.shape[1])
                label_list = torch.zeros(total_images)
                h, w = images.shape[2], images.shape[3]
                allImages = torch.zeros(total_images, 3, h, w)
            full_embedding[num_saved:num_saved+embedding_batch.shape[0], :] = embedding_batch
            label_list[num_saved:num_saved+embedding_batch.shape[0]] = labels
            allImages[num_saved:num_saved+embedding_batch.shape[0]] = images
            num_saved = num_saved + embedding_batch.shape[0]

    model.train()
    return full_embedding, label_list, allImages

datasets = ['cub', 'cars']

nQueries = 5
imagesPerQuery = 6
for dataset in datasets:
    modelFileName = f'trained_data/n_pair_model_dict_{dataset}.pt'

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = googlenet_metric(embed_size=512)
    model.load_state_dict(torch.load(modelFileName, map_location=torch.device(dev)))
    model.to(dev)

    testset = eval_dataset.load(name=dataset,
                                root='./data/'+dataset.upper()+'/',
                                mode='eval',
                                transform = eval_dataset.utils.make_transform(is_train=False))
    test_loader = torch.utils.data.DataLoader(testset, batch_size =100,
                        shuffle=False, num_workers=8, pin_memory=True,
                        drop_last=False)

    data_embedding, label_list, images = get_embedding_images(model, test_loader)

    indicesToQuery = torch.randperm(len(data_embedding))[:nQueries]
    imageList = []
    for index in indicesToQuery:
        queryEmbedding = data_embedding[index]
        distance_matrix = torch.cdist(queryEmbedding.view(1,1,512), data_embedding.view(1,len(data_embedding),512))
        closestIndices = distance_matrix.topk(imagesPerQuery, largest=False)[1][0,0]
        imageList.append(images[closestIndices])
    queryImages = images[indicesToQuery]
    h, w = images[0].shape[1], images[0].shape[2]
    canvas = torch.ones(3, nQueries*(h+10)-10, (imagesPerQuery-1)*w+20+w)
    for i in range(nQueries):
        queryImage = torch.flip(queryImages[i], [1, 2])
        queryImage = (queryImage-torch.min(queryImage))/(torch.max(queryImage)-torch.min(queryImage))
        canvas[:, i*(h+10):i*(h+10)+h, :w] = queryImage
        for j in range(imagesPerQuery-1):
            imageToWrite = torch.flip(imageList[i][j+1], [1, 2])
            imageToWrite = (imageToWrite-torch.min(imageToWrite))/(torch.max(imageToWrite)-torch.min(imageToWrite))
            canvas[:, i*(h+10):i*(h+10)+h, w+20+j*w:w+20+j*w+w] = imageToWrite
    imageGrid = canvas.numpy().transpose(1, 2, 0)
    totalH, totalW = imageGrid.shape[0], imageGrid.shape[1]
    plt.figure()
    plt.imshow(imageGrid)
    plt.plot([w+10, w+10], [-10., totalH+10.], 'k--')
    plt.xlim([0., totalW])
    plt.ylim([0., totalH])
    plt.axis('off')
    plt.title('Retrieved images from queries')
    plt.savefig(f'queries-{dataset}.pdf', bbox_inches='tight')

