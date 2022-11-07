from sklearn.cluster import KMeans as KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
def get_full_recall(model, dataloader ):
    data_embedding, label_list = get_embedding(model, dataloader)
    distance_matrix = torch.cdist(data_embedding, data_embedding)

    recall_k_list = [1,2,4,8]
    neighbors = distance_matrix.topk(recall_k_list[len(recall_k_list) - 1]+1, largest = False)[1][:,1:recall_k_list[len(recall_k_list) - 1]+1] # as 0 th element is the trivially the same point
    recalls = []
    for k in recall_k_list:
        recall = get_recall(label_list, neighbors, k)
        recalls.append(recall)
        print("Recall@{} {:.4f}".format(k,recall*100))
    return recalls

def get_recall(label_list, neighbors, k):
    matches = 0
    num1 = 0
    for num1 in range(label_list.shape[0]):
        if(label_list[num1] in label_list[neighbors[num1,:k]] ):
            matches+=1
    recall_k = matches / label_list.shape[0]
    return recall_k

def get_embedding(model, dataloader):
    model.eval()
    total_images = len(dataloader)
    num_saved = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.cuda()
            labels = labels.cuda()
            embedding_batch = model(images)
            if(num_saved == 0):
                full_embedding = torch.zeros(total_images, embedding_batch.shape[1])
                label_list = torch.zeros(total_images)
            full_embedding[num_saved:num_saved+embedding_batch.shape[0], :] = embedding_batch
            label_list[num_saved:num_saved+embedding_batch.shape[0]] = labels

    model.train()
    return full_embedding, label_list

def get_NMI(model, dataloader):
    data_embedding, label_list = get_embedding(model, dataloader)
    num_clusters = dataloader.dataset.n_classes

    clusters = Kmeans(num_clusters)
    cluster_labels = clusters.fit(data_embedding.cpu().numpy(), num_classes).labels_
    nmi = NMI(cluster_labels , label_list)
    return nmi

def get_recall_and_NMI(model, dataloader):
    data_embedding, label_list = get_embedding(model, dataloader)
    num_clusters = dataloader.dataset.n_classes

    clusters = Kmeans(num_clusters)
    cluster_labels = clusters.fit(data_embedding.cpu().numpy(), num_classes).labels_
    nmi = NMI(cluster_labels , label_list.cpu().numpy())
    print('NMI = {:.4f}'.format(nmi))

    distance_matrix = torch.cdist(data_embedding, data_embedding)
    recall_k_list = [1,2,4,8]
    neighbors = distance_matrix.topk(recall_k_list[len(recall_k_list) - 1]+1, largest = False)[1][:,1:recall_k_list[len(recall_k_list) - 1]+1] # as 0 th element is the trivially the same point
    recalls = []
    for k in recall_k_list:
        recall = get_recall(label_list, neighbors, k)
        recalls.append(recall)
        print("Recall@{} {:.4f}".format(k,recall*100))
    return recalls, nmi
