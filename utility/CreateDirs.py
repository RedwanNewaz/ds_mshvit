from pyTree.Tree import Tree as Tree
from queue import Queue
import os

def getDirModel():
    root = Tree('Resources')
    dataset = Tree('Dataset')
    SewerML = Tree('SewerML')
    Carencro = Tree('Carencro')
    dataset.addChildren([SewerML, Carencro])

    ## model
    model = Tree('Model')
    data = Tree('Data')

    ## leaves
    train = Tree('Train')
    test = Tree('Test')
    validation = Tree('Validation')


    data_leaves = [train, validation, test]
    data.addChildren(data_leaves)
    annotation = Tree('Annotation')
    annotation.addChildren(data_leaves)

    ## networks
    BotNet_50_S1 = Tree('BotNet-50-S1')
    CoAtNet_0 = Tree('CoAtNet-0')
    CoAtNet_1 = Tree('CoAtNet-1')

    ResNet_18 = Tree('ResNet-18')
    ResNet_34 = Tree('ResNet-34')
    ResNet_50 = Tree('ResNet-50')
    ResNet_101 = Tree('ResNet-101')

    TResNet_M = Tree('TResNet-M')
    TResNet_L = Tree('TResNet-L')

    networks = [BotNet_50_S1, CoAtNet_0, CoAtNet_1, ResNet_18, ResNet_34, ResNet_50, ResNet_101, TResNet_M, TResNet_L]

    # leaves
    weight = Tree('Weight')
    prediction = Tree('Prediction')
    evaluation = Tree('Evaluation')

    # connect to root node
    backbone = Tree('Backbone')
    mshvit = Tree('MSHViT')
    ds_mshvit = Tree('DS-MSHViT')
    netChildren = [backbone, mshvit, ds_mshvit]

    leaves = [weight, prediction, evaluation]

    for net in networks:
        net.addChildren(netChildren)

    for child in netChildren:
        child.addChildren(leaves)

    model.addChildren(networks)

    datasetChildren = [data, annotation, model]
    SewerML.addChildren(datasetChildren)
    Carencro.addChildren(datasetChildren)
    root.addChild(dataset)

    return root


def DFS_create_dirs(root, dirName=""):
    if not root:
        return

    dirName += f"{root}" if root.isRoot() else f"/{root}"
    for child in root.getChildren():
        DFS_create_dirs(child, dirName)
    if root.isBranch() and len(dirName):
        print(f"{dirName}")
        os.makedirs(dirName, exist_ok=True)



def DFS_with_filter(root, query, dirName=""):
    if not root:
        return

    dirName += f"{root}" if root.isRoot() else f"/{root}"
    for child in root.getChildren():
        DFS_with_filter(child, query, dirName)
    if root.isBranch() and len(dirName):
        token = dirName.split("/")
        check = [q in token for q in query]
        if all(check):
            print(f"{dirName}")
        # os.makedirs(dirName, exist_ok=True)


if __name__ == '__main__':
    model = getDirModel()
    # model.prettyTree()
    # DFS_create_dirs(model)
    token = ["SewerML", "MSHViT", "Weight"]
    # token = ["SewerML", "Backbone", "Weight"]
    DFS_with_filter(model, token)