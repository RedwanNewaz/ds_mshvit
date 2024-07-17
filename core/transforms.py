import torchvision.transforms as transforms


SEWERML_MEAN = [0.523, 0.453, 0.345] 
SEWERML_STD = [0.210, 0.199, 0.154]

    
def create_sewerml_train_transformations(args):
    transform_list = [transforms.Resize(size = (args["img_size"], args["img_size"])),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.ColorJitter(0.1),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean = SEWERML_MEAN, std = SEWERML_STD)]

    return transforms.Compose(transform_list)


def create_sewerml_eval_transformations(args):
    transform_list = [transforms.Resize(size = (args["img_size"], args["img_size"])),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean = SEWERML_MEAN, std = SEWERML_STD)]

    return transforms.Compose(transform_list)