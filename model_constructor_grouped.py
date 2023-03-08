import torch.nn as nn
import torchvision
import torch


class SimpleResNet18Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.weight.shape[1], num_classes)

    def forward(self, x):
        y = self.model(x)
        return y


class SimpleResNet34Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.weight.shape[1], num_classes)

    def forward(self, x):
        y = self.model(x)
        return y


class SimpleResNet50Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.weight.shape[1], num_classes)

    def forward(self, x):
        y = self.model(x)
        return y


class SimpleResNet18Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.weight.shape[1], num_classes)

    def forward(self, x):
        y = self.model(x)
        return y


class SimpleVGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.vgg16(pretrained=True)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].weight.shape[1], num_classes)

    def forward(self, x):
        y = self.model(x)
        return y


class SimpleMobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.mobilenet_v2(pretrained=True)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].weight.shape[1], num_classes)

    def forward(self, x):
        y = self.model(x)
        return y


class SimpleDenseNet121(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.densenet121(pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.weight.shape[1], num_classes)
        # self.model.classifier = nn.Sequential(
        #     *[nn.Dropout(0.5), nn.Linear(self.model.classifier.weight.shape[1], num_classes)])

    def forward(self, x):
        y = self.model(x)
        return y


class TwoHeadedMobileNetV2(nn.Module):
    """
    num_classes_1 - classes for "manufacturer+carmodel" label
    num_classes_2 - classes for "manufacturer+year" label
    """

    def __init__(self, num_classes, num_classes_2, dropout_1, dropout_2):
        super().__init__()
        self.model = torchvision.models.mobilenet_v2(pretrained=True).features
        self.size = torchvision.models.mobilenet_v2().classifier[-1].weight.shape[1]
        self.dropout = nn.Dropout(p=dropout_1)
        # classifier for manufacturer + carmodel
        self.classifier_1 = nn.Linear(self.size, num_classes)
        # classifier for manufacturer + year
        self.classifier_2 = nn.Linear(self.size, num_classes_2)

    def _forward_impl(self, x):
        x = self.model(x)  # from feature extractor
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        y_1 = self.classifier_1(x)
        y_2 = self.classifier_2(x)
        return y_1, y_2

    def forward(self, x):
        return self._forward_impl(x)


class TwoHeadedEfficientnetB4Model(nn.Module):
    """
    num_classes_1 - classes for "manufacturer+carmodel" label
    num_classes_2 - classes for "manufacturer+year" label
    """

    def __init__(self, num_classes, num_classes_2, dropout_1, dropout_2):
        super().__init__()
        self.model = torchvision.models.efficientnet_b4(pretrained=True).features
        self.size = torchvision.models.efficientnet_b4().classifier[-1].weight.shape[1]
        self.dropout = nn.Dropout(p=dropout_1)
        # classifier for manufacturer + carmodel
        self.classifier_1 = nn.Linear(self.size, num_classes)
        # classifier for manufacturer + year
        self.classifier_2 = nn.Linear(self.size, num_classes_2)

    def _forward_impl(self, x):
        x = self.model(x)  # from feature extractor
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        y_1 = self.classifier_1(x)
        y_2 = self.classifier_2(x)
        return y_1, y_2

    def forward(self, x):
        return self._forward_impl(x)


class TwoHeadedMobileNetV2_modified_input(nn.Module):
    """
    num_classes_1 - classes for "gender" label
    num_classes_2 - classes for "reset_labels" label
    """
    def __init__(self, num_classes, num_classes_2, dropout_1, dropout_2, groups):
        super().__init__()
        self.model = torchvision.models.mobilenet_v2(pretrained=True).features
        self.model[0][0] = nn.Sequential(*[
                nn.Conv2d(groups*3, 27, groups=groups, kernel_size=3, stride=2, padding=1, bias=False),
                nn.Conv2d(27, 32, groups=1, kernel_size=1, stride=1, padding=0, bias=False),
        ])
        self.size = torchvision.models.mobilenet_v2().classifier[-1].weight.shape[1]
        self.dropout = nn.Dropout(p=dropout_1)
        self.classifier_1 = nn.Linear(self.size, 1)
        self.classifier_2 = nn.Linear(self.size, num_classes_2)

    def _forward_impl(self, x):
        x = self.model(x)  # from feature extractor
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        y_1 = self.classifier_1(x)
        y_2 = self.classifier_2(x)
        return y_1, y_2

    def forward(self, x):
        return self._forward_impl(x)


# class TwoHeadedMobileNetV2_modified_inputH3(nn.Module):
#     """
#     num_classes_1 - classes for "manufacturer+carmodel" label
#     num_classes_2 - classes for "manufacturer+year" label
#     """
#
#     def __init__(self, num_classes, num_classes_2, num_classes_3, dropout_1, dropout_2, groups):
#         super().__init__()
#         self.model = torchvision.models.mobilenet_v2(pretrained=True).features
#         self.model[0][0] = nn.Sequential(*[
#             nn.Conv2d(groups*3, 27, groups=groups, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.Conv2d(27, 32, groups=1, kernel_size=1, stride=1, padding=0, bias=False),
#         ])
#         self.size = torchvision.models.mobilenet_v2().classifier[-1].weight.shape[1]
#         self.dropout = nn.Dropout(p=dropout_1)
#         self.classifier_1 = nn.Linear(self.size, 1)
#         self.classifier_2 = nn.Linear(self.size, num_classes_2)
#         self.classifier_3 = nn.Linear(self.size, num_classes_3)
#
#     def forward(self, x):
#         x = self.model(x)
#         x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
#         x = torch.flatten(x, 1)
#         x = self.dropout(x)
#         y_1 = self.classifier_1(x)
#         y_2 = self.classifier_2(x)
#         y_3 = self.classifier_3(x)
#         return y_1, y_2, y_3


class TwoHeadedMobileNetV2_modified_inputH3(nn.Module):
    """
    num_classes_1 - classes for "gender" label
    num_classes_2 - classes for "reset_labels" label
    num_classes_2 - classes for "age" label
    """
    def __init__(self,
                 num_classes,
                 num_classes_2,
                 num_classes_3,
                 dropout_1,
                 dropout_2,
                 groups,
                 focal_gamma
                 ):
        super().__init__()
        self.model = torchvision.models.mobilenet_v2(pretrained=True).features
        self.model[0][0] = nn.Sequential(*[
            nn.Conv2d(groups*3, 27, groups=groups, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(27, 32, groups=1, kernel_size=1, stride=1, padding=0, bias=False),
        ])
        self.size = torchvision.models.mobilenet_v2().classifier[-1].weight.shape[1]
        self.dropout = nn.Dropout(p=dropout_1)
        self.classifier_1 = nn.Linear(self.size, 1)
        self.classifier_2 = nn.Linear(self.size, num_classes_2)
        self.classifier_3 = nn.Linear(self.size, num_classes_3)
        self.gamma_1 = torch.nn.Parameter(torch.Tensor([focal_gamma]))
        self.gamma_2 = torch.nn.Parameter(torch.Tensor([focal_gamma]))
        self.gamma_3 = torch.nn.Parameter(torch.Tensor([focal_gamma]))

    def forward(self, x):
            x = self.model(x)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            y_1 = self.classifier_1(x)
            y_2 = self.classifier_2(x)
            y_3 = self.classifier_3(x)
            return y_1, y_2, y_3


class TwoHeadedMobileNetV2_modified_inputH4(nn.Module):
    """
    num_classes_1 - classes for "gender" label
    num_classes_2 - classes for "reset_labels" label
    num_classes_2 - classes for "age" label
    num_classes_2 - classes for "accessory" label
    """
    def __init__(self,
                 num_classes,
                 num_classes_2,
                 num_classes_3,
                 num_classes_4,
                 dropout_1,
                 dropout_2,
                 groups,
                 focal_gamma
                 ):
        super().__init__()
        self.model = torchvision.models.mobilenet_v2(pretrained=True).features
        self.model[0][0] = nn.Sequential(*[
            nn.Conv2d(groups*3, 27, groups=groups, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(27, 32, groups=1, kernel_size=1, stride=1, padding=0, bias=False),
        ])
        self.size = torchvision.models.mobilenet_v2().classifier[-1].weight.shape[1]
        self.dropout = nn.Dropout(p=dropout_1)
        self.classifier_1 = nn.Linear(self.size, 1)
        self.classifier_2 = nn.Linear(self.size, num_classes_2)
        self.classifier_3 = nn.Linear(self.size, num_classes_3)
        self.classifier_4 = nn.Linear(self.size, num_classes_4)
        self.gamma_1 = torch.nn.Parameter(torch.Tensor([focal_gamma]))
        self.gamma_2 = torch.nn.Parameter(torch.Tensor([focal_gamma]))
        self.gamma_3 = torch.nn.Parameter(torch.Tensor([focal_gamma]))
        self.gamma_4 = torch.nn.Parameter(torch.Tensor([focal_gamma]))

    def forward(self, x):
        x = self.model(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        y_1 = self.classifier_1(x)
        y_2 = self.classifier_2(x)
        y_3 = self.classifier_3(x)
        y_4 = self.classifier_4(x)
        return y_1, y_2, y_3, y_4


class MobileNetV2_custom(nn.Module):
    """
    num_classes_1 - classes for "gender" label
    num_classes_2 - classes for "reset_labels" label
    num_classes_2 - classes for "age" label
    num_classes_2 - classes for "accessory" label
    """
    def __init__(self,
                 num_classes,
                 dropout,
                 groups,
                 label_column_dict,
                 ):

        super().__init__()
        self.model = torchvision.models.mobilenet_v2(pretrained=True).features
        self.model[0][0] = nn.Sequential(*[
            nn.Conv2d(groups*3, 27, groups=groups, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(27, 32, groups=1, kernel_size=1, stride=1, padding=0, bias=False),
        ])
        self.size = torchvision.models.mobilenet_v2().classifier[-1].weight.shape[1]
        self.dropout = nn.Dropout(p=dropout)
        self.classifiers = [nn.Linear(self.size, num_classes[head]) for head in sorted(num_classes.keys())]
        self.classifiers = torch.nn.ModuleList(self.classifiers)

    def forward(self, x):
        x = self.model(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        outputs = [classifier(x) for classifier in self.classifiers]
        return outputs


def model_constructor_grouped(model,
                              unfreeze,
                              groups,
                              num_classes,
                              dropout,
                              label_column_dict
                              ):
    """Build model from corresponding model class and
    set number of freeze layers.
    Args:
        model - model name according to "torchvision.models"
        unfreeze - number of layers to unfreeze from top
        num_classes - number of classes for the classifier
    -----------------
    Supporter Models:
        resnet18,
        resnet34,
        resnet50,
        vgg16,
        mobilenet_v2,
        densenet121,
        mobilenet_v2_2heads
    """
    print(f"Loading model {model}...")
    if model == 'resnet18':
        model = SimpleResNet18Model(num_classes)
    elif model == 'resnet34':
        model = SimpleResNet34Model(num_classes)
    elif model == 'resnet50':
        model = SimpleResNet50Model(num_classes)
    elif model == 'vgg16':
        model = SimpleVGG16(num_classes)
    # elif model == 'mobilenet_v2':
    #     model = SimpleMobileNetV2(num_classes)
    elif model == 'densenet121':
        model = SimpleDenseNet121(num_classes)

    elif model == 'mobilenet_v2':
        model = MobileNetV2_custom(num_classes, dropout, groups, label_column_dict)


    # elif model == 'mobilenet_v2_2heads':
    #     assert len(num_classes) > 2, \
    #         "You use two-headed network! num_classes_2 must be > 0"
    #     model = TwoHeadedMobileNetV2(num_classes, num_classes_2, dropout_1, dropout_2)
    # elif model == 'efficientnet_b4_2heads':
    #     assert num_classes_2 > 0, \
    #         "You use two-headed network! num_classes_2 must be > 0"
    #     model = TwoHeadedEfficientnetB4Model(num_classes, num_classes_2, dropout_1, dropout_2)
    # elif model == 'mobilenet_v2_2heads_modified_input':
    #     assert num_classes_2 > 0, \
    #         "You use two-headed network! num_classes_2 must be > 0"
    #     model = TwoHeadedMobileNetV2_modified_input(num_classes, num_classes_2, dropout_1, dropout_2, groups)
    # elif model == 'mobilenet_v2_3heads_modified_input':
    #     assert num_classes_2 > 0 and num_classes_3 > 0, \
    #         "You use two-headed network! num_classes_2 must be > 0"
    #     model = TwoHeadedMobileNetV2_modified_inputH3(num_classes, num_classes_2, num_classes_3,
    #                                                   dropout_1, dropout_2, groups,
    #                                                   focal_gamma)
    # elif model == 'mobilenet_v2_3heads_modified_input':
    #     assert num_classes_2 > 0 and num_classes_3 > 0, \
    #         "You use two-headed network! num_classes_2 must be > 0"
    #     model = TwoHeadedMobileNetV2_modified_inputH3(num_classes, num_classes_2, num_classes_3,
    #                                                   dropout_1, dropout_2, groups,
    #                                                   focal_gamma)
    # elif model == 'mobilenet_v2_4heads_modified_input':
    #     assert num_classes_2 > 0 and num_classes_3 > 0 and num_classes_4 > 0, \
    #         "You use two-headed network! num_classes_2 must be > 0"
    #     model = TwoHeadedMobileNetV2_modified_inputH4(num_classes, num_classes_2, num_classes_3, num_classes_4,
    #                                                   dropout_1, dropout_2, groups, focal_gamma)

    # HOWTO check layers:
    # [(g.requires_grad, p[0]) for g, p in zip(model.parameters(), model.named_parameters())]

    # Add here your custom model!
    # ----------- >< ------------
    # Add here your custom model!

    else:
        raise Exception(f'Wrong model: {model}')

    print("Setting model's architecture...")
    parameters = [parameter for parameter in model.parameters()]

    # freeze all layers
    for p in parameters:
        p.requires_grad = False

    # unfreeze two bottom layers (Conv, Conv)
    # parameters[0].requires_grad = True
    # parameters[1].requires_grad = True

    # unfreeze n top layers
    n = unfreeze
    if n > 0:
        for parameter in parameters[-n:]:
            parameter.requires_grad = True

    # check number of unfreezed layers
    n = 0
    for p in model.parameters():
        n += 1 if p.requires_grad else 0
    print(f'{n} of {len(parameters)} layer(s) unfreezed')

    return model
