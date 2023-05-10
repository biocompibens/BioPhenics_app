import sys

import torch
from pandas import DataFrame, Series, concat, read_excel
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch.nn as nn
from torchvision.models import ResNet18_Weights

from scripts.global_name import (
    FIELDS, WELLS, BARCODE, WAVE, CONTENT, FEATURE_NAME_PREFIX_DEEP_LEARNING, ALL_DATA, IMG_PATH
)
from scripts.images import ScreeningImage

RESNET_SIZE = 224


def get_imagette(inpt, x_start, y_start, size=RESNET_SIZE):
    return torch.narrow(torch.narrow(inpt, 1, x_start, size), 2, y_start, size)


class CutImage(torch.nn.Module):
    def __init__(self, imagette_size=RESNET_SIZE):
        super().__init__()
        self.imagette_size = imagette_size

    def forward(self, inpt):
        _, in_height, in_width = inpt.shape
        rest_height = int((in_height % self.imagette_size) / 2)
        rest_width = int((in_width % self.imagette_size) / 2)
        return torch.stack([
            get_imagette(inpt, i, j)
            for j in range(rest_width, in_width-self.imagette_size+1, self.imagette_size)
            for i in range(rest_height, in_height-self.imagette_size+1, self.imagette_size)
        ])


class PlateDataset(Dataset):
    """"""

    def __init__(self, plate_df, loader=ScreeningImage.from_series,
                 transform=None):
        # columns should be : [[BARCODE, WELLS, CONTENT, FIELDS, WAVE, PATH]]
        self.plate_df = plate_df
        self.loader = loader
        self.transform = transform

    def __getitem__(self, item):
        img = self.plate_df.iloc[item]
        image = self.loader(img)
        image.normalize()
        image = image.image if image.image.mode == "RGB" else image.image.convert(mode='RGB')
        transformed = self.transform(image) if self.transform is not None else np.array(image.image)
        return transformed, img[[BARCODE, WELLS, CONTENT, FIELDS, WAVE]].to_dict()

    def __len__(self):
        return len(self.plate_df)


class PlateLoader(DataLoader):
    """"""

    def __init__(self, dataset, **kwargs):
        if isinstance(dataset, DataFrame):
            platedataset_dict = {
                arg: kwargs.pop(arg)
                for arg in ['loader', 'transform']
                if arg in kwargs
            }
            dataset = PlateDataset(dataset, **platedataset_dict)

        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = None
            # disable automatic batching

        super().__init__(dataset, **kwargs)


class DeepModule(object):
    def __init__(self, plate_df, model=None, my_transforms=None, **kwargs):
        self.input_size = RESNET_SIZE
        self.use_pretrained = kwargs.get('use_pretrained', True)
        self.plate_df = plate_df

        if not model:
            model = self.initialize_model(self.use_pretrained,
                                          weights=kwargs.get('model_weights', None))
        self.model = model
        print(f"GPU availability : {torch.cuda.is_available()}")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        if my_transforms is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                # normalization from training in ImageNet
                CutImage(),
            ])
        else:
            self.transform = my_transforms

    @staticmethod
    def initialize_model(use_pretrained=True, weights=None):
        """ Resnet18
        """
        if weights is None and use_pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
        model_ft = models.resnet18(weights=weights)

        for param in model_ft.parameters():
            param.requires_grad = False

        # remove last layer
        model_ft.fc = nn.Identity()
        return model_ft

    def predict(self, df=None, **kwargs):
        if df is None:
            df = self.plate_df

        predict_loader = PlateLoader(df, transform=self.transform, **kwargs)
        self.model.eval()
        result = []
        for i, (img, metadata) in enumerate(predict_loader):
            # shape for resnet input is : (batch_size, height, width, RGB=3)
            # if we set batch_size = 1
            # we can squeeze img to get a shape of (nb_imagette, height, width, 3)
            # each channel is repeated over the three colors..
            img = img.to(self.device)
            y_hat = self.model.forward(img)
            # y_hat shape should be (nb_imagette, 512)
            # agg by median to get a 512 features long vector by image
            result.append(concat([Series(metadata), Series(torch.quantile(y_hat, q=0.5, dim=0))]))

        result = DataFrame(result).pivot(
            index=[BARCODE, WELLS, CONTENT, FIELDS], columns=WAVE
        ).swaplevel(axis=1).sort_index(axis=1)

        result.columns = [f'{FEATURE_NAME_PREFIX_DEEP_LEARNING}{col[0]}_{i}'
                          for i, col in enumerate(result.columns)]
        return result


if __name__ == '__main__':
    module = DeepModule(read_excel(sys.argv[1], sheet_name=IMG_PATH))
    prediction = module.predict()
    prediction = prediction.groupby(
        [BARCODE, WELLS, CONTENT]
    ).agg('median').reset_index().to_excel(
        sys.argv[2], sheet_name=ALL_DATA, index=False
    )
