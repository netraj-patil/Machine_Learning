from datasets import load_dataset
import os
import ast
from pathlib import Path
from PIL import Image
import pandas as pd

load_dataset('/mnt/e/Machine_Learning/dataset/SROIE2019')

_CITATION = """\
@article{,
  title={},
  author={},
  journal={},
  year={},
  volume={}
}
"""

_DESCRIPTION = """\
This is a sample dataset.
"""

data_path = '/mnt/e/Machine_Learning/dataset/SROIE2019'

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w,h = image.size
    return image, (w,h)


class DatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for dataExtraction Dataset"""
    def __init__(self, **kwargs):
        """BuilderConfig for dataExtraction Dataset.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(DatasetConfig, self).__init__(**kwargs)

class DataExtraction(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = DatasetConfig
    BUILDER_CONFIGS = [
        DatasetConfig(name="InvoiceExtraction", version=datasets.Version("1.0.0"), description="InvoiceExtraction dataset"),
    ]
    
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
               {
                   "id": datasets.Value(dtype='string', id=None),
                   
                   "tokens": datasets.Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
                   "bboxes": datasets.Sequence(feature=Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), length=-1, id=None),
                   "ner_tags": Sequence(feature=ClassLabel(names=['O', 'I-company','B-company', 'I-date','B-date', 'I-address', 'B-address', 'I-total', 'B-total'], id=None), length=-1, id=None),
                   "image": datasets.Image(mode=None, decode=True, id=None),
               }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )
    
    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        """Uses local files located with data_dir"""
        train_dir = {
            "image": "/mnt/e/Machine_Learning/dataset/SROIE2019/train/img",
            "box": "/mnt/e/Machine_Learning/dataset/SROIE2019/train/box",
            "entities": "/mnt/e/Machine_Learning/dataset/SROIE2019/train/entities",
        }
        test_dir = {
            "image": "/mnt/e/Machine_Learning/dataset/SROIE2019/test/img",
            "box": "/mnt/e/Machine_Learning/dataset/SROIE2019/test/box",
            "entities": "/mnt/e/Machine_Learning/dataset/SROIE2019/test/entities",
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "image_dir": train_dir["image"],
                    "box_dir": train_dir["box"],
                    "entities_dir": train_dir["entities"],
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "image_dir": test_dir["image"],
                    "box_dir": test_dir["box"],
                    "entities_dir": test_dir["entities"],
                }
            ),
        ]
    
    def _generate_examples(self, image_dir, box_dir, entities_dir):
        image_files = os.listdir(image_dir)
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            box_path = os.path.join(box_dir, image_file.replace('.jpg', '.box'))  # Adjust extension as needed
            entities_path = os.path.join(entities_dir, image_file.replace('.jpg', '.entities'))  # Adjust extension as needed

            yield {
                "id": datasets.Value(dtype='string', id=None),
                "tokens": datasets.Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
                "bboxes": datasets.Sequence(feature=Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), length=-1, id=None),
                "ner_tags": Sequence(feature=ClassLabel(names=['O', 'I-company','B-company', 'I-date','B-date', 'I-address', 'B-address', 'I-total', 'B-total'], id=None), length=-1, id=None),
                "image": datasets.Image(mode=None, decode=True, id=None),
                
                # "image": image_path,
                # "box": box_path,
                # "entities": entities_path,
            }
