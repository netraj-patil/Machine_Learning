import datasets
from datasets import load_dataset
import os
# from pathlib import Path
from PIL import Image
import json

# print('step 1')
# load_dataset('/mnt/e/Machine_Learning/dataset/SROIE2019')
# print('step 2')

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
                   "id": datasets.Value(dtype='int32', id=None),
                   
                   "tokens": datasets.Sequence(feature=datasets.Value(dtype='string', id=None), length=-1, id=None),
                   "bboxes": datasets.Sequence(feature=datasets.Sequence(feature=datasets.Value(dtype='int64', id=None), length=-1, id=None), length=-1, id=None),
                   "ner_tags": datasets.Sequence(feature=datasets.ClassLabel(names=['O', 'company', 'date', 'address', 'total'], id=None), length=-1, id=None),
                   "image": datasets.Image(mode=None, decode=True, id=None),
               }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )
    
    def _split_generators(self,abc):
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
                name='train',
                gen_kwargs={
                    "image_dir": train_dir["image"],
                    "box_dir": train_dir["box"],
                    "entities_dir": train_dir["entities"],
                }
            ),
            datasets.SplitGenerator(
                name='test',
                gen_kwargs={
                    "image_dir": test_dir["image"],
                    "box_dir": test_dir["box"],
                    "entities_dir": test_dir["entities"],
                }
            ),
        ]
    
    def _generate_examples(self, image_dir, box_dir, entities_dir):
        
        image_files = os.listdir(image_dir)
        for id,image_file in enumerate(image_files):
            
            image_path = os.path.join(image_dir, image_file)
            box_path = os.path.join(box_dir, image_file.replace('.jpg', '.txt'))  # Adjust extension as needed
            entities_path = os.path.join(entities_dir, image_file.replace('.jpg', '.txt'))  # Adjust extension as needed
            
            # Open the text file
            with open(entities_path, "r",encoding="utf-8", errors="ignore") as file:
              # Read the entire content
              data = file.read()
            # Parse the JSON data
            data_dict = json.loads(data)
            data_list = []
            bbox_list = []
            ner_tags_list = []
            with open(box_path, "r", encoding="utf-8", errors="ignore") as file:
                for line in file:
                    if line == '':
                        continue
                    # Split the line by comma
                    split_line = line.strip().split(",")

                    # Extract first 8 numbers and convert to integers
                    try:
                        numbers = [int(num) for num in split_line[:8] if num.strip()]
                    except Exception('ValueError'):
                        continue
                    
                    del(numbers[2:4])
                    del(numbers[-2:])
                    
                    # Get the name as a string (everything after the 8th element)
                    text = ", ".join(split_line[8:])
                    
                    for ner_tag, key in enumerate(data_dict,1):
                        if data_dict[key] == text.strip():
                            ner_tags_list.append(ner_tag)
                        else:
                            ner_tags_list.append(0)
                    
                    # Append data to respective lists
                    bbox_list.append(numbers)
                    data_list.append(text)

            image, dim = load_image(image_path)
            yield id, {
                "id": id,
                "tokens": data_list,
                "bboxes": bbox_list,
                "ner_tags": ner_tags_list,
                "image": image,
            }
