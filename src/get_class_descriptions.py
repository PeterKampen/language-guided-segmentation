from openai import OpenAI
from scripts.regsetup import description

from coco_dataset import COCODataset
import os
import json
BASE_PATH = r'C:\Users\pjtka\Documents\COCO'


class ClassDescriptionsGetter:

    def __init__(self, outdir = None):
        self.outdir = outdir
        if self.outdir is None:
            self.outdir = os.path.join(BASE_PATH, 'class_descriptions')
            if not os.path.exists(self.outdir):
                os.makedirs(self.outdir)


        self.dataset = COCODataset(BASE_PATH)

        self.openai_prompt = lambda class_name: f"Please provide a detailed description of the visual characteristics that uniquely identify the {class_name} object class, distinguishing it from other similar object categories. Focus solely on the distinguishing visual features in a comprehensive paragraph."
        self.class_descriptions = []

        self.client = OpenAI(
            api_key="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"
        )

    @property
    def class_names(self, ):
        return self.dataset.get_category_names()


    def generate_(self, class_name):
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            store=True,
            messages=[
                {"role": "user", "content": self.openai_prompt(class_name)}
            ]
        )

        return completion.choices[0].message



    def generate(self):

        for class_name in self.class_names:
            description = self.generate_(class_name)
            self.class_descriptions.append({'class': class_name,'description': description.content})


        with open(os.path.join(self.outdir, 'class_descriptions.json'), 'w') as f:
            json.dump(self.class_descriptions, f, ensure_ascii=False)





if __name__ == '__main__':

    getter = ClassDescriptionsGetter()
    print(getter.class_names)
    print(len(getter.class_names))




