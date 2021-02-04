# Classification
I implement a simple image classification task on custom dataset, with code for both keras and pytorch.

Because the size of the whole dataset is big, so in this project, i only create a sample of few images for demo.
You can get the whole dataset from this link:
https://www.kaggle.com/puneet6060/intel-image-classification
After downloading the dataset from above links, please arange the images as the same as folder "data" in this project.
There are 3 folders "train", "test", "val", each folder contains 6 folders that contains images from 6 classes.

To train:
- With keras model:
Simply run the file "train.py" inside the folder "keras_model".
Then,the trained model will be saved inside the folder "keras_model/checkpoint" and the history plot will be saved in folder "keras_model/logs".
