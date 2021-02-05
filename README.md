# Classification
I implement a simple image classification task on custom dataset, with code for both keras and pytorch.

Because the size of the whole dataset is big, so in this project, i only create a sample of few images for demo.
You can get the whole dataset from this link:
https://www.kaggle.com/puneet6060/intel-image-classification
After downloading the dataset from above links, please arange the images as the same as folder "data" in this project.
There are 3 folders "train", "test", "val", each folder contains 6 folders that contains images from 6 classes.

- With keras model:
To train:
Simply run the file "train.py" inside the folder "keras_model".
Then,the trained model will be saved inside the folder "keras_model/checkpoint" and the history plot will be saved in folder "keras_model/logs".
For example, if there are already two folders "keras_model/checkpoint/1" and "keras_model/logs/1" before training, then after training model, the program will create two new folders "keras_model/checkpoint/2" and "keras_model/logs/2" and finally weights and logs will be saved into these two folders.

To test:
At the end of the file "keras_model/train.py", please comment the code for training and then uncomment the code for function test(). You can also pass another weight path inside the function test() if you want to load another weight model. 

With pytorch model:
To train:
Simply run the file "train.py" inside the folder "pytorch_model".
Then,the trained model will be saved inside the folder "pytorch_model/checkpoint" and the history plot will be saved in folder "pytorch/logs".
For example, if there are already two folders "pytorch_model/checkpoint/1" and "pytorch_model/logs/1" before training, then after training model, the program will create two new folders "pytorch_model/checkpoint/2" and "pytorch_model/logs/2" and finally weights and logs will be saved into these two folders.

TO test: 
At the end of the file "pytorch_model/train.py", please comment the code for training and then uncomment the code for function test(). You can also pass another weight path inside the function test() if you want to load another weight model. 





