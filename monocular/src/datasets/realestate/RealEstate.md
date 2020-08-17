### Real Estate Dataset

### Original Dataset
The dataset offers a set of text files that contain information about each video clip.
Each video clip is associated with a one text file.
Read more here
https://google.github.io/realestate10k/download.html

#### Text files
The text files can be found in ``` /home5/anwar/data/realestate10k/text_files/ ```

## Training/Test clips
Each video clip has been downloaded and the frames are extracted. Here you will find Training and Test sequences
```
# Train
/home5/anwar/data/realestate10k/extracted/train
# Test
/home5/anwar/data/realestate10k/extracted/test
```

The train and test subforders have a subfolder per video clip.
Each clip is stored in folder that has the same name as the text file. For instance,
folder
```
/home5/anwar/data/realestate10k/extracted/train/71e276d963c867d3.txt
# even though it ends in .txt this is a folder
```
corresponds to the text file
```
/home5/anwar/data/realestate10k/text_files/train/71e276d963c867d3.txt
```


#### How to use this in data loaders
##### Read Images:
from the folders under extracted/train, extraced/test
##### Read Pose:
from the text files under text_files/train, text_files/test
# read the website on how K, R and T are defined



#### Faulty data

Due to copyright some video clips are not downloaded. So you can find the list of these clips under
```
/home5/anwar/data/realestate10k/faulty_folders_test.txt
/home5/anwar/data/realestate10k/faulty_folders_train.txt
```
