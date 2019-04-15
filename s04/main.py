import preprocess
import read_data as data
import features
from dtw import DTW

redo_preprocess = False
folder_unprocessed = "PatRec17_KWS_Data"
lst = ['270','271','272','273','274','275','276','277','278','279','300','301','302','303','304']


if redo_preprocess:
    # Crop images + use OTSU
    print("------------- Preprocess image (crop and OTSU)")
    preprocess.svg_to_data(folder_unprocessed,lst)
    print("------------- End of preprocessing")
    print("-------------")
print("------------- Create training dictionnary")
training_dict = data.create_training_set('data/train',
                                         folder_unprocessed+'/ground-truth/transcription.txt',
                                         folder_unprocessed+'/task/train.txt')
print("Number of different words in the training data: %d" % len(training_dict))
print("------------- End of creation of the training dictionnary")

keyword = open(folder_unprocessed + "/task/keywords.txt").read().splitlines()