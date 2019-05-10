from s04 import preprocess, read_data as data
from s04.evaluate_performance import evaluate_performance as evaluate
from s04.dtw import DTW
import csv
import operator

from itertools import islice

redo_preprocess = False
folder_unprocessed = "PatRec17_KWS_Data"
lst = ['270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '300', '301', '302', '303', '304']

if redo_preprocess:
    # Crop images + use OTSU
    print("------------- Preprocess image (crop and OTSU)")
    preprocess.svg_to_data(folder_unprocessed, lst)
    print("------------- End of preprocessing")
    print("-------------")

print("Create training dictionnary")
training_dict = data.create_training_set('data/train',
                                         folder_unprocessed + '/ground-truth/transcription.txt',
                                         folder_unprocessed + '/task/train.txt')
print("Number of different words in the training data: %d" % len(training_dict))
print("End of creation of the training dictionnary")

keyword = open(folder_unprocessed + "/task/keywords.txt").read().splitlines()
validation_dict = data.create_set('data/valid')
transcriptions_dict = data.get_transcriptions(folder_unprocessed + '/ground-truth/transcription.txt')

print("Number of element in the validation dict: %d" % len(validation_dict))

output_lines = []
for key, image in validation_dict.items():
    dtw_o = DTW(image)
    output_line = [key]
    score_dict = {}
    i = 0
    for key2, image2 in training_dict.items():

        score = dtw_o.calculate_cost(image2)
        score_dict[key2] = score
        i += 1
        if i % 100 == 0:
            print(i)
        break
    sorted_score = dict(sorted(score_dict.items(), key = operator.itemgetter(1)))
    for name, score in sorted_score.items():
        output_line.append(name)
        output_line.append(score)
    output_lines.append(output_line)
    break
with open("output3.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(output_lines)
