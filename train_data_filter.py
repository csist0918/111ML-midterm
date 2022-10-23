from random import shuffle
import argparse
import os
import ember

#----------Parser----------
parser = argparse.ArgumentParser()
parser.add_argument("original_filepath", help="Original training file location.", type=str)
parser.add_argument("number_of_train_lines", help="Total lines amount.", type=int)
#parser.add_argument("original_test_filepath", help="Original test file location.", type=str)
#parser.add_argument("number_of_test_lines", help="Total test lines amount.", type=int)
args = parser.parse_args()


#----------New Balanced Training Set----------
original_filepath = args.original_filepath

f = open(original_filepath, "r")
lines = f.readlines()
f.close()

counter = 0

benign_counter = 0
malware_counter = 0

train_filepath = 'train.jsonl'
train = open(train_filepath, "w")

train_list = []

for line in lines:
    if "\"label\": 0" in line or "\"label\": 1" in line:
        train_list.append(line)

print(f"{'-'*10} Training File Info {'-'*10}")
print(f"Total Train original file Benign & Malware lines: {len(train_list)}")
lines = train_list
shuffle(lines)

for line in lines:
    if "\"label\": 0" in line:
        #if benign_counter < args.number_of_train_lines / 2:
        train.write(line)
        benign_counter += 1
    else:
        #if malware_counter < args.number_of_train_lines / 2:
        train.write(line)
        malware_counter += 1
train.close()
print(f"New Train File lines:\nBenign: {benign_counter} Malware: {malware_counter}")