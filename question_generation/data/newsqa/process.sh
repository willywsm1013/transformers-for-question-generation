# follow the READMD.md in https://github.com/Maluuba/newsqa to download NewsQA data

# convert to squad format
python convert_to_squad.py

# extract context in train.json and truncate 
python truncate.py train.json 2
