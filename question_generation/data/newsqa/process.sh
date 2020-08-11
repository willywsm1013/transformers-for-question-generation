# download NewsQA from https://github.com/Maluuba/newsqa
# run python maluuba/newsqa/data_generator.py using py2

# convert to squad format
python convert_to_squad.py

# create short context NewsQA
python truncate.py train.json 2
