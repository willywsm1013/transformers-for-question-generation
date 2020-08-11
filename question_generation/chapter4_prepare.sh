#####################
# prepare Infersent #
#####################
# in my thesis, use fastText embedding version of InferSent,
# so only need to download version 2 embedding and model

echo 'Preparing Infersent'
cd InferSent
mkdir encoder
embedding=fasttext
if [ $embedding == glove ]; then
    mkdir GloVe
    curl -Lo GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
    unzip GloVe/glove.840B.300d.zip -d GloVe/
    curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
elif [ $embedding == fasttext ];then
    mkdir fastText
    curl -Lo fastText/crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
    unzip fastText/crawl-300d-2M.vec.zip -d fastText/
    curl -Lo encoder/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl
else
    echo 'Unknown embedding $embedding , exit'
    exit
fi

#################
# prepare Squad #
#################
if [ ! -d data/squad ] ; then
    echo "data/squad directory not found"
    mkdir -p data/squad
fi 
if [ ! -f data/squad/train.json ];then
    echo "data/squad/train.json not found, start downloading"
    curl -Lo data/squad/train.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
fi
if [ ! -f data/squad/dev.json ]; then
    echo "data/squad/dev.json not found, start downloading"
    curl -Lo data/squad/dev.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
fi

echo "split squad data"
cd data/squad
python split_data.py
cd ../../
##########################
# prepare Wikipedia data #
##########################
cd data/wiki/
bash extract.sh
cd ../../
