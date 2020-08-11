evaluation_dir=save/chapter3/evaluation

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

#####################
# prepare Infersent #
#####################
# in my thesis, use fastText embedding version of InferSent,
# so only need to download version 2 embedding and model

echo 'Preparing Infersent'
cd InferSent
mkdir -p encoder
embedding=fasttext

if [ $embedding == glove ]; then
    mkdir -p GloVe
    if [ ! -f Glove/glove.840B.300d ]; then
        curl -Lo GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
        unzip GloVe/glove.840B.300d.zip -d GloVe/
    fi
    if [ ! -f encoder/infersent1.pkl ];then
        curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
    fi
elif [ $embedding == fasttext ];then
    mkdir -p fastText
    if [ ! -f fastText/crawl-300d-2M.vec ];then
        curl -Lo fastText/crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
        unzip fastText/crawl-300d-2M.vec.zip -d fastText/
    fi
    if [ ! -f encoder/infersent2.pkl ];then
        curl -Lo encoder/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl
    fi
else
    echo 'Unknown embedding $embedding , exit'
    exit
fi
cd ../

############################
# prepare Paraphrase model #
############################
# make sure Quora Question Pair train.csv is already downloaded from https://www.kaggle.com/c/quora-question-pairs

if [ ! -f data/quora/train.csv ] ; then
    echo $PWD
    echo "data/quora/train.csv not found, please download first"
    exit
fi 

if [ ! -f data/quora/train.json ];then
    echo ' ******************************'
    echo ' * Preprocess Quora train.csv *'
    echo ' ******************************'

    cd data/quora
    python extract_all.py
    cd ../../
fi

echo ' *****************************'
echo ' * Training paraphrase model *'
echo ' *****************************'
model_type=albert
model=albert-base-v2
batch_size=128
accum_steps=2
lr=3e-5
epoch=3
python run_paraphrase.py \
    --model_type ${model_type} \
    --model_name_or_path  ${model}\
    --output_dir ${evaluation_dir}/paraphrase/ \
    --train_file_path data/quora/train.json \
    --eval_file_path data/quora/dev.json \
    --do_train \
    --do_lower_case \
    --evaluate_during_training \
    --per_gpu_train_batch_size $(($batch_size/${accum_steps}))\
    --gradient_accumulation_steps ${accum_steps} \
    --per_gpu_eval_batch_size 64 \
    --save_steps 1000 \
    --warmup_steps 1000 \
    --learning_rate ${lr} \
    --num_train_epochs ${epoch} \
    --fp16 --fp16_opt_level O2


##########################
# prepare language model #
##########################

echo ' ***************************'
echo ' * Training language model *'
echo ' ***************************'

train_file=data/squad/train.json
dev_file=data/squad/dev.json
for data_mode in q cq
do
    lr=1e-5
    batch_size=32
    warm=0
    if [ $data_mode == q ];then
        epoch=7
        accu_steps=1
    elif [ $data_mode == cq ];then
        epoch=9
        accu_steps=8
    fi

    python run_question_language_model.py train \
            --model_name_or_path gpt2 \
            --train_file_path ${train_file} \
            --dev_file_path ${dev_file} \
            --save_dir ${evaluation_dir}/qlm/${data_mode}/ \
            --data_mode ${data_mode} \
            --epoch ${epoch} \
            --train_batch_size $((${batch_size}/${accu_steps})) \
            --gradient_accumulation_steps ${accu_steps} \
            --eval_batch_size 32 \
            --eval_step 1000 \
            --warmup_steps  ${warm}\
            --learning_rate ${lr}
done

#####################################
# prepare question answering  model #
#####################################
echo ' *************************************'
echo ' * Training Question answering model *'
echo ' *************************************'

model_type=bert
model_name=bert-large-uncased

# train
python run_squad.py --model_type ${model_type} \
    --model_name_or_path ${model_name} \
    --output_dir ${evaluation_dir}/qa/ \
    --train_file data/squad/train.json \
    --predict_file data/squad/dev.json\
    --do_train \
    --per_gpu_train_batch_size 12 \
    --gradient_accumulation_steps 2 \
    --evaluate_during_training \
    --do_lower_case \
    --learning_rate 3e-5 \
    --save_steps 500 \
    --num_train_epochs 3 \
    --fp16 --fp16_opt_level O2
