# date : 2020/8/11
# Chapter 4

# This script run question generation on half of squad training set, 
#   and generating question-answer pairs on the other half of squad.
# Train question answering model using generated question-answer pairs.

# step 1 : train QA model and Language model on half squad training set     -> prepare scoring models

# step 2-1 : train QG model on half squad                                   -> training
# step 2-2 : train AG model on half squad

# step 3-1 : generate answer on the other half of squad                      -> inference
# step 3-2 : inference using different search algorithms with beam width 50 

# step 4 : get p(q|c) and p(a|q,c)                                          -> scoring
# step 5 : clustering question using sentence embedding                     -> clustering
# step 6 : train QA model                                                   -> evaluation

mode=squad
if [ $mode == squad ];then
    train_file=data/squad/QG_train.json
    dev_file=data/squad/QG_dev.json
    gen_file=data/squad/QG_gen.json
elif [ $mode == wiki ];then
    train_file=data/squad/train.json
    def_file=data/squad/QG_dev.json
    gen_file=data/wiki/wiki.json

save_dir=save/chapter4-2
result_dir=result/chapter4-2

beam_size=50
answer_file=${result_dir}/answers.json
inference_file=${result_dir}/beam-${beam_size}.json
qlm_scoring_file=${result_dir}/beam-${beam_size}_qlm.json
qa_scoring_file=${result_dir}/beam-${beam_size}_qa.json
scoring_file=${result_dir}/beam-${beam_size}_scoring.json

cluater=use # or cluster=infersent
clustering_file=${result_dir}/beam-${beam_size}_clustering-${cluster}.json

##########################
# Prepare scoring models #
##########################
echo ' ***************************'
echo ' * Training language model *'
echo ' ***************************'

lr=1e-5
batch_size=32
warm=0
data_mode=cq
epoch=9
accu_steps=4

python run_question_language_model.py train \
        --model_name_or_path gpt2 \
        --train_file_path ${train_file} \
        --dev_file_path ${dev_file} \
        --save_dir ${save_dir}/scoring/qlm/${data_mode}/ \
        --data_mode ${data_mode} \
        --epoch ${epoch} \
        --train_batch_size $batch_size \
        --gradient_accumulation_steps $((${batch_size}/${accumu_steps})) \
        --eval_batch_size 32 \
        --eval_step 1000 \
        --warmup_steps  ${warm}\
        --learning_rate ${lr}

echo ' *************************************'
echo ' * Training Question answering model *'
echo ' *************************************'

model_type=bert
model_name=bert-base-uncased

# train
python run_squad.py --model_type ${model_type} \
    --model_name_or_path ${model_name} \
    --output_dir ${save_dir}/scoring/qa/ \
    --train_file ${train_file} \
    --predict_file ${dev_file}\
    --do_train \
    --per_gpu_train_batch_size 12 \
    --gradient_accumulation_steps 2 \
    --evaluate_during_training \
    --do_lower_case \
    --learning_rate 3e-5 \
    --save_steps 500 \
    --num_train_epochs 3 \
    --fp16 --fp16_opt_level O2

# ------------------------------------------------

#########################
# Train AG and QG model #
#########################
echo " ******************** "
echo " * Training AG model* "
echo " ******************** "

lr=3e-5
epoch=3
python run_answer_generation.py \
    --model_name_or_path bert-base-uncased \
    --output_dir ${save_dir}/ag \
    --train_file ${train_file} \
    --predict_file ${dev_file}\
    --do_train \
    --per_gpu_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --evaluate_during_training \
    --do_lower_case \
    --learning_rate ${lr} \
    --save_steps 500 \
    --num_train_epochs ${epoch} \
    --balance \
    --fp16 --fp16_opt_level O2

echo " ******************** "
echo " * Training QG model* "
echo " ******************** "

lr=5e-5
warm=500
epoch=4
data_mode=acp

python run_question_generation.py train \
		--model_name_or_path gpt2 \
        --train_file_path ${train_file} \
        --dev_file_path ${dev_file} \
        --save_dir ${save_dir}/qg \
		--data_mode ${data_mode} \
        --epoch ${epoch} \
        --warmup_steps  ${warm}\
        --learning_rate ${lr} \
        --train_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --eval_batch_size 16 \
		--eval_step 1000 \
        --evaluate_during_training \
        --fp16 --fp16_opt_level O2

# ------------------------------------------------

#############
# Inference #
#############

echo " ************* "
echo " * Inference * "
echo " ************* "

# generate answer by ner or answer generation model
# if want to extract  NER as answer, run folllowing script
#   python extract_ner.py ${gen_file} ${answer_file}

python run_answer_generation.py \
    --model_name_or_path  ${save_dir}/ag/checkpoint\
    --output_dir ${save_dir}/ag \
    --train_file  ${train_file}\
    --gen_file ${gen_file}\
    --output_file ${answer_file} \
    --do_gen\
    --do_lower_case \
    --start_sample_num 20\
    --end_sample_num 2 \
    --sample_num 10 \

# change inference arguments to use different search algorithms
python run_question_generation.py gen \
		--model_name_or_path ${save_dir}/checkpoint \
        --dev_file_path ${answer_file} \
		--data_mode ${data_mode} \
        --eval_batch_size 8 \
        --inference beam \
        --beam_width 50 \
        --output_file ${inference_file} \
        --ag

# ------------------------------------------------

echo " *********** "
echo " * Scoring * "
echo " *********** "

echo " Scoring by question language model"
python run_question_language_model.py score \
		--model_name_or_path ${save_dir}/scoring/qlm/cq/checkpoint \
        --score_file_path ${inference_file} \
        --output_file ${qlm_scoring_file} \
        --eval_batch_size 8 \
        --is_qg

echo " Scoring by question answering  model"
python run_squad.py --model_type ${model_type} \
    --model_name_or_path ${save_dir}/scoring/qa/checkpoint \
    --score_file ${inference_file}\
    --output_file ${qa_scoring_file}
    --do_score \
    --do_lower_case \
    --score_is_qg

echo " Mrege scoring file"
python merge_scoring_file_and_check.py ${qlm_scoring_file} ${qa_scoring_file} ${scoring_file}

# ------------------------------------------------
echo " ************** "
echo " * Clustering * "
echo " ************** "
python cluater.py ${scoring_file} ${clustering_file} use

# ------------------------------------------------

echo " ********************* "
echo " * Training QA model * "
echo " ********************* "
model_type=bert
model_name=bert-base-uncased

if [ $mode == squad ];then
    # when generated qa pairs from squad, using QG_train.json and generated qa-pairs together
    python run_squad.py --model_type ${model_type} \
        --model_name_or_path ${model_name} \
        --output_dir ${save_dir}/generated_squad/ \
        --train_file ${clustering_file} \
        --predict_file ${dev_file}\
        --squad_file ${train_file} \
        --sampler balance \
        --balance_ratio 1 1 \
        --do_train \
        --per_gpu_train_batch_size 12 \
        --gradient_accumulation_steps 2 \
        --evaluate_during_training \
        --do_lower_case \
        --learning_rate 3e-5 \
        --save_steps 500 \
        --num_train_epochs 3 \
        --train_is_qg \
        --fp16 --fp16_opt_level O2

elif [ $mode == wiki ];then
    # use generated qa-pairs to pretrain qa model
    python run_squad.py --model_type ${model_type} \
        --model_name_or_path ${model_name} \
        --output_dir ${save_dir}/generated_wiki/pretrain/ \
        --train_file ${clustering_file} \
        --predict_file ${dev_file}\
        --do_train \
        --train_is_qg \
        --per_gpu_train_batch_size 12 \
        --gradient_accumulation_steps 2 \
        --evaluate_during_training \
        --do_lower_case \
        --learning_rate 3e-5 \
        --save_steps 500 \
        --num_train_epochs 3 \
        --fp16 --fp16_opt_level O2

    # fine-tune using training data
    python run_squad.py --model_type ${model_type} \
        --model_name_or_path ${save_dir}/generated_wiki/pretrain/checkpoint \
        --output_dir ${save_dir}/generated_wiki/finetune/ \
        --train_file ${train_file} \
        --predict_file ${dev_file}\
        --do_train \
        --per_gpu_train_batch_size 12 \
        --gradient_accumulation_steps 2 \
        --evaluate_during_training \
        --do_lower_case \
        --learning_rate 3e-5 \
        --save_steps 500 \
        --num_train_epochs 3 \
        --fp16 --fp16_opt_level O2
