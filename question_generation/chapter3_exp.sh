# date : 2020/8/9
# Chapter 3
# This script will train a question generation model using default arguments, and run evaluation automatically.

# step 1 : traing QG model                                -> training
# step 2 : inference using beam search with beam width 10 -> inference
# step 3 : get p(q|c) and p(a|q,c)                        -> scoring
# step 4 : compare four selection methods                 -> evaluation

train_file=data/squad/train.json
dev_file=data/squad/dev.json

lr=5e-5
warm=500
epoch=4
data_mode=acp

save_dir=save/chapter3/qg/${epoch}_${lr}_${warm}_${data_mode}
inference_file=result/chapter3/beam-10.json
qlm_scoring_file=result/chapter3/beam-10_qlm.json
qa_scoring_file=result/chapter3/beam-10_qa.json
scoring_file=result/chapter3/beam-10_scoring.json
result_file=result/chapter3/beam-10

echo " ************ "
echo " * Training * "
echo " ************ "

python run_question_generation.py train \
		--model_name_or_path gpt2 \
        --train_file_path ${train_file} \
        --dev_file_path ${dev_file} \
        --save_dir ${save_dir} \
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

# -------------------------------------------------------------

echo " ************* "
echo " * Inference * "
echo " ************* "

python run_question_language_model.py gen \
		--model_name_or_path ${save_dir}/checkpoint \
        --dev_file_path ${dev_file} \
		--data_mode ${data_mode} \
        --eval_batch_size 8 \
        --inference beam \
        --beam_width 10 \
        --output_file ${inference_file}

# -------------------------------------------------------------

echo " *********** "
echo " * Scoring * "
echo " *********** "

echo " Scoring by question language model"
python run_question_language_model.py score \
		--model_name_or_path ${evaluation_dir}/qlm/cq/checkpoint \
        --score_file_path ${inference_file} \
        --output_file ${qlm_scoring_file} \
        --eval_batch_size 8 \
        --is_qg

echo " Scoring by question answering  model"
python run_squad.py --model_type ${model_type} \
    --model_name_or_path ${evaluation_dir}/qa/checkpoint \
    --score_file ${inference_file}\
    --output_file ${qa_scoring_file}
    --do_score \
    --do_lower_case \
    --score_is_qg

echo " Mrege scoring file"
python merge_scoring_file_and_check.py ${qlm_scoring_file} ${qa_scoring_file} ${scoring_file}

# -------------------------------------------------------------

echo " ************** "
echo " * Evaluation * "
echo " ************** "
for selection in qg qg_ln a qa
do
    bash chapter3_evaluation.sh ${scoring_file} ${result_file}_${selection}.csv ${selection}
done

