input_file=$1
result_file=$2
selection=$3
num=1
evaluation_dir=save/evaluation

golden=tmp/golden.txt
generated=tmp/generated.txt
score_file=tmp/generated.json
output_file=tmp/all_result.json

mkdir tmp

python selection.py ${input_file} ${selection} ${num}

# bleu, meteor, rouge-l
cd qgevalcap
python eval.py -out ../${generated} -src ../${golden} -tgt ../${golden} -o ../${output_file}
cd ../

# bertscore
python bertscore_evaluate.py --golden ${golden} --generated ${generated} --output_file ${output_file}

# infersent
cd InferSent
python eval.py --golden ../${golden} --generated ../${generated} --output_file ../${output_file}
cd ../

# paraphrase
python run_paraphrase.py \
    --model_type labert \
    --model_name_or_path ${evaluation_dir}/paraphrase/checkpoint \
    --output_dir ${evaluation_dir}/paraphrase \
    --tokenizer_name albert-base-v2 \
    --score_file_path ${golden} ${generated} \
    --output_file ${output_file} \
    --do_score \
    --do_lower_case \
    --per_gpu_eval_batch_size 32

# ppl(q), ppl(q|c)
for data_mode in q cq
do
    python run_question_language_model.py eval \
		--model_name_or_path ${evaluation_dir}/qlm/${data_mode}/checkpoint \
        --dev_file_path ${score_file} \
		--data_mode ${data_mode} \
        --output_file ${output_file} \
        --eval_batch_size 8 \
        --is_qg
done

# qa
python run_squad.py --model_type bert \
    --model_name_or_path ${evaluation_dir}/qa/checkpoint \
    --output_dir ${evaluation_dir}/qa/ \
    --predict_file ${score_file}\
    --output_file ${output_file} \
    --do_eval \
    --do_lower_case \
    --evaluate_is_qg

# convert evaluation result to .csv file
python write_to_csv.py ${output_file} ${result_file}

# remove tmp dir
rm tmp
