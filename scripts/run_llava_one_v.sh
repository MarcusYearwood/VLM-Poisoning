# cd ../evaluation

# ##### multimodal-llava_one_v #####
# # generate solution
# python3 generate_response.py \
# --model llava_one_v \
# --output_dir ../results/llava_one_v \
# --output_file output_llava_one_v.json \ 
# --input_file testsupermini.json

cd ../evaluation

task_name=mini_MathVista_grid

##### multimodal-llava_one_v #####
# generate solution
python generate_response.py \
--model llava_one_v \
--poison_data_dir ../data/poisons/$task_name \
--task_data_pth ../data/task_data/$task_name \
--output_dir ../results/llava_one_v \
--output_file output_llava_one_v.json \
--input_file questions.json

# # extract answer
# python3 extract_answer.py \
# --output_dir ../results/llava_one_v \
# --output_file output_llava_one_v.json 

# # calculate score
# python3 calculate_score.py \
# --output_dir ../results/llava_one_v \
# --output_file output_llava_one_v.json \
# --score_file scores_llava_one_v.json 
