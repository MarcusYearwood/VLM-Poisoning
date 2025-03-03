cd ../evaluation

task_name=mini_MathVista_grid

##### multimodal-internvl #####
# generate solution
python generate_response.py \
--model internvl \
--poison_data_dir ../data/poisons/$task_name \
--task_data_pth ../data/task_data/$task_name \
--output_dir ../results/internvl \
--output_file output_internvl.json \
--input_file questions.json

# extract answer
# python3 extract_answer.py \
# --output_dir ../results/internvl \
# --output_file output_internvl.json 


# # calculate score
# python3 calculate_score.py \
# --output_dir ../results/internvl \
# --output_file output_internvl.json \
# --score_file scores_internvl.json
