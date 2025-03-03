cd ../evaluation

##### multimodal-internlm #####
# generate solution
python3 generate_response.py \
--model internlm \
--output_dir ../results/internlm \
--output_file output_internlm.json \
--input_file testsupermini.json

# extract answer
python3 extract_answer.py \
--output_dir ../results/internlm \
--output_file output_internlm.json 


# calculate score
python3 calculate_score.py \
--output_dir ../results/internlm \
--output_file output_internlm.json \
--score_file scores_internlm.json
