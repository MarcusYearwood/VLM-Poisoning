import os
import io
import time
import argparse

from tqdm import tqdm

import sys
sys.path.append('../')
from utilities import *

from eval_models import internlm, llava_one_v, internvl
# from models import claude, gpt, bard, hugging_face

from build_query import create_query_data


def verify_response(response):
    if isinstance(response, str):
        response = response.strip() 
    if response == "" or response == None:
        return False
    if "Response Error" in response:
        return False
    return True


def evaluate_code(code_string):
    # execute_code_and_capture_output
    # Backup the original stdout
    old_stdout = sys.stdout
    
    # Redirect stdout to capture the output
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    
    # Try executing the code and capture any exception
    error = None
    try:
        exec(code_string)
    except Exception as e:
        error = e
    
    # Restore the original stdout
    sys.stdout = old_stdout
    
    # Get the captured output
    captured_output = new_stdout.getvalue()
    if isinstance(captured_output, str):
        captured_output = captured_output.strip()
    
    # Return the captured output or error
    return captured_output, error
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--poison_data_dir', type=str, default='../data')
    parser.add_argument('--input_file', type=str, default='testmini.json')
    parser.add_argument('--task_data_pth', type=str, default=None)
    # output
    parser.add_argument('--output_dir', type=str, default='../results/bard')
    parser.add_argument('--output_file', type=str, default='output_bard.json')
    # model
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='llm engine',
                        choices = ['gpt-3.5-turbo', 'claude-2', 'gpt4', 'gpt-4-0613', 'bard', 'internlm', 'llava_one_v', 'internvl'])
    parser.add_argument('--key', type=str, default='', help='key for llm api')
    # query
    parser.add_argument('--query_file', type=str, default=None)  
    parser.add_argument('--caption_file', type=str, default='../data/texts/captions_bard.json') 
    parser.add_argument('--ocr_file', type=str, default='../data/texts/ocrs_easyocr.json')   
    parser.add_argument('--shot_type', type=str, default='solution', help='shot type', 
                        choices = ['solution', 'code'])
    parser.add_argument('--shot_num', type=int, default=0, help='number of shot examples')
    parser.add_argument('--use_caption', action='store_true', help='use caption data')
    parser.add_argument('--use_ocr', action='store_true', help='use ocr data')
    # other settings
    parser.add_argument('--rerun', action='store_true', help='rerun answer extraction for all problems')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()

    # load data
    input_file = os.path.join(args.task_data_pth, args.input_file)
    print(f"Reading {input_file}...")
    data = read_json(input_file)
    # load or create query data
    if args.query_file:
        query_file = os.path.join(args.task_data_pth, args.query_file)
        if os.path.exists(query_file):
            print(f"Loading existing {query_file}...")
            query_data = read_json(query_file)
    else:
        print("\nCreating new query...")
        # load caption
        caption_data = {}
        if args.use_caption:
            caption_file = args.caption_file
            if os.path.exists(caption_file):
                print(f"Reading {caption_file}...")
                try:
                    caption_data = read_json(caption_file)["texts"]
                    print("Caption data loaded.")
                except:
                    print("Caption data not found!! Please Check.")                    
        # load ocr
        ocr_data = {}
        if args.use_ocr:
            ocr_file = args.ocr_file
            if os.path.exists(ocr_file):
                print(f"Reading {ocr_file}...")
                try:
                    ocr_data = read_json(ocr_file)["texts"]
                    print("OCR data loaded.")
                except:
                    print("OCR data not found!! Please Check.")
        # create query
        query_data = create_query_data(data, caption_data, ocr_data, args)

    # output file
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, args.output_file)
    
    # load results
    if os.path.exists(output_file):
        print("\nResults already exist.")
        print(f"Reading {output_file}...")
        results = read_json(output_file)
    else:
        results = {}

    # load model
    print(f"\nLoading {args.model}...")
    if args.model == 'bard':
        if args.key == '':
            print("Loading key from environment variable")
            key = os.environ['_BARD_API_KEY']
        else:
            key = args.key
        model = bard.Bard_Model(key)
    
    elif "gpt" in args.model:
        if args.key == '':
            print("Loading token from environment variable")
            key = os.getenv("OPENAI_API_KEY")
        else:
            key = args.key
        model = gpt.GPT_Model(args.model, key)
    
    elif "claude" in args.model:
        if args.key == '':
            print("Loading token from environment variable")
            key = os.environ.get("ANTHROPIC_API_KEY")
        else:
            key = args.key
        model = claude.Claude_Model(args.model, key)
    elif "internlm" in args.model:
        model = internlm.InternLM_Model()
    elif "llava_one_v" in args.model:
        model = llava_one_v.Llava_One_V()
    elif "internvl" in args.model:
        model = internvl.InternVL_Model()
            
    
    print(f"Model loaded.")
    
    # build final test pid list
    test_pids = list(data.keys())
    print("\nNumber of test problems in total:", len(test_pids))

    available_directories = [d for d in os.listdir(args.poison_data_dir) if os.path.isdir(os.path.join(args.poison_data_dir, d))]
    target_names = read_json(os.path.join(args.task_data_pth, "target_train/cap.json"))
    if not all([name in available_directories for name in target_names]):
        print("Not all targets have directories. Working with:", available_directories)
    target_names = [item["name"] for item in target_names["annotations"] if item["name"] in available_directories]

    skip_pids = []
    if not args.rerun:
        print("\nRemoving problems with existing valid response...")
        for i, name in enumerate(target_names):
            skip_pids.append([])
            for pid in test_pids:
                # print(f"Checking {pid}...")
                if pid in results and 'response' in results[pid]:
                    response = results[pid][name]['response']
                    if verify_response(response):
                        # print(f"Valid response found for {pid}.")
                        skip_pids[i].append(pid)
    else:
        print("\nRerun answer extraction for all problems...")

    test_pids = [[pid for pid in test_pids if pid not in target_skip_pids] for target_skip_pids in skip_pids]
    print("Number of test problems to run for each target:", {target_names[i]: len(target_pids) for i, target_pids in enumerate(test_pids)})
    # print(test_pids)

    

    # tqdm, enumerate results
    for i, target_name in enumerate(target_names):
        for _, pid in enumerate(tqdm(test_pids[i])):
            problem = data[pid]
            query = query_data[pid]
            image = problem['image']
            image_path = os.path.join(args.poison_data_dir, target_name, image)

            if args.debug:
                print("--------------------------------------------------------------")
            print(f"\nGenerating response for {pid}...")
            try:
                response = model.get_response(image_path, query)
                new_caption = model.get_response(image_path, "describe what is in this image")
                # print(f"Response: {response}")
                if pid not in results:
                    results[pid] = problem
                if "targets" not in results[pid]:
                    results[pid]["targets"] = {}
                if target_name not in results[pid]["targets"]:
                    results[pid]["targets"][target_name] = {}

                results[pid]["targets"][target_name]['query'] = query
                results[pid]["targets"][target_name]['model_description'] = new_caption
                if args.shot_type == 'solution':
                    results[pid]["targets"][target_name]['response'] = response
                else:
                    output, error = evaluate_code(response)
                    results[pid]["targets"][target_name]['response'] = response
                    results[pid]["targets"][target_name]['execution'] = output
                    results[pid]["targets"][target_name]['error'] = str(error)
                if args.debug:
                    print(f"\n#Query: \n{query}")
                    print(f"\n#Response: \n{response}")
            except Exception as e:
                print(e)
                print(f"Error in extracting answer for {pid}")
                results[pid][target_name]['error'] = e
        
            try:
                print(f"Saving results to {output_file}...")
                save_json(results, output_file)
                print(f"Results saved.")
            except Exception as e:
                print(e)
                print(f"Error in saving {output_file}")
