import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes as bnb
from accelerate import Accelerator
from accelerate.utils import gather_object
from parallelformers import parallelize
import warnings

# build gpt class
class InternLM_Model():
    def __init__(self, model_name="internlm/internlm-xcomposer2d5-7b", num_beams=3, patience=1000000, sleep_time=0, d_type=torch.float16):
        warnings.filterwarnings("ignore")
        # self.accelerator = Accelerator()

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,  # Use 8-bit quantization
            llm_int8_threshold=200.0  # Adjust threshold for 8-bit quantization if necessary
        )

        self.model = AutoModel.from_pretrained(
            model_name, 
            torch_dtype=d_type, 
            # quantization_config=quantization_config,
            trust_remote_code=True, 
            device_map='auto'
        )
        # self.model = self.accelerator.prepare(self.model)
        # parallelize(self.model, num_gpus=2, fp16=True, verbose='detail')
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model.tokenizer = self.tokenizer
        self.num_beams = num_beams
        self.patience = patience
        self.sleep_time = sleep_time
        self.d_type = d_type

    def get_response(self, image_path, user_prompt):
        query = user_prompt
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=self.d_type):
            # with self.accelerator.autocast():
                response, his = self.model.chat(self.tokenizer, query, [image_path], do_sample=False, num_beams=self.num_beams, use_meta=True)
        prediction = response.strip()
        return prediction
