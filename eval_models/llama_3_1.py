import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import bitsandbytes as bnb

class LLaMA:
    def __init__(self):
        # Load the model and tokenizer
        model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        # model_name = "meta-llama/Meta-Llama-3.1-405B-Instruct"

        # quantization_config = BitsAndBytesConfig(
        #     load_in_8bit=True,  # Use 8-bit quantization
        #     llm_int8_threshold=200.0  # Adjust threshold for 8-bit quantization if necessary
        # )

        # Use quantization for memory efficiency
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # quantization_config=quantization_config,
            device_map="auto",  # Automatically distribute across available GPUs
            torch_dtype=torch.float16,  # Mixed precision for faster inference
            low_cpu_mem_usage=True  # Avoid using too much CPU memory during loading
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)


        # Define the pipeline for text generation
        self.generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def get_chat_response(self, prompt, max_tokens=256):
        messages = [
            {"role": "system", "content": "You are tasked with extracting only the final numerical or algebraic answer from each sentence. Output only the answer, with no explanations or additional text."},
            {"role": "user", "content": prompt},
        ]

        outputs = self.generator(messages, max_new_tokens=max_tokens, do_sample=False)

        response = outputs[0]["generated_text"][-1]["content"].strip()

        return response
