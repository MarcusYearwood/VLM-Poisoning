import time
import openai
from openai import OpenAI
import base64

# Updated GPT class using OpenAI SDK v1.x
class GPT_Model():
    def __init__(self, model="gpt-3.5-turbo", api_key="", temperature=0, max_tokens=1024, n=1, patience=1000000, sleep_time=0):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n = n
        self.patience = patience
        self.sleep_time = sleep_time
        self.client = OpenAI(api_key=self.api_key)

    def get_response(self, image_path=None, user_prompt=""):
        patience = self.patience
        max_tokens = self.max_tokens
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            data_url = f"data:image/png;base64,{base64_image}"
        # messages = [
        #     {"role": "user", "content": user_prompt},
        # ]
        messages = [
        {
            "role": "user",
            "content": [
                { "type": "text", "text": user_prompt },
                { "type": "image_url", "image_url": { "url": data_url } }
            ]
        }
        ]
        while patience > 0:
            patience -= 1
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    n=self.n
                )

                if self.n == 1:
                    prediction = response.choices[0].message.content.strip()
                    if prediction:
                        return prediction
                else:
                    prediction = [choice.message.content.strip() for choice in response.choices]
                    if prediction[0]:
                        return prediction

            except Exception as e:
                print("Exception:", e)
                if "Please reduce the length of the messages or completion" in str(e):
                    max_tokens = int(max_tokens * 0.9)
                    print("!! Reduce max_tokens to", max_tokens)
                if max_tokens < 8:
                    return ""
                if self.sleep_time > 0:
                    time.sleep(self.sleep_time)

        return ""
