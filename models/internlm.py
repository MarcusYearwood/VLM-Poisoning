def get_image_encoder_internlm():
    model_name="internlm/internlm-xcomposer2d5-7b"
    num_beams=3
    d_type=torch.float16
    warnings.filterwarnings("ignore")

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Use 8-bit quantization
        llm_int8_threshold=200.0  # Adjust threshold for 8-bit quantization if necessary
    )

    model = AutoModel.from_pretrained(
        model_name, 
        torch_dtype=d_type, 
        # quantization_config=quantization_config,
        trust_remote_code=True, 
        device_map='auto'
    )

def encode_image_internlm(model, images):
    image = __resize_img__(image)
    image = model.vis_processor(image).unsqueeze(0).cuda()
    image = torch.nn.functional.interpolate(image.float(), size=(560,560), mode='bicubic',).to(torch.float16)
    image_embeds = model.encode_img(image)
    return image_embeds