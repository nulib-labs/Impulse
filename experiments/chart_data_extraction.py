from transformers import AutoModel, AutoTokenizer
from pdf2image import convert_from_path

tokenizer = AutoTokenizer.from_pretrained('kppkkp/OneChart', trust_remote_code=True, use_fast=False, padding_side="right", device_map="cpu")
model = AutoModel.from_pretrained('kppkkp/OneChart', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cpu')
model = model.eval().cpu()

# input your test image
image_file = './out0.jpg'
res = model.chat(tokenizer, image_file, reliable_check=True)
print(res)
