from transformers import Qwen2ForCausalLM, AutoTokenizer
from transformers_stream_generator import init_stream_support

init_stream_support()

repo_id = "Monor/Qwen1.5-0.5B-h-world"
tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
model = Qwen2ForCausalLM.from_pretrained(
    repo_id, device_map="auto", trust_remote_code=True
).eval()


import streamlit as st

st.set_page_config(layout="wide")
st.title("18禁小说生成器")

recurrent = st.sidebar.checkbox("循环生成", value=True)
max_tokens = st.sidebar.text_input("最大字数（粗略)", value=128)
prompt = st.sidebar.text_area("起始文本", value="小青突然来到了我的房间")
resultbox = st.empty()
resultbox.text_area(" ", height=400)
countbox = st.empty()
countbox.text("生成长度：0")

if st.sidebar.button("开始生成"):
    max_tokens = int(max_tokens)
    if recurrent:  # 循环生成，每次只生成32个字
        for _ in range(max_tokens // 32):
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = inputs.to(model.device)
            generator = model.generate(
                **inputs,
                max_new_tokens=32,
                do_stream=True,
                do_sample=True,
                # top_k=30,
                # top_p=0.85,
                temperature=0.95,
                # repetition_penalty=1.2,
                # early_stopping=True,
            )
            content = prompt
            for token in generator:
                word = tokenizer.decode(token, skip_special_tokens=True)
                content += word
                resultbox.text_area(" ", content, height=400)
                countbox.text(f"生成长度：{len(content)}")
            prompt = content
    else:  # 单次直接生成max_tokens个字
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(model.device)
        generator = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_stream=True,
            do_sample=True,
            # top_k=30,
            # top_p=0.85,
            temperature=0.95,
            # repetition_penalty=1.2,
            # early_stopping=True,
        )
        content = prompt
        for token in generator:
            word = tokenizer.decode(token, skip_special_tokens=True)
            content += word
            resultbox.text_area("", content, height=400)
            countbox.text(f"生成长度：{len(content)}")
