# portable_ai_person/app.py

import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sounddevice as sd
import scipy.io.wavfile as wav
import whisper
import os
import subprocess
from TTS.api import TTS
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load models
st.session_state.setdefault("model", SentenceTransformer("all-MiniLM-L6-v2"))
st.session_state.setdefault("whisper_model", whisper.load_model("base"))
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto")
    return tokenizer, model

tokenizer, llm_model = load_llm()

tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# Load or create memories.json
def create_memories(input_texts, json_path="data/memories.json"):
    model = st.session_state.model
    embeddings = model.encode(input_texts)
    memory_data = [
        {"text": text, "embedding": emb.tolist()} for text, emb in zip(input_texts, embeddings)
    ]
    with open(json_path, "w") as f:
        json.dump({"memories": memory_data}, f, indent=2)

def load_memories(path):
    if not os.path.exists(path):
        create_memories([
            "I loved painting mountains in silence.",
            "Sunsets by the river were my escape."
        ], path)
    with open(path, "r") as f:
        data = json.load(f)["memories"]
    texts = [m["text"] for m in data]
    vectors = np.array([m["embedding"] for m in data])
    return texts, vectors

memory_texts, memory_vectors = load_memories("data/memories.json")

def search_memory(query):
    embed = st.session_state.model.encode([query])
    sims = cosine_similarity(embed, memory_vectors)[0]
    best_idx = np.argmax(sims)
    return memory_texts[best_idx]

def generate_response(memory, question):
    prompt = f"""<bos><start_of_turn>user\nMemory: {memory}\nQ: {question}\n<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(llm_model.device)
    outputs = llm_model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("<start_of_turn>model\n")[-1].strip()

def generate_voice(text, wav_path="data/voices/response.wav"):
    tts_model.tts_to_file(text=text, speaker_wav="data/voices/sample.wav", file_path=wav_path)

def generate_video(img_path="data/images/face.jpg", audio_path="data/voices/response.wav", out_path="data/videos/response.mp4"):
    os.makedirs("data/videos", exist_ok=True)
    cmd = ["python", "SadTalker/inference.py", "--driven_audio", audio_path, "--source_image", img_path, "--result_dir", os.path.dirname(out_path)]
    subprocess.run(cmd)

# UI
st.title("🧠 Revived AI Person (Offline)")
st.write("Talk to a memory-based virtual person")

if st.button("🎙️ Record Question (5 sec)"):
    fs = 16000
    duration = 5
    st.write("Recording...")
    recording = sd.rec(int(fs * duration), samplerate=fs, channels=1)
    sd.wait()
    wav.write("input.wav", fs, recording)
    st.success("Recorded!")

    result = st.session_state.whisper_model.transcribe("input.wav")
    user_question = result["text"]
    st.audio("input.wav")
    st.write("🧑 You asked:", user_question)

    matched_memory = search_memory(user_question)
    st.write("📌 Matched memory:", matched_memory)

    reply = generate_response(matched_memory, user_question)
    st.write("🧠 AI Response:", reply)

    generate_voice(reply)
    st.audio("data/voices/response.wav")

    if os.path.exists("data/images/face.jpg"):
        generate_video()
        st.video("data/videos/response.mp4")

st.markdown("---")
st.subheader("📝 Add Memories")
new_mem = st.text_area("Enter one or more memories (separated by new lines):")
if st.button("➕ Add to memories"):
    if new_mem.strip():
        lines = [line.strip() for line in new_mem.split("\n") if line.strip()]
        create_memories(memory_texts + lines)
        st.success("Memories added!")
