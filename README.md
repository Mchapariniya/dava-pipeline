# DAVA — Data Audio/Video Analysis

Local, end-to-end pipeline that turns a recording into a transcribed, diarized,
annotated and **searchable** document — with an ELAN export for linguistic work
and a semantic, cross-lingual (DE/EN/FR) Q&A that cites audio timestamps. No
external APIs; generation runs on a local Ollama model.

**Six stages:** preprocess audio → transcribe/diarize/align (WhisperX) →
analyse (NER · sentiment · emotion · topics) → ELAN export → one Streamlit
dashboard → ask the document (RAG).

## Quick start

```bash
# 1. environments (see ANALYSIS.md for details)
conda create -n pipeline python=3.10 -y && conda activate pipeline
pip install -r requirements_pipeline.txt && conda install -c conda-forge ffmpeg -y

conda create -n dava_env python=3.10 -y && conda activate dava_env
pip install -r requirements_dava_env.txt && python -m spacy download xx_ent_wiki_sm

# 2. preprocess + transcribe  (env: pipeline)
python pipeline/preprocess_audio_for_asr.py --input data/x.mp3 --output data/x_16k.wav --device cuda
CUDA_VISIBLE_DEVICES=3 python pipeline/process_single_file_pipeline_AG.py \
    --episode_path data/x_16k.wav --out_dir ./output --gpu_index 0 \
    --language de --asr_backend openai --hf_token <HF_TOKEN>

# 3. explore everything in one dashboard  (env: dava_env)
streamlit run dashboard/app.py
```

Full instructions, the Ollama setup, output format and troubleshooting are in
**[ANALYSIS.md](ANALYSIS.md)**.
