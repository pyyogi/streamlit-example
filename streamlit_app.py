import streamlit as st
import torchaudio
import torch
import whisper
from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
import io

language_model = whisper.load_model("medium")
emotion_model = HubertForSequenceClassification.from_pretrained(
    "xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned")
emotion_model.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
emotion_model.num2emotion = {0: 'neutral', 1: 'angry', 2: 'positive', 3: 'sad', 4: 'other'}

st.title("Распознавание речи и эмоций")

audio_file = st.file_uploader("Выберите аудиофайл", type=["wav", "mp3"])

duration = st.slider("Выберите продолжительность записи (секунды):", 1, 10, 3)

start_recording = st.button("Начать запись")

if start_recording:
    st.info("Запись...")

    audio_data = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype=np.int16)
    sd.wait()

    st.success("Запись завершена!")

    audio_segment = AudioSegment(
        audio_data.tobytes(),
        frame_rate=16000,
        sample_width=audio_data.dtype.itemsize,
        channels=1
    )

    audio_bytes = audio_segment.export(format="wav").read()

    st.audio(audio_bytes, format="audio/wav")
    audio_array = np.array(audio_segment.get_array_of_samples())
    waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
    transform = torchaudio.transforms.Resample(16000, 16000)
    waveform = transform(waveform)

    inputs = emotion_model.feature_extractor(
        waveform,
        sampling_rate=emotion_model.feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
        max_length=16000 * 10,
        truncation=True
    )

    inputs['input_values'] = inputs['input_values'].view(1, 1, -1)

    logits = emotion_model(inputs['input_values'][0]).logits
    predictions = torch.argmax(logits, dim=-1)
    predicted_emotion = emotion_model.num2emotion[predictions.numpy()[0]]

    st.write("Предсказанная эмоция: ", predicted_emotion)

    waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)

    transform = torchaudio.transforms.Resample(16000, 16000)
    waveform = transform(waveform)

    waveform = whisper.pad_or_trim(waveform)

    mel = whisper.log_mel_spectrogram(waveform).to(language_model.device)

    languages = language_model.detect_language(mel)
    probs = {str(i): prob for i, prob in enumerate(languages)}
    max_language = max(probs["1"][0], key=lambda x: probs["1"][0][x])
    st.write("Язык: ", max_language)

    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(language_model, mel, options)

    transcripted_text = result[0].text
    st.write("Распознанный текст: ", transcripted_text)

if audio_file:
    waveform, sample_rate = torchaudio.load(audio_file, normalize=True, num_frames=int(duration * 16000))
    st.audio(audio_file, format="audio/wav")
    waveform, sample_rate = torchaudio.load(audio_file, normalize=True, num_frames=int(duration * 16000))
    transform = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = transform(waveform)

    inputs = emotion_model.feature_extractor(
        waveform,
        sampling_rate=emotion_model.feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
        max_length=16000 * 10,
        truncation=True
    )

    logits = emotion_model(inputs['input_values'][0]).logits
    predictions = torch.argmax(logits, dim=-1)
    predicted_emotion = emotion_model.num2emotion[predictions.numpy()[0]]

    st.write("Предсказанная эмоция: ", predicted_emotion)

    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(language_model.device)

    _, probs = language_model.detect_language(mel)
    language = max(probs, key=probs.get)
    st.write("Язык: ", language)

    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(language_model, mel, options)
    transcripted_text = result.text

    st.write("Распознанный текст: ", transcripted_text)
