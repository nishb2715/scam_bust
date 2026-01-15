import pandas as pd
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'http\S+', ' URL ', text)
    text = re.sub(r'\b\d{4,6}\b', ' OTP ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_all():
    sms = pd.read_csv("../data/public_sms.csv")
    sms["text"] = sms["message_text"]
    sms["modality"] = "sms"

    wa = pd.read_csv("../data/public_whatsapp.csv")
    wa["text"] = wa["conversation_text"]
    wa["modality"] = "whatsapp"

    calls = pd.read_csv("../data/public_calls.csv")
    calls["text"] = calls["call_transcript"]
    calls["modality"] = "call"

    audio = pd.read_csv("../data/public_audio_transcripts.csv")
    audio["text"] = audio["audio_transcript_text"]
    audio["modality"] = "audio"

    df = pd.concat([sms, wa, calls, audio])
    df = df[["text", "modality", "is_scam", "severity"]]
    df["clean_text"] = df["text"].apply(clean_text)

    df.to_csv("processed_data.csv", index=False)

if __name__ == "__main__":
    load_all()
