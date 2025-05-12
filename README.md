# Arabic to English Neural Machine Translation

This project focuses on that translates Arabic news text into English using a pretrained MarianMT model. The system is trained and evaluated on the Global Voices dataset.

---

## 📚 Dataset

We use the [Global Voices Arabic-English parallel corpus](https://opus.nlpl.eu/GlobalVoices/ar&en/v2018q4/GlobalVoices), which contains sentence-aligned news articles in Modern Standard Arabic (Fusha) and English.

- Total sentence pairs: ~63,000  
- Format: Each line is a sentence, aligned across Arabic and English files.

---

## 🧠 Model

We fine-tuned the [`Helsinki-NLP/opus-mt-ar-en`](https://huggingface.co/Helsinki-NLP/opus-mt-ar-en) transformer model for translation. 

---
