import os

import torch
from IPython.display import Markdown, display, update_display
from google.colab import drive
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from utils import download_audio, is_valid_audio

def transcribe(audio_file=None, audio_url=""):
    """
    Transcribes spoken audio from either an uploaded file or a URL using the Whisper model.

    Args:
        audio_file (str or None): Local path to uploaded audio file (optional).
        audio_url (str): URL or Google Drive share link to an audio file (optional).

    Returns:
        str: Transcribed text or error message.
    """
    # file_path = None
    try:
        if audio_file is not None:
            file_path = audio_file
        elif audio_url:
            file_path, error = download_audio(audio_url)
            if error:
                return f"Error: {error} invalid url"
        else:
            return "Please upload a file or enter an audio URL."

        if not is_valid_audio(file_path):
            return "The audio file is invalid or not supported. Try uploading a different file."

        # Specifies the pre-trained Whisper Medium model hosted on Hugging Face. This model is capable of converting speech into text.
        audio_model = "openai/whisper-medium"
        speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(audio_model, torch_dtype=torch.float16,
                                                                 low_cpu_mem_usage=True, use_safetensors=True)
        speech_model.to('cuda')  # move the model to GPU

        # process the audio using autoprocessor. This uses feature extractor to extract the text and a tokenizer to process the text into a format useful to the model
        processor = AutoProcessor.from_pretrained(audio_model)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=speech_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch.float16,
            device='cuda')

        # return_timestamps = True, is used when the audio is longer than 3000 mel, language =en is used to translate the transcripttion to English
        result = pipe(file_path, return_timestamps=True)
        transcription = result["text"]

        if audio_url and os.path.exists(file_path):
            os.remove(file_path)
        return transcription
    except Exception as e:
        return f"Transcription error: {str(e)}"

def generate_summary_from_transcript(transcription: str, user_prompt) -> str:
    """
    Generates structured discussion summary in Markdown format from a transcript using a quantized LLaMA model.

    Args:
        transcription (str): The transcript text to summarize.

    Returns:
        str: Markdown-formatted summary.
    """
    # Prompt configuration for LLM
    system_message = (
        "You are an assistant that produces highlights of a discussion from transcripts, "
        "with summary, key discussion points, takeaways and action items with owners, in markdown."
    )
    user_message = (
        f"{user_prompt}"
        f"Here is the discussion transcript {transcription}. Please provide highlights including"
         "including summary, key discussion points, takeaways and action items in markdown"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    # Model quantization configuration
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    # Load tokenizer and model
    llama = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(llama)
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize prompts and generate output
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
    streamer = TextStreamer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        llama,
        device_map="auto",
        quantization_config=quant_config
    )
    outputs = model.generate(inputs, max_new_tokens=2000, streamer=streamer)

    # Decode and return result
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response