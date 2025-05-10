import gradio as gr
from summarizer import transcribe, generate_summary_from_transcript


def transcribe_and_generate(audio_file, audio_url, prompt):
    # get audio transcript
    transcript = transcribe(audio_file, audio_url)

    # generate summary from transcript
    try:
        markdown_summary = generate_summary_from_transcript(transcript, prompt)
        # Save to file for download
        output_path = "audio_summary.md"
        with open(output_path, "w") as f:
            f.write(markdown_summary)
        return markdown_summary, output_path
    except Exception as e:
        return f"Generation error: {str(e)}", None

# ----------- Gradio UI -----------

with gr.Blocks() as demo:
    gr.Markdown("## 📝 Audio to Text Summary Generator")

    with gr.Row():
        audio_file = gr.Audio(label="Upload audio file", type="filepath")
        audio_url = gr.Textbox(label="Or paste audio URL (e.g. Google Drive)", placeholder="https://...")

    prompt = gr.Textbox(label="Custom prompt (optional)", lines=4, placeholder="Leave empty to use default audio summarizer prompt")

    with gr.Row():
        transcribe_btn = gr.Button("Generate Summary")
        download_file = gr.File(label="Download .md file")

    markdown_output = gr.Markdown()

    transcribe_btn.click(
        transcribe_and_generate,
        inputs=[audio_file, audio_url, prompt],
        outputs=[markdown_output, download_file]
    )

# ----------- Launch App -----------
if __name__ == "__main__":
    demo.launch(share=True)