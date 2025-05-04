import requests
import soundfile as sf
import gdown
from urllib.parse import urlparse, parse_qs

from google.colab import userdata


def download_audio(audio_url):
    """
    Downloads audio from a direct link or Google Drive link using gdown if needed.

    Args:
        audio_url (str): Direct audio URL or Google Drive share link.

    Returns:
        Tuple[str or None, str or None]: File path and error (if any).
    """
    try:
        if "drive.google.com" in audio_url:
            file_id = ""
            if "file/d/" in audio_url:
                file_id = audio_url.split("file/d/")[1].split("/")[0]
            elif "id=" in audio_url:
                from urllib.parse import urlparse, parse_qs
                file_id = parse_qs(urlparse(audio_url).query).get("id", [""])[0]
            else:
                return None, "Unsupported Google Drive link format."

            download_url = f"https://drive.google.com/uc?id={file_id}"
            file_path = "temp_url_audio.mp3"
            gdown.download(download_url, file_path, quiet=False)
        else:
            response = requests.get(audio_url)
            if response.status_code != 200:
                return None, f"Failed to download audio. Status code: {response.status_code}"
            file_path = "temp_url_audio.mp3"
            with open(file_path, "wb") as f:
                f.write(response.content)

        return file_path, None
    except Exception as e:
        return None, str(e)

def is_valid_audio(file_path):
    """
    Checks if a given file is a valid audio file.

    Args:
        file_path (str): Path to the file to be checked.

    Returns:
        bool: True if the file is a valid audio file, False otherwise.
    """
    try:
        with sf.SoundFile(file_path) as f:
            return True
    except RuntimeError:
        return False


