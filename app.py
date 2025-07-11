from flask import Flask, render_template, request, send_file
import os
import uuid
import torch
from fairseq.data.dictionary import Dictionary
from rvc_python.infer import RVCInference
from types import MethodType
from scipy.io import wavfile
from pydub import AudioSegment

# Allow fairseq dictionary during safe deserialization
torch.serialization.add_safe_globals([Dictionary])

app = Flask(__name__)

# Define directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "cloned_audio_files")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Patch RVC inference

def patched_infer_file(self, input_path, output_path):
    if not self.current_model:
        raise ValueError("Model not loaded.")
    model_info = self.models[self.current_model]
    file_index = model_info.get("index", "")

    result = self.vc.vc_single(
        sid=0,
        input_audio_path=input_path,
        f0_up_key=self.f0up_key,
        f0_method=self.f0method,
        file_index=file_index,
        index_rate=self.index_rate,
        filter_radius=self.filter_radius,
        resample_sr=self.resample_sr,
        rms_mix_rate=self.rms_mix_rate,
        protect=self.protect,
        f0_file="",
        file_index2=""
    )

    wav = result[0] if isinstance(result, tuple) else result
    wavfile.write(output_path, self.vc.tgt_sr, wav)

# Route for upload page
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Route for processing
@app.route("/process", methods=["POST"])
def process():
    audio = request.files.get("audio")
    models = request.files.getlist("models")

    if not audio or not models:
        return "Missing audio or model files", 400

    input_filename = f"input_{uuid.uuid4().hex}.wav"
    input_path = os.path.join(UPLOAD_DIR, input_filename)
    audio.save(input_path)

    # Load audio and split into chunks
    audio_segment = AudioSegment.from_wav(input_path)
    duration_ms = len(audio_segment)
    chunk_duration = duration_ms // len(models)

    output_filenames = []

    for idx, model in enumerate(models):
        # Save model
        model_filename = f"model_{uuid.uuid4().hex}.pth"
        model_path = os.path.join(UPLOAD_DIR, model_filename)
        model.save(model_path)

        # Export audio chunk
        chunk = audio_segment[idx * chunk_duration:(idx + 1) * chunk_duration]
        chunk_filename = f"chunk_{uuid.uuid4().hex}.wav"
        chunk_path = os.path.join(UPLOAD_DIR, chunk_filename)
        chunk.export(chunk_path, format="wav")

        # Clone with RVC
        output_filename = f"cloned_{uuid.uuid4().hex}.wav"
        output_path = os.path.join(UPLOAD_DIR, output_filename)

        rvc = RVCInference(model_path=model_path)
        rvc.infer_file = MethodType(patched_infer_file, rvc)
        rvc.set_params(f0up_key=0, index_rate=0.75)
        rvc.infer_file(input_path=chunk_path, output_path=output_path)

        output_filenames.append(output_filename)

    return render_template("result.html", filenames=output_filenames)

@app.route("/download/<filename>")
def download_file(filename):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        return "File not found.", 404
    return send_file(
        file_path,
        as_attachment=True,
        download_name=filename,
        mimetype="audio/wav"
    )

if __name__ == "__main__":
    app.run(debug=True)