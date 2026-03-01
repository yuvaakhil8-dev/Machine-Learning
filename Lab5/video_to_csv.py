import os
import pandas as pd
import subprocess
import json

base_path = "C:/Games/Desktop/Documents/Projects_Sem4/ML/dataset"

data = []

def extract_video_metadata(video_path):
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        info = json.loads(result.stdout)

        duration = float(info["format"]["duration"])
        bitrate = float(info["format"]["bit_rate"])

        # Extract resolution
        video_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
        width = video_stream["width"]
        height = video_stream["height"]

    except:
        duration = 0
        bitrate = 0
        width = 0
        height = 0

    file_size = os.path.getsize(video_path)

    return duration, bitrate, width, height, file_size


# SAFE
for file in os.listdir(os.path.join(base_path, "Safe")):
    if file.endswith(".mp4"):
        path = os.path.join(base_path, "Safe", file)
        features = extract_video_metadata(path)
        data.append(list(features) + [0])

# UNSAFE
for file in os.listdir(os.path.join(base_path, "Unsafe")):
    if file.endswith(".mp4"):
        path = os.path.join(base_path, "Unsafe", file)
        features = extract_video_metadata(path)
        data.append(list(features) + [1])


df = pd.DataFrame(data, columns=[
    "duration_sec",
    "bitrate",
    "width",
    "height",
    "file_size_bytes",
    "label"
])

df.to_csv(os.path.join(base_path, "video_features.csv"), index=False)

print(" CSV created successfully!")
print("Total samples:", len(df))