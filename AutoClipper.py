import os
import subprocess as sp
import numpy as np
import whisper
import shutil
import librosa
import torch
import json
from openai import OpenAI
import faster_whisper


def read_env()-> dict:
    ret_val = {}
    with open(".env", "r") as f:
        env_lines = f.readlines()
    
    for line in env_lines:
        if line.strip() == "": continue
        parts = line.strip().split("=")
        if len(parts) < 2: continue
        ret_val[parts[0]] = "=".join(parts[1:])
    return ret_val


env = read_env()

TWITCHDL_PATH = env["TWITCHDL_PATH"]
FFMPEG_PATH = env["FFMPEG_PATH"]
OPENAI_TOKEN = env["OPENAI_TOKEN"]
OPENAI_MODEL = env["OPENAI_MODEL"]

SPEAKER1 = env["SPEAKER1_NAME"]
SPEAKER2 = env["SPEAKER2_NAME"]
PITCH_THRESHOLD = int(env["SPEAKER_PITCH_THRESHOLD"])

PY_PATH = "venv/Scripts/python.exe"

VOD_MP4_PATH = "out/vod.mp4"
VOD_WAV_PATH = "out/audio.wav"
VOD_PCM_PATH = "out/audio.pcm"
VOD_CHAT_PATH = "out/chat.json"
VOD_TRANSCRIPT_PATH = "out/transcript.txt"

SILENCE_DURATION = 0.2
SAMPLE_RATE = 16000
SAMPLED_SILENCE_DURATION = SILENCE_DURATION * SAMPLE_RATE
SILENCE_THRESHOLD = 2750
AUDIO_CHUNK = 1024


def get_average_pitch(audio_data, sr=16000)-> int | None:
    pitches, _, _ = librosa.pyin(audio_data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    valid_pitches = pitches[~np.isnan(pitches)]
    if len(valid_pitches) == 0:
        return None
    return int(np.mean([np.mean(valid_pitches), np.median(valid_pitches)]))


def seconds_to_hms(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def hms_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s


def load_audio_chunks(file_path, chunk_size_bytes=1024):
    with open(file_path, "rb") as f:
        read_buffer = b""
        while True:
            if not read_buffer:
                read_buffer = f.read(chunk_size_bytes * 10)
                if not read_buffer:
                    break

            chunk = read_buffer[:chunk_size_bytes]
            read_buffer = read_buffer[chunk_size_bytes:]
            yield chunk


def local_whisper_transcribe()-> None:
    # [For faster-whisper]
    #if "tiny.en" not in faster_whisper.available_models():
    #    faster_whisper.download_model("tiny.en")
    #model = faster_whisper.WhisperModel("tiny.en", device="cuda", compute_type="float16")
    
    # [For openai-whisper]
    model = whisper.load_model("small.en", device=torch.device("cuda"), in_memory=True)
    
    buffer = b""
    buffer_len = 0

    buffer_samples = []
    samples_len = 0

    was_silent_before = True
    transcript: list[dict] = []

    print("\nStarted whisper transcribing...")

    S16_CHUNK_SIZE = int(AUDIO_CHUNK / 2)
    SILENCE_DURATION_SEC = int(SAMPLED_SILENCE_DURATION)
    BUFFER_MAX_THRESH = 1_000_000
    BUFFER_MIN_THRESH = AUDIO_CHUNK * 2
    
    for i, data in enumerate(load_audio_chunks(VOD_PCM_PATH, AUDIO_CHUNK)):
        audio_data = np.frombuffer(data, dtype=np.int16)
        buffer_samples.extend(audio_data)
        samples_len += S16_CHUNK_SIZE

        timestamp = (i * S16_CHUNK_SIZE) / SAMPLE_RATE

        # Check for silence in end of last samples
        last_samples = buffer_samples if samples_len < SILENCE_DURATION_SEC else buffer_samples[-SILENCE_DURATION_SEC:]
        is_silent = np.max(np.abs(last_samples)) < SILENCE_THRESHOLD

        # Add non-silent audio to buffer
        if not is_silent:
            buffer += data
            buffer_len += AUDIO_CHUNK
        
        # Make sure the buffer we send to whisper is not too small
        # Reduces risks of wrong translation
        if buffer_len < BUFFER_MIN_THRESH:
            continue

        # Buffer is too long with no silence, either silence threshold is too low
        # or is loud music.
        if buffer_len > BUFFER_MAX_THRESH:
            buffer = b""
            buffer_samples = []
            samples_len = 0
            buffer_len = 0
            print("Skipped audio segment, this may be caused by loud music or a silence threshold set too low.")
            continue

        # Transcribe with whisper
        if is_silent and not was_silent_before:
            was_silent_before = True

            # Pcm16 to f32 samples
            audio_np = np.frombuffer(buffer, np.int16).flatten().astype(np.float32) / 32768.0
            avg_pitch = get_average_pitch(audio_np, SAMPLE_RATE)
            
            trs = {}
            trs["timestamp"] = seconds_to_hms(timestamp)

            if avg_pitch is not None:
                trs["speaker"] = SPEAKER1 if avg_pitch > PITCH_THRESHOLD else SPEAKER2
            else:
                trs["speaker"] = "Unknown"

            buffer = b""
            buffer_samples = []
            samples_len = 0
            buffer_len = 0

            # [For faster-whisper]
            #segments, _ = model.transcribe(audio_np,
            #                        language='en',
            #                        initial_prompt="Hello Hilda, welcome to my lecture.",
            #                        hallucination_silence_threshold=1.5)
            
            # [For openai-whisper]
            result = model.transcribe(audio_np,
                                    language='en',
                                    initial_prompt="Hello Hilda, welcome to my lecture.",
                                    hallucination_silence_threshold=1.5)

            # [For faster-whisper]
            #trs["content"] = ""
            #for segment in segments:
            #    trs["content"] += segment.text
            
            # [For openai-whisper]
            trs["content"] = result.get('text', '').strip()

            # Send transcription
            if trs["content"] != "" and "lecture" not in trs["content"]:
                print(trs["timestamp"], f"(pitch: {int(avg_pitch) if avg_pitch else 0})", trs["speaker"] + ":", trs["content"])
                
                trs_len = len(transcript)
                if trs_len == 0:
                    transcript.append(trs)
                elif transcript[trs_len - 1]["speaker"] == trs["speaker"]:
                    transcript[trs_len - 1]["content"] += " " + trs["content"]
                else:
                    transcript.append(trs)
        
        if not is_silent:
            was_silent_before = False

    with open(VOD_TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        for line in transcript:
            f.write(f'{line["timestamp"]} {line["speaker"]}: {line["content"]}\n')
    
    print("\nFinished transcribing.\n")


def get_top_active_intervals(json_file_path, interval_seconds=5, top_n=5):
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        comments = data.get("comments", [])
        if not comments:
            print("No comments found in the JSON data.")
            return []
        
        interval_counts: dict[int, int] = {}
        
        for comment in comments:
            timestamp = comment['content_offset_seconds']
            interval_index = int(timestamp // interval_seconds) * interval_seconds
            interval_counts[interval_index] += 1
        
        top_intervals = sorted(interval_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_active_intervals = [(start, count) for start, count in top_intervals]
        
        top_active = []
        for x in top_active_intervals:
            top_active.append({
                "title": "top_chat_activity_" + x[1],
                "start": seconds_to_hms(x[0] - 60),
                "end": seconds_to_hms(x[0] + 30)
            })

        return top_active
    
    except json.JSONDecodeError:
        print("Error: The JSON file is not properly formatted.")
        return []
    except FileNotFoundError:
        print("Error: The JSON file was not found.")
        return []
    except KeyError as e:
        print(f"Error: Missing expected key in JSON data - {e}")
        return []


def main()-> None:
    print("AutoClipper 1.0.0")
    print("by w-AI-fu_DEV")
    client = OpenAI(api_key=OPENAI_TOKEN)

    if os.path.isdir("out"):
        shutil.rmtree("out")

    os.mkdir("out")

    vod_id: int = int(input("VOD ID: "))

    res: sp.CompletedProcess = sp.run([TWITCHDL_PATH, "videodownload", "--id", str(vod_id), "--ffmpeg-path", FFMPEG_PATH, "-o", VOD_MP4_PATH])
    if res.returncode > 0:
        raise Exception("Something went wrong when downloading VOD.")

    res = sp.run([TWITCHDL_PATH, "chatdownload", "--id", str(vod_id), "-o", VOD_CHAT_PATH, "-E"])
    if res.returncode > 0:
        raise Exception("Something went wrong when downloading VOD's chat.")

    res = sp.run([FFMPEG_PATH, "-y", "-i", VOD_MP4_PATH, "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", VOD_WAV_PATH])
    if res.returncode > 0:
        raise Exception("Something went wrong when converting VOD audio to wav.")

    res = sp.run([FFMPEG_PATH, "-y", "-i", VOD_WAV_PATH, "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-acodec", "pcm_s16le", VOD_PCM_PATH])
    if res.returncode > 0:
        raise Exception("Something went wrong when converting VOD audio to pcm.")

    local_whisper_transcribe()

    with open("openai-prompt.txt", "r") as f:
        sys_prompt = f.read()

    with open("out/transcript.txt", "r") as f:
        transcript = f.read()

    print("Using Openai to identify best clips...")

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": transcript}
        ],
        response_format={"type": "json_object"}
    )

    result = completion.choices[0].message.content
    print(result)

    os.mkdir("out/clips")

    data = json.loads(result)
    data["top chat moments"] = get_top_active_intervals("out/chat.json", interval_seconds=30, top_n=5)

    for category, clips in data.items():
        for clip in clips:
            start_time = max(0, hms_to_seconds(clip["start"]) - 30)
            end_time = hms_to_seconds(clip["end"] + 30)
            duration = end_time - start_time
            title_safe = clip["title"].replace(" ", "_").replace("'", "").replace(":", "")
            output_file = f"out/clips/{title_safe}.mp4"
            sp.run([FFMPEG_PATH, '-i', VOD_MP4_PATH, '-ss', str(start_time), '-t', str(duration), '-c', 'copy', output_file])
            print(f"Extracted clip: {output_file}")

    print("\nProcess finished, the clips can be found in out/clips.")


main()