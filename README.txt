AutoClipper by w-AI-fu_DEV


How does it work?

AutoClipper uses a mix of audio transcription, chat activity and LLM to identify
the best moments of a stream and automatically extract those clips from the VOD.


How to use?

First-time use requires you to run INSTALL.bat first, then use RUN.bat.
The program will ask you for the ID of the VOD, to find it go on Twitch and to
the watch page of the VOD and copy the digits at the end of the URL.
Example:
For https://www.twitch.tv/videos/2300856XXX the ID would be 2300856XXX.


Noteable:

- Process can take around an hour using a RTX2060.
- Works best for a 1 to 2 speaker stream.
- Speaker identification is based on average pitch of a spoken audio segment,
  a segment with a average pitch higher than SPEAKER_PITCH_THRESHOLD will be
  identified as SPEAKER1, and any segments lower the the threshold will be
  identified as SPEAKER2. You can get an idea of what threshold to use by doing
  a test run and checking the average pitch of each transcribed lines. The
  threshold should be equal to ((usual SPEAKER1 pitch) + (usual SPEAKER2 pitch)) / 2


Requirements:
- OpenAI account


Optional features:
- Use your own Fine-tuned OpenAI model.
- Use faster-whisper instead of openai-whisper (not recommended unless you have
  a beefy PC) (Needs some changes to AutoClipper.py)
