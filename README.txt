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


Requirements:
- OpenAI account


Optional features:
- Use your own Fine-tuned OpenAI model.
- Use faster-whisper instead of openai-whisper (not recommended unless you have
  a beefy PC) (Needs some changes to AutoClipper.py)