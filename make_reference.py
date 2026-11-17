import asyncio
import edge_tts

TEXT = (
    "Hello there. This is a calm and steady reference recording, "
    "spoken slowly and clearly at a comfortable pace. "
    "The morning light fell gently across the quiet valley, "
    "and a soft breeze moved through the trees. "
    "Every word is measured and relaxed, with even pauses between phrases, "
    "so that the character of this voice can be heard clearly."
)
VOICE = "en-US-JennyNeural"
OUT = "reference_raw.mp3"

async def main():
    tts = edge_tts.Communicate(TEXT, VOICE, rate="-12%", pitch="-2Hz")
    await tts.save(OUT)
    print(f"Saved {OUT}")

asyncio.run(main())