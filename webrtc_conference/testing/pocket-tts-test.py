from pocket_tts import TTSModel
import soundfile as sf

model = TTSModel.load_model(language="spanish")

print("Model loaded")

state = model.get_state_for_audio_prompt(
    r"C:\Users\user\Music\project\LivePortrait\Recording (2).wav"
)

print("Voice cloned")

audio = model.generate_audio(
    state,
    "Hola, ¿cómo estás?"
)

sf.write(
    "pocket_spanish_test.wav",
    audio.numpy(),
    model.sample_rate
)

print("Done")