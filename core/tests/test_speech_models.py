from app.models.character import Character
from app.models.speech_segment import SpeechSegment


def test_speech_segment():
    segment = SpeechSegment(
        id=1,
        start=10.5,
        end=14.0,
        speaker="SPEAKER_01",
        text="Что ты здесь делаешь?",
    )

    assert segment.duration == 3.5
    assert segment.speaker == "SPEAKER_01"


def test_character():
    character = Character(
        id="character_01",
        name="Маомао",
        speaker_id="SPEAKER_01",
        voice_id="voice_female_01",
        volume=1.0,
        speed=0.95,
        pitch=0.0,
    )

    assert character.name == "Маомао"
    assert character.speaker_id == "SPEAKER_01"
    assert character.voice_id == "voice_female_01"
    assert character.speed == 0.95
