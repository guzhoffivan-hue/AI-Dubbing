from app.models.character import Character
from app.models.speech_segment import SpeechSegment
from app.models.transcript import Transcript


def test_transcript():
    transcript = Transcript()

    first_segment = SpeechSegment(
        id=1,
        start=0.0,
        end=2.5,
        speaker="SPEAKER_01",
        text="Привет.",
    )

    second_segment = SpeechSegment(
        id=2,
        start=3.0,
        end=6.0,
        speaker="SPEAKER_02",
        text="Привет.",
    )

    character = Character(
        id="character_01",
        name="Персонаж 1",
        speaker_id="SPEAKER_01",
    )

    transcript.add_segment(first_segment)
    transcript.add_segment(second_segment)
    transcript.add_character(character)

    assert len(transcript.segments) == 2
    assert len(transcript.characters) == 1
    assert transcript.duration == 6.0
    assert transcript.segments[0].text == "Привет."
    assert transcript.characters[0].name == "Персонаж 1"