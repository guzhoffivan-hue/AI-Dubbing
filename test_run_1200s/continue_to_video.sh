#!/bin/sh
# Continue the isolated 1200 s test from the finished translation to the first MP4.
#
# Each stage is invoked directly with the same AI_DUB_* environment that cli.py sets
# (cli.py has no stage-subset flag and would also re-run ASR/alignment/diarization),
# under caffeinate so the Mac cannot sleep mid-run. The chain stops at the first failure
# and writes one log per stage.

set -e
cd "$(dirname "$0")"
export AI_DUB_WORKDIR="$PWD/work/42503442096_first1200s"
export AI_DUB_INPUT_VIDEO="$PWD/input/42503442096_first1200s.mp4"
export AI_DUB_OUTPUT_DIR="$PWD/output"
export AI_DUB_OUTPUT_FILE="$PWD/output/test_dub_1200s.mp4"
L=logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "waiting for translation v3 to finish"
i=0
while [ $i -lt 600 ]; do
  pgrep -f "scripts/translate.py" >/dev/null || break
  sleep 10
  i=$((i+1))
done
if ! grep -q "Translated 382 speaker-labelled" "$L/translate_run3.log"; then
  log "ABORT: translation v3 did not complete"
  tail -3 "$L/translate_run3.log"
  exit 1
fi
log "translation v3 complete ($(grep -c 'Translated ' "$L/translate_run3.log") lines)"

step() {
  name=$1; shift
  log "START $name"
  caffeinate -i "$@" > "$L/chain_$name.log" 2>&1
  log "DONE  $name"
}

step dialogue ./.venvs/asr/bin/python       scripts/dialogue.py
step tts      ./.venvs/translate/bin/python scripts/tts.py
step fit_tts  ./.venvs/asr/bin/python       scripts/fit_tts.py
step sfx      ./.venvs/asr/bin/python       scripts/sfx.py timeline
step ambience ./.venvs/asr/bin/python       scripts/ambience.py
step mix      ./.venvs/asr/bin/python       scripts/mix_audio.py
step mux      ./.venvs/asr/bin/python       scripts/mux.py

log "ALL DONE -> $AI_DUB_OUTPUT_FILE"
ls -l "$AI_DUB_OUTPUT_FILE"
