#!/bin/bash
# Live status for the locally-running folded O-LoRA / InfLoRA hedge runs (bank_400t_folded.sh).
# The training logs are polluted by tqdm carriage-return progress bars; this strips them and
# shows only the signal lines: current task, per-task [mem] (live_boundary + step_peak -- the
# fold should keep step_peak FLAT in #tasks), and InfLoRA's DualGPM subspace-fill diagnostic.
#
#   ./bank_status.sh          # one-shot snapshot
#   ./bank_status.sh -w       # refresh every 20s (watch)
#   ./bank_status.sh -f olora # live-follow one lane's signal lines (Ctrl-C to stop)

OUT="$(cd "$(dirname "$0")" && pwd)/run_logs/a100_400t"
SIG='^\[task |\[mem\] task|DualGPM fill|==== |OLORA  LANE|INFLORA LANE|COMPLETE'

snapshot () {
  echo "===================  $(date '+%F %T')  ==================="
  # which python lanes are alive
  ps -o pid=,etime=,args= -C python 2>/dev/null | grep -E "main_svdlora_baseline" \
    | grep -oE "method [a-z]+|order_seed [0-9]+|--method [a-z]+" | paste - - 2>/dev/null
  for f in "$OUT"/exp_olora_*.out "$OUT"/exp_inflora_*.out; do
    [ -f "$f" ] || continue
    echo "---- $(basename "$f") ----"
    # last task header, last mem line, last DualGPM fill (whichever exist)
    grep -aE "$SIG" "$f" | grep -avE "it/s|it\]" | tail -3
  done
  echo "GPU mem:"; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null
}

case "$1" in
  -w) while true; do clear; snapshot; sleep 20; done ;;
  -f) lane="${2:-olora}"; tail -F "$OUT"/exp_${lane}_*.out | grep --line-buffered -aE "$SIG" | grep -avE "it/s|it\]" ;;
  *)  snapshot ;;
esac