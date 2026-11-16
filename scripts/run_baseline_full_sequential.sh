
#!/usr/bin/env bash

set -euo pipefail



ROOT="$HOME/Maize-DRL-Benchmark"

ALGORITHM="${1:?Algorithm required}"



case "$ALGORITHM" in

    ppo|crossq|dsac_t)

        PYTHON="$ROOT/.venv/bin/python"

        ;;

    dreamerv3)

        PYTHON="$ROOT/.venv_dreamer/bin/python"

        ;;

    *)

        echo "Unknown algorithm: $ALGORITHM"

        exit 1

        ;;

esac



cd "$ROOT"



export MAIZE_BENCHMARK_OUTPUT_ROOT=outputs

export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"



SCENARIOS=(baseline_full)

SEEDS=(1 2 3)



for SCENARIO in "${SCENARIOS[@]}"; do

    for SEED in "${SEEDS[@]}"; do



        echo "================================"

        echo "ALGORITHM: $ALGORITHM"

        echo "SCENARIO: $SCENARIO"

        echo "SEED: $SEED"

        echo "TIMESTEPS: 1500000"

        echo "================================"



        "$PYTHON" scripts/train.py --project-root "$ROOT" --algorithm "$ALGORITHM" --scenario "$SCENARIO" --seed "$SEED" --timesteps 1500000



        "$PYTHON" scripts/evaluate.py --project-root "$ROOT" --algorithm "$ALGORITHM" --scenario "$SCENARIO" --seed "$SEED" --split test



        echo "COMPLETED: $ALGORITHM $SCENARIO seed=$SEED"



    done

done



echo "ALL 3 EXPERIMENTS COMPLETED: $ALGORITHM"

