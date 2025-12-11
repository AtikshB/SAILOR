Run collect_demos -> robocasa
Run SAILOR -> robocasa_env

python robocasa/robocasa/scripts/collect_demos.py --env PickPlaceCan --robots GR1FixedLowerBody --device keyboard


SUITE="robocasa"
TASK="pickplacecan"
NUM_EXP_TRAJS=1
SEED=0
python3 train_sailor.py \
    --configs cfg_dp_mppi ${SUITE}\
    --task "${SUITE}__${TASK}" \
    --num_exp_trajs ${NUM_EXP_TRAJS} \
    --seed ${SEED}