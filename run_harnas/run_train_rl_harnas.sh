# Description: Run training for HARNAS on UCI dataset
for i in $(seq 1 10); do
    for data in uci uni wis opp kar; do
        CUDA_VISIBLE_DEVICES=1 python ./rl_harnas/train.py --dataset $data --arch RLNAS --seed $i --trial $i
    done
done