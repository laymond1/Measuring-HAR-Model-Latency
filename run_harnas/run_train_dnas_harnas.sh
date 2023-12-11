# Description: Run training for HARNAS on UCI dataset
for i in $(seq 2 10); do
    for data in uci uni wis opp kar; do
        CUDA_VISIBLE_DEVICES=0 python ./dnas_harnas/train.py --dataset $data --arch OPPA31 --classifier LSTM --seed $i --trial $i
    done
done