# Description: Run training for HARNAS on UCI dataset
for i in $(seq 1 10); do
    for data in uci uni wis opp kar; do
        CUDA_VISIBLE_DEVICES=0 python train_ea_harnas.py --dataset $data --arch EANAS --seed $i --trial $i
    done
done