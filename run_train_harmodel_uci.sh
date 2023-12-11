# har model
# for i in $(seq 1 10); do
#     for model in RTWCNN HARLSTM HARBiLSTM HARConvLSTM ResNetTSC FCNTSC; do
#         CUDA_VISIBLE_DEVICES=1 python train_harmodel.py --dataset uci --arch $model --seed $i --trial $i
#     done
# done
for i in $(seq 1 10); do
    for model in HARLSTM HARBiLSTM HARConvLSTM; do
        CUDA_VISIBLE_DEVICES=1 python train_harmodel.py --dataset uci --arch $model --seed $i --trial $i --early_stop 100 --optimizer Adam --learning_rate 1e-3 --epochs 500
    done
done