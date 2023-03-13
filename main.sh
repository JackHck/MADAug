GPU=0

function run_reduced_svhn {
    DATASET=reduced_svhn
    MODEL=wresnet28_10
    EPOCH=200
    BATCH=128
    LR=0.05
    WD=0.01
    CUTOUT=16
    SLR=0.001
    SF=5
    tem=2
    KOS=2
}

# cifar10
function run_reduced_cifar10 {
    DATASET=reduced_cifar10
    MODEL=wresnet28_10
    EPOCH=240
    BATCH=128
    LR=0.1
    WD=0.0005
    SLR=0.001
    CUTOUT=16
    SF=3
    tem=3
    KOS=2
}


if [ $1 = "reduced_cifar10" ]; then
    run_reduced_cifar10
elif [ $1 = "reduced_svhn" ]; then
    run_reduced_svhn
fi

SAVE=./${DATASET}_${MODEL}_${BATCH}_${EPOCH}_SLR${SLR}_SF${SF}_cutout_${CUTOUT}_lr${LR}_wd${WD}_kops${KOS}

python main_higher.py --k_ops ${KOS} --report_freq 10 --num_workers 4 --epochs ${EPOCH} --batch_size ${BATCH} --learning_rate ${LR} --dataset ${DATASET} --model_name ${MODEL}  --gpu ${GPU} --weight_decay ${WD} --proj_learning_rate ${SLR} --search_freq ${SF} --temperature ${tem} --cutout --cutout_length ${CUTOUT} --save ${SAVE} 
