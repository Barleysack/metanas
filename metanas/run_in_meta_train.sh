

DATASET=omniglot
DATASET_DIR=../dataset
TRAIN_DIR=../results/
		
mkdir -p $TRAIN_DIR





args=(
    # Execution
    --name metatest_in \
    --job_id 0 \
    --path ${TRAIN_DIR} \
    --data_path ${DATASET_DIR} \
    --dataset $DATASET
    --hp_setting 'in_metanas' \
    --use_hp_setting 1 \
    --workers 0 \
    --gpus 0 \
    --test_adapt_steps 1.0 \

    # few shot params
     # examples per class
    --n 5 \
    # number classes  
    --k 5 \
    # test examples per class
    --q 1

    --meta_model_prune_threshold 0.01 \
    --alpha_prune_threshold 0.01 \
    # Meta Learning
    --meta_model searchcnn \
    --meta_epochs 100 \
    --warm_up_epochs 0 \
    --use_pairwise_input_alphas \
    --eval_freq 1 \
    --eval_epochs 0 \

    --normalizer softmax \
    --normalizer_temp_anneal_mode linear \
    --normalizer_t_min 0.1 \
    --normalizer_t_max 1.0 \
    --drop_path_prob 0.2 \
    --test_task_train_steps 10

    # Architectures
    --init_channels 28 \
    --layers 4 \
    --reduction_layers 1 3 \
    --use_first_order_darts \

    --use_torchmeta_loader \
    # experiments
    --exp_const 1
    --exp_cell 1
    --wandb 0
)


python -m metanas_main "${args[@]}"

