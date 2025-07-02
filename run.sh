DATASET=Musical_Instruments
DEVICE=cuda:0


python main.py \
    --dataset=$DATASET \
    --lr=0.001 \
    --neg_num=24000 \
    --text_types title brand features categories description \
    --mask_ratio=0.5 \
    --cl_weight=0.4 \
    --mlm_weight=0.6 \
    --data_path=./dataset \
    --text_index_path=.code.pq.20_256.pca128.title_brand_features_categories_description.json \
    --code_level=20 \
    --n_codes_per_lel=256 \
    --max_his_len=20 \
    --batch_size=400 \
    --dropout_prob=0.3 \
    --dropout_prob_cross=0.3 \
    --n_layers=2 \
    --n_heads=2 \
    --embedding_size=128 \
    --hidden_size=512 \
    --device=$DEVICE

    


    

