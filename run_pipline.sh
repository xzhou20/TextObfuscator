# install requirements
pip install -r ./requirements.txt

# start to train
model_save_path=./save/models
# conll2003
python ./train_token.py --dataset_name conll2003 --w_cluster_close 0.5 --w_cluster_away 0.3 --output_dir ${model_save_path}

# # ontonotes5
# # python ./train_token.py --dataset_name tner/ontonotes5 --w_cluster_close 0.5 --w_cluster_away 0.3

# # SST-2
# # python ./train_sentence.py --task_name sst2 --w_cluster_close 0.5 --w_cluster_away 0.1 --learning_rate 1e-5

# # AGNews
# # python ./train_sentence.py --task_name ag_news --w_cluster_close 0.5 --w_cluster_away 0.1

# # privacy attack
python ./inversion_attack.py --task_name conll2003 --model_path ${model_save_path} 

python ./mlc_attack.py --task_name conll2003 --model_path ${model_save_path} 