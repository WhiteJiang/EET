#！/bin/bash

code_lengths=("16" "32" "48" "64")

for code_length in "${code_lengths[@]}";do
	python main.py --dataset cub_bird --model-name EET --code-length "$code_length" --epoch 90 --batch-size 64 --w 4 --lr 0.01 --gpu-ids 5
done