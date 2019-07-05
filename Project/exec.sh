CUDA_VISIBLE_DEVICES=2 python3 Main.py -d ./data/olid-Maining-v1.0.tsv -m KNN -t taskB | grep -v "warning" >> Result.txt
CUDA_VISIBLE_DEVICES=2 python3 Main.py -d ./data/olid-Maining-v1.0.tsv -m DT -t taskB | grep -v "warning" >> Result.txt
CUDA_VISIBLE_DEVICES=2 python3 Main.py -d ./data/olid-Maining-v1.0.tsv -m RF -t taskB | grep -v "warning" >> Result.txt
CUDA_VISIBLE_DEVICES=2 python3 Main.py -d ./data/olid-Maining-v1.0.tsv -m LR -t taskB | grep -v "warning" >> Result.txt
CUDA_VISIBLE_DEVICES=2 python3 Main.py -d ./data/olid-Maining-v1.0.tsv -m SVM -t taskB | grep -v "warning" >> Result.txt
