# Description: Inference script for testing the model in Colab
# Some path need to edit to run the script in your environment

python -u main.py --use_type inference --model_path ./model/12_9000 --load_model 
cp ./inference.log /content/drive/MyDrive/comp5423/
cp ./data/result.txt /content/drive/MyDrive/comp5423/