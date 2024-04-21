# Description: Inference script for testing the model in Colab
# Some path need to edit to run the script in your environment

python -u main.py --use_type inference --load_model --model_path ./model/
cp ./inference.log /content/drive/MyDrive/comp5423/
cp ./data/result.txt /content/drive/MyDrive/comp5423/