# Description: Finetune script for training the model in Colab
# Some path need to edit to run the script in your environment

python -u main.py --use_type train
cp ./train.log /content/drive/MyDrive/comp5423/
cp ./0420.log /content/drive/MyDrive/comp5423/
cp -r ./model /content/drive/MyDrive/comp5423/bart_model_0420