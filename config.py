# some training parameters
EPOCHS = 1000
BATCH_SIZE = 1
NUM_CLASSES = 3
image_height = 128
image_width = 128
channels = 3

model_save_name = "TestNet"
model_dir = "trained_models/IMAS_Salmon/"+model_save_name+"/" # = save_dir

dir_base = "/home/mirap/0_DATABASE/IMAS_Salmon/6_Salmon_Yolo_Balanced/"
train_dir = dir_base + "set1"
valid_dir = dir_base + "5_folds/1fold"
test_dir = dir_base + "test"
test_image_path = dir_base + "test/1/untitled-1054_2053_679_1.jpg"
