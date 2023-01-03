# some training parameters
EPOCHS = 1000
BATCH_SIZE = 16
NUM_CLASSES = 3
image_height = 128
image_width = 128
channels = 3

model_save_name = "TestNet"
model_dir = "trained_models/ivy_coverage/"+model_save_name+"/" # = save_dir

dir_base = "/home/mirap/0_DATABASE/ivy_coverage/cropped/"
train_dir = dir_base + "set1"
valid_dir = dir_base + "5_folds/1fold"
test_dir = dir_base + "test"
test_image_path = dir_base + "test/ivy/resultsivy_006_2112.jpg"
