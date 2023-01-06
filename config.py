# some training parameters
EPOCHS = 100
BATCH_SIZE = 4
NUM_CLASSES = 5
image_height = 224
image_width = 224
channels = 3

model_save_name = "EfficientNetV2B0"
model_dir = "../tf2_image_classification/trained_models/IMAS_Salmon/"+model_save_name+"/" # = save_dir

dir_base = "/home/mirap/0_DATABASE/IMAS_Salmon/6_Salmon_Yolo_Balanced/"
train_dir = dir_base + "set1"
valid_dir = dir_base + "5_folds/1fold"
test_dir = dir_base + "test"
test_image_path = dir_base + "test/4/untitled-34_2693_1319_4.jpg"
