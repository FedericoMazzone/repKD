import os
import re


def get_architecture_by_model_name(model_name):
    s = re.search("_a(.*)", model_name)
    return list(map(int, s.group(1).split('-')))


def get_batch_size_by_model_name(model_name):
    return int(re.search("_b(\d+)", model_name).group(1))


def get_train_size_by_model_name(model_name):
    return int(re.search("_d(\d+)", model_name).group(1))


def disable_cuda():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def log(message):
    log_file_path = f"log{os.getpid()}.txt"
    with open(log_file_path, 'a+') as f:
        f.write(message + "\n")
    print(message)


# model_name = "modeloriginal_d400_oadam_lcc_b10_a30-10"
# print("model_name: {}".format(model_name))
# architecture = get_architecture_by_model_name(model_name)
# print("Architecture: {}".format(architecture))
# batch_size = get_batch_size_by_model_name(model_name)
# print("Batch size: {}".format(batch_size))
# train_size = get_train_size_by_model_name(model_name)
# print("Train size: {}".format(train_size))
