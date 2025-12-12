import os
import cv2
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings

warnings.filterwarnings("ignore")


def sigmoid(x):
    """Sigmoid function for a scalar or NumPy array."""
    return 1 / (1 + np.exp(-x))

def getFileList(dir, Filelist, exts=None):
    if exts is None:
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    else:
        exts = {e.lower() for e in exts}

    newDir = dir
    if os.path.isfile(dir):
        if os.path.splitext(dir)[1].lower() in exts:
            Filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, exts)

    return Filelist

def read_image(image_path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # 安全读取图像（支持中文路径）
    img_array = np.fromfile(image_path, dtype=np.uint8)
    if img_array.size == 0:
        raise ValueError(f"File is empty or not found: {image_path}")

    src = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if src is None:
        raise ValueError(f"Failed to decode image: {image_path}. It may be corrupted or not an image file.")

    image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (896, 896))
    image = image.astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)
    mean = np.array(mean, dtype=np.float32).reshape((3, 1, 1))
    std = np.array(std, dtype=np.float32).reshape((3, 1, 1))
    normalized_image = (image - mean) / std
    normalized_image = np.expand_dims(normalized_image, axis=0)
    return normalized_image, src

def load_engine(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def create_context(engine):
    return engine.create_execution_context()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = {}
    stream = cuda.Stream()

    for idx in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(idx)
        tensor_mode = engine.get_tensor_mode(tensor_name)
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
        shape = engine.get_tensor_shape(tensor_name)

        size = trt.volume(shape)
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings[tensor_name] = int(device_mem)

        buf = {
            'name': tensor_name,
            'host': host_mem,
            'device': device_mem,
            'shape': shape,
            'dtype': dtype
        }

        if tensor_mode == trt.TensorIOMode.INPUT:
            inputs.append(buf)
        else:
            outputs.append(buf)

    return inputs, outputs, bindings, stream

def infer(context, inputs, outputs, bindings, stream, input_data):
    input_buf = inputs[0]
    np.copyto(input_buf['host'], input_data.ravel().astype(input_buf['dtype']))
    cuda.memcpy_htod_async(input_buf['device'], input_buf['host'], stream)

    for inp in inputs:
        context.set_tensor_address(inp['name'], inp['device'])
    for out in outputs:
        context.set_tensor_address(out['name'], out['device'])

    context.execute_async_v3(stream_handle=stream.handle)

    results = []
    for out in outputs:
        cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
        results.append(out['host'].copy())

    stream.synchronize()
    return results

def main(image_path, context, inputs, outputs, bindings, stream):
    try:
        input_data, src = read_image(image_path)
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return -1, 0.0, "none"

    output_buffers = infer(context, inputs, outputs, bindings, stream, input_data)

    pred_boxes = output_buffers[0].reshape(1, 300, 4)
    pred_logits = output_buffers[1].reshape(1, 300, 91)

    probs = sigmoid(pred_logits[0])
    object_probs = probs[:, 1:81]
    background_score = probs[:, 0]

    scores = np.max(object_probs, axis=1)
    class_ids = np.argmax(object_probs, axis=1)

    valid = scores > 0.3
    if not np.any(valid):
        return -1, 0.0, "none"

    best_idx = np.argmax(scores * valid)
    predicted_index = int(class_ids[best_idx])
    score = float(scores[best_idx])

    if predicted_index >= len(labels):
        predicted_index = 0

    label = labels[predicted_index]
    return predicted_index, score, label

if __name__ == '__main__':
    image_dir = r"media"
    engine_file_path = 'onnnx/inference_model.sim.engine'
    labels = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
        "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
        "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
        "40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
        "50", "51", "52", "53", "54", "55", "56", "57", "58", "59",
        "60", "61", "62", "63", "64", "65", "66", "67", "68", "69",
        "70", "71", "72", "73", "74", "75", "76", "77", "78", "79"
    ]
    engine = load_engine(engine_file_path)
    context = create_context(engine)
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    img_list = []
    exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    img_list = getFileList(image_dir, img_list, exts)
    count = 0
    start = time.time()
    y_true = []
    y_pred = []
    count_time = 0

    for img in img_list:
        true_label = img.split('/')[-2]
        try:
            start_1 = time.time()
            predicted_index, score, label = main(img, context, inputs, outputs, bindings, stream)
            count_time += time.time() - start_1
        except Exception as e:
            print(f"Skip invalid image: {img}, error: {e}")
            continue

        y_true.append(true_label)
        y_pred.append(label)
        if label == true_label:
            count += 1

    accuracy = count / len(img_list) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Correct predictions: {count}, Total images: {len(img_list)}")
    print(f"Time taken: {time.time() - start:.6f} seconds")
    print(f"Inference took {count_time} seconds for {len(img_list)} images")
