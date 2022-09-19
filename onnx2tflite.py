"""
Convert onnx model to tflite.

Usage:
  onnx2tflite --onnx-model <onnx_model_path> --test-image <test_image_path> --class-names <class_names_path> [--representative-images <representative_images_file_path> --representative-number <representative_images_number>] [--int8 | --full-int8 | --float16]
  onnx2tflite --version

Options:
  -h --help                                                     Show this screen.
  --version                                                     Show version.
  --onnx-model <onnx_model_path>                                ONNX model path. [type: path]
  --test-image <test_image_path>                                Test image path. [type: path]
  --int8                                                        Quantize model to int8.
  --float16                                                     Quantize model to float16.
  --full-int8                                                   Fully quantize model to int8.
  --representative-images <representative_images_file_path>     File containing paths to representative images for a full quantization. [type: path]
  --representative-number <representative_images_number>        Number of representative images for calibration [default: 100]. [type: int]
  --class-names <class_names_path>                              Path to the file containing class names. [type: path]

"""  # noqa: E501

from pathlib import Path
import random
import time
from typing import Any, Callable, Generator, List, Optional, Tuple
import tensorflow as tf
import onnx
import onnx.checker
import onnx.helper
import onnx_tf
from type_docopt import docopt
from google.protobuf.json_format import MessageToDict
import numpy as np
import numpy.typing as npt
import cv2


def load_onnx(model_path: Path) -> onnx.ModelProto:
    model = onnx.load(model_path.as_posix())
    onnx.checker.check_model(model=model, full_check=True)
    return model


def get_image_paths(image_list_filepath: Path) -> List[Path]:
    with open(image_list_filepath, "r") as image_list_file:
        return [Path(line.split("\n")[0]) for line in image_list_file.readlines()]


def representative_dataset(
    image_list_filepath: Path, representative_number: int, height: int, width: int
) -> Callable[[None], Generator[npt.NDArray[Any], None, None]]:
    def representative_dataset_gen() -> Generator[npt.NDArray[Any], None, None]:
        for image_path in random.sample(
            get_image_paths(image_list_filepath=image_list_filepath),
            representative_number,
        ):
            yield [load_image(image_path=image_path, height=height, width=width)]

    return representative_dataset_gen


def get_onnx_input_size(onnx_model: onnx.ModelProto) -> Tuple[int, int]:
    onnx_model_inputs = [
        MessageToDict(model_input) for model_input in onnx_model.graph.input
    ]
    input_shape = onnx_model_inputs[0]["type"]["tensorType"]["shape"]["dim"]
    return int(input_shape[2]["dimValue"]), int(input_shape[3]["dimValue"])


def get_tflite_output_path(
    onnx_model_path: Path, int8: bool, full_int8: bool, float16: bool, representative_data: bool
) -> Path:
    quantization_mode_string = ""
    if int8 or full_int8:
        quantization_mode_string = "{}int8{}".format(
            "full_" if full_int8 else "", "-data" if representative_data else ""
        )
    if float16:
        quantization_mode_string = "flt16"
    return onnx_model_path.parent / "{}-{}.tflite".format(
        onnx_model_path.stem, quantization_mode_string
    )


def onnx2tf(
    onnx_model: onnx.ModelProto, tf_output_path: Path, test_image: tf.Tensor
) -> tf.Tensor:
    tf_model = onnx_tf.backend.prepare(onnx_model, device="CPU")
    tf_model.export_graph(tf_output_path.as_posix())
    tf_model = tf.saved_model.load(tf_output_path.as_posix())
    tf_model.trainable = False
    return tf_model(images=test_image)


def tf2tflite(
    tf_output_path: Path,
    tflite_output_path: Path,
    height: int,
    width: int,
    int8: bool,
    full_int8: bool,
    float16: bool,
    representative_data: Optional[Path],
    representative_number: int,
) -> None:
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_output_path.as_posix())
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    if int8 or full_int8 or float16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if representative_data is not None:
        converter.representative_dataset = representative_dataset(
            image_list_filepath=representative_data,
            representative_number=representative_number,
            height=height,
            width=width,
        )
    if float16:
        converter.target_spec.supported_types = [tf.float16]
    if full_int8:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    with open(tflite_output_path, "wb") as tflite_file:
        tflite_file.write(tflite_model)


def test_tflite(tflite_export_path: Path, test_image: tf.Tensor) -> tf.Tensor:
    interpreter = tf.lite.Interpreter(model_path=tflite_export_path.as_posix())
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], test_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    return output_data[output_data[:, -1] >= 0.25]


def generate_color_map(class_names: List[str]) -> List[Tuple[int, int, int]]:
    return [
        (
            random.randrange(start=0, stop=256),
            random.randrange(start=0, stop=256),
            random.randrange(start=0, stop=256),
        )
        for _ in class_names
    ]


def bounding_box_coco_to_standard(
    coco_bounding_box: List[float],
) -> List[float]:
    x_min, y_min, w, h = coco_bounding_box
    return [x_min, y_min, x_min + w, y_min + h]


def visualize_bounding_boxes(
    image_path: Path,
    suffix: str,
    bounding_boxes: tf.Tensor,
    classes: tf.Tensor,
    scores: tf.Tensor,
    class_names: List[str],
    height: int,
    width: int,
) -> None:
    image = cv2.resize(
        cv2.imread(image_path.as_posix()),
        (width, height),
    )
    color_map = generate_color_map(class_names=class_names)
    for bounding_box, class_id, score in zip(bounding_boxes, classes, scores):
        class_id = int(class_id)
        image = cv2.rectangle(
            img=image,
            pt1=(int(bounding_box[0]), int(bounding_box[1])),
            pt2=(int(bounding_box[2]), int(bounding_box[3])),
            color=color_map[class_id],
            thickness=2,
        )
        image = cv2.putText(
            img=image,
            text=class_names[class_id] + ": {:.2f}".format(score),
            org=(int(bounding_box[0]), min(max(int(bounding_box[1]) - 5, 0), height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.25,
            thickness=1,
            color=color_map[class_id],
        )
    cv2.imwrite(
        filename=(Path("tmp") / (image_path.stem + suffix + ".jpg")).as_posix(),
        img=image,
    )


def load_image(
    image_path: Path, height: int, width: int, int8: bool = False
) -> npt.NDArray[Any]:
    image = (
        np.array(
            [
                cv2.resize(
                    cv2.imread(image_path.as_posix()),
                    (width, height),
                )
            ]
        )
        .astype(np.float32)
        .transpose(0, 3, 1, 2)
    )
    if int8:
        return image.astype(np.uint8)
    else:
        return image / 255.0


def get_tf_output_path(onnx_model_path: Path) -> Path:
    temporary_directory_path = Path("tmp")
    if not temporary_directory_path.exists():
        temporary_directory_path.mkdir()
    return temporary_directory_path / (onnx_model_path.stem + ".pb")


def read_class_names(class_names_path: Path) -> List[str]:
    with open(class_names_path, "r") as class_names_file:
        return [line.split("\n")[0] for line in class_names_file.readlines()]


def main():
    arguments = docopt(__doc__, version="0.1", types={"path": Path})
    print(arguments)
    if arguments["--full-int8"] and arguments["--representative-images"] is None:
        raise RuntimeError(
            "Cannot convert to a fully int8 quantized model if no representative dataset "
            "is provided."
        )

    onnx_model = load_onnx(model_path=arguments["--onnx-model"])
    height, width = get_onnx_input_size(onnx_model=onnx_model)

    class_names = read_class_names(class_names_path=arguments["--class-names"])

    tf_output_path = get_tf_output_path(onnx_model_path=arguments["--onnx-model"])
    tf_result = onnx2tf(
        onnx_model=onnx_model,
        tf_output_path=tf_output_path,
        test_image=load_image(
            image_path=arguments["--test-image"], height=height, width=width
        ),
    )
    visualize_bounding_boxes(
        image_path=arguments["--test-image"],
        suffix="-tf",
        bounding_boxes=tf_result["output"][:, 1:-2],
        classes=tf_result["output"][:, -2],
        scores=tf_result["output"][:, -1],
        class_names=class_names,
        height=height,
        width=width,
    )

    tflite_output_path = get_tflite_output_path(
        onnx_model_path=arguments["--onnx-model"],
        int8=arguments["--int8"],
        full_int8=arguments["--full-int8"],
        float16=arguments["--float16"],
        representative_data=(arguments["--representative-images"] is not None),
    )

    tf2tflite(
        tf_output_path=tf_output_path,
        tflite_output_path=tflite_output_path,
        height=height,
        width=width,
        int8=arguments["--int8"],
        full_int8=arguments["--full-int8"],
        float16=arguments["--float16"],
        representative_data=arguments["--representative-images"],
        representative_number=arguments["--representative-number"],
    )
    start_time = time.time()
    tflite_result = test_tflite(
        tflite_export_path=tflite_output_path,
        test_image=load_image(
            image_path=arguments["--test-image"],
            height=height,
            width=width,
            int8=arguments["--full-int8"],
        ),
    )
    print(time.time() - start_time)
    visualize_bounding_boxes(
        image_path=arguments["--test-image"],
        suffix="-tflite",
        bounding_boxes=tflite_result[:, 1:-2],
        classes=tflite_result[:, -2],
        scores=tflite_result[:, -1],
        class_names=class_names,
        height=height,
        width=width,
    )


if __name__ == "__main__":
    with tf.device("cpu"):
        main()
