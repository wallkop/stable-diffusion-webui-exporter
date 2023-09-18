import base64
import contextlib
import json
from io import BytesIO
from PIL import Image
import gradio as gr
import pickle
from modules import scripts
import gzip

TYPE_IMAGE = "Image"
TYPE_STR = "str"
TYPE_OBJ = "object"


def compress_base64(data):
    buffer = BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
        f.write(data)
    compressed_data = buffer.getvalue()
    compressed_base64 = base64.b64encode(compressed_data).decode('utf-8')
    return compressed_base64


def decompress_base64(compressed_base64):
    compressed_data = base64.b64decode(compressed_base64)
    buffer = BytesIO(compressed_data)
    with gzip.GzipFile(fileobj=buffer, mode='rb') as f:
        decompressed_data = f.read()
    return decompressed_data


def base64_to_image(base64_str):
    img_bytes = decompress_base64(base64_str)
    img_file = BytesIO(img_bytes)
    img = Image.open(img_file)
    return img


def image_to_base64(img, format="PNG") -> object:
    buffered = BytesIO()
    img.save(buffered, format=format)
    img_str = compress_base64(buffered.getvalue())
    return img_str


def image_mask_to_base64(img_array):
    image_pil = Image.fromarray(img_array['image'], "RGB")
    alpha_pil = Image.fromarray(img_array['mask'][:, :, 3], 'L')
    image_pil.putalpha(alpha_pil)
    return image_to_base64(image_pil)


def export_data(*args):
    i = 0
    result = {}
    for key in getParamsPlugin.args_params.keys():
        if i >= len(args):
            break
        value = args[i]
        i += 1
        field_type = TYPE_STR
        object_type = str(type(value))

        if "<class 'PIL.Image.Image'>" == object_type:
            value = image_to_base64(value)
            field_type = TYPE_IMAGE
        elif "<class 'dict'>" == object_type and "image" in value and "mask" in value:
            value = image_mask_to_base64(value)
            field_type = TYPE_IMAGE
        elif "<class 'str'>" != object_type:
            value = compress_base64(pickle.dumps(value))
            field_type = TYPE_OBJ
        result[key] = {"v": value, "t": field_type, "o": object_type}
    json_str = json.dumps(result, indent=4)
    filename = "export-data.txt"
    with open(filename, "w") as file:
        file.write(json_str)

    return filename


def import_data(upload_file):
    file_path = upload_file.name
    with open(file_path, "r") as file:
        content = file.read()
    json_data = json.loads(content)
    result = []
    for key in sorted(json_data.keys(), key=int):
        value = json_data[key]
        v = value["v"]
        t = value["t"]
        if t == TYPE_IMAGE:
            v = base64_to_image(v)
        elif t == TYPE_OBJ:
            bv = decompress_base64(v)
            v = pickle.loads(bv)
        result.append(v)
    return result


class getParamsPlugin(scripts.Script):

    args_params = {}

    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "get-all-params"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):

        args_list = []
        for ele in self.args_params.values():
            args_list.append(ele)

        with gr.Group():
            with gr.Accordion("模型参数管理", open=False):
                export_button = gr.Button(value="导出参数", variant='primary')
                upload_button = gr.UploadButton("上传参数")
                download_file = gr.outputs.File(label="模型参数文件下载")

        with contextlib.suppress(AttributeError):
            export_button.click(fn=export_data, inputs=args_list, outputs=download_file)
            upload_button.upload(fn=import_data, inputs=upload_button, outputs=args_list)

        return [download_file, export_button, upload_button]

    def after_component(self, component, **kwargs):
        self.args_params[component._id] = component

    def postprocess(self, p, processed, *args):
        if len(processed.info) == 0:
            return p
        print('----------------custom extension postprocess')
        print(p)