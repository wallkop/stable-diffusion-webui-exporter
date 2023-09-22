import base64
import contextlib
import json
from io import BytesIO
from PIL import Image
import gradio as gr
import pickle
from modules import scripts
from modules.api import api
import gzip
import math
import numpy as np
import tempfile

TYPE_IMAGE = "Image"
TYPE_IMAGE_DICT = "ImageDict"
TYPE_STR = "str"
TYPE_INT = "int"
TYPE_OBJ = "object"

BLACK_COMPONENT_TYPE_LIST = ["state"]


def json_serializable(obj):
    result = {}
    for key, value in vars(obj).items():
        try:
            json.dumps({key: value})
            result[key] = value
        except TypeError:
            result[key] = str(value)

    return result


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


def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="png")
    img_str = compress_base64(buffered.getvalue())
    return img_str


def image_dict_to_base64(img_array):
    image_pil = Image.fromarray(img_array['image'])
    result = api.encode_pil_to_base64(image_pil).decode('utf-8')
    return result


def base64_to_image_dict(base64_str):
    result = api.decode_base64_to_image(base64_str)
    result = np.array(result)
    return result


def export_data(*args):
    i = 0
    result = {}
    for key in exporterPlugin.args_params.keys():
        if i >= len(args):
            break
        value = args[i]
        item = exporterPlugin.args_params[key]
        item_name = str(item)
        i += 1

        # prepare
        if item_name == "dropdown" and isinstance(value, int):
            value = item.choices[value]
        if item_name == "radio":
            if isinstance(item.choices, list) and item.type == 'index':
                if isinstance(value, int):
                    value = item.choices[value]

        field_type = TYPE_STR
        object_type = str(type(value))

        if "<class 'PIL.Image.Image'>" == object_type:
            value = image_to_base64(value)
            field_type = TYPE_IMAGE
        elif "<class 'dict'>" == object_type and "image" in value and "mask" in value:
            value = image_dict_to_base64(value)
            field_type = TYPE_IMAGE_DICT
        elif "<class 'int'>" == object_type:
            object_type = TYPE_INT
        elif "<class 'str'>" != object_type:
            try:
                value = compress_base64(pickle.dumps(value))
            except:
                pass
            field_type = TYPE_OBJ

        #result[key] = {"v": value, "t": field_type, "o": object_type, "name": item_name, "it": json_serializable(item)}
        result[key] = {"v": value, "t": field_type, "o": object_type, "name": item_name}
    json_str = json.dumps(result, indent=4)
    filename = "export-ui-params.json"
    with open(filename, "w") as file:
        file.write(json_str)
    return filename


def reset_params(content):
    json_data = json.loads(content)
    result = []
    for key in sorted(json_data.keys(), key=int):
        value = json_data[key]
        v = value["v"]
        t = value["t"]
        if t == TYPE_IMAGE:
            v = base64_to_image(v)
        elif t == TYPE_IMAGE_DICT:
            v = base64_to_image_dict(v)
        elif t == TYPE_INT:
            pass
        elif t == TYPE_OBJ:
            bv = decompress_base64(v)
            v = pickle.loads(bv)
        result.append(v)
    return result


def import_data(upload_file):
    file_path = upload_file.name
    with open(file_path, "r") as file:
        content = file.read()
    filename = "import-ui-params.json"
    with open(filename, "w") as file:
        file.write(content)
    result = reset_params(content)
    return result


def refresh_data():
    with open("import-ui-params.json", "r") as file:
        content = file.read()
    result = reset_params(content)
    return result


def download_json():
    if exporterPlugin.is_ran:
        exporterPlugin.is_ran = False
        return "export-exec-params.json"
    else:
        gr.Warning("请生成图片后, 再导出运行参数")


class exporterPlugin(scripts.Script):

    args_params = {}

    exec_params = ""

    download_file_obj = None

    is_ran = False

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
            with gr.Accordion("WebUI-参数管理插件", open=False):
                with gr.Blocks():
                    gr.Markdown(
"""
### WebUI-参数管理插件使用说明

#### 一.导出运行参数
```angular2html
该功能主要是为了快速导出JSON提交到SD后台批量训练。

1.需要先进行一次图片生成, 否则无法导出。
2.点击 [导出-运行参数JSON] 按钮后, 即生成本次运行的JSON。
3.点击下方 [参数下载] 文件区域生成的链接即可下载。
```

#### 二.导出/上传UI参数
```angular2html
该功能主要是为了保存记录当前WebUI的参数, 并在以后进行导入还原。

【如何导出】
1.点击 [导出-UI参数文件] 按钮, 即生成当前WebUI界面的参数。
2.点击下方 [参数下载] 文件区域生成的链接即可下载。

【如何上传】
1.点击 [上传-UI参数文件] 按钮, 弹出文件框后选择对应的参数文件。
2.点击 [确认上传] 按钮, 让导入的文件参数生效。
```

"""
                    )
                    with gr.Column():
                        with gr.Row():
                            upload_button = gr.UploadButton("上传-UI参数文件")
                            refresh_button = gr.Button(value="确认上传(传完一定要点这)", variant='primary')
                        with gr.Row():
                            export_button = gr.Button(value="导出-UI参数文件")
                            download_button = gr.Button(value="导出-运行参数JSON", variant='stop')
                        with gr.Row():
                            download_file = gr.outputs.File(label="参数下载")
                        self.download_file_obj = download_file

        with contextlib.suppress(AttributeError):
            export_button.click(fn=export_data, inputs=args_list, outputs=download_file)
            upload_button.upload(fn=import_data, inputs=upload_button, outputs=args_list)
            download_button.click(fn=download_json, inputs=[], outputs=download_file)
            refresh_button.click(fn=refresh_data, inputs=[], outputs=args_list)

        return [download_file, export_button, upload_button, download_button]


    def after_component(self, component, **kwargs):
        if str(component) not in BLACK_COMPONENT_TYPE_LIST:
            self.args_params[component._id] = component


    def before_process(self, p, *args):
        # set tempfile to None
        index = 0
        for v in p.__dict__["script_args"]:
            if isinstance(v, tempfile._TemporaryFileWrapper):
                tmp_list = list(p.__dict__["script_args"])
                tmp_list[index] = None
                p.__dict__["script_args"] = tuple(tmp_list)
                print(">>> Set p.__dict__[\"script_args\"][%s] = None" % index)
            index += 1


    def postprocess(self, p, processed, *args):

        if len(processed.info) == 0:
            return

        allProcessKeys = p.__dict__.keys()
        execParam = {
            "script_args": [""],
            "alwayson_scripts": {},
        }
        for key in allProcessKeys:
            value = p.__dict__[key]
            if (isinstance(value, (int, float, bool, complex)) and not math.isinf(value)) or isinstance(value, str):
                execParam[key] = value
            elif key == 'init_images':
                # init_images = []
                # for image in value:
                #     init_images.append(api.encode_pil_to_base64(image).decode('utf-8'))
                execParam[key] = ["{{PPP}}"]

        for script in p.scripts.scripts:
            scriptTitle = script.title()
            scriptArgs = p.script_args[script.args_from : script.args_to]
            if scriptTitle == 'ControlNet':
                execParam['alwayson_scripts']['controlnet'] = {
                    "args": []
                }
                for scriptArg in scriptArgs:
                    try:
                        if not scriptArg.__dict__['enabled']:
                            continue
                        controlnetParams = {
                            # "mask": None,
                            # "module": scriptArg.module,
                            # "model": scriptArg.model,
                            # "weight": scriptArg.weight,
                            # "resize_mode": scriptArg.resize_mode,
                            # "control_mode": scriptArg.control_mode,
                            # "pixel_perfect": scriptArg.pixel_perfect,
                            # "guidance_start": scriptArg.guidance_start,
                            # "guidance_end": scriptArg.guidance_end,
                            # "threshold_a": scriptArg.threshold_a,
                            # "threshold_b": scriptArg.threshold_b,
                            # "processor_res": scriptArg.processor_res,
                            # "lowvram": scriptArg.low_vram,  # 有疑问
                        }

                        allControlnetKeys = scriptArg.__dict__.keys()
                        for key in allControlnetKeys:
                            value = scriptArg.__dict__[key]
                            if (isinstance(value, (int, float, bool, complex)) and not math.isinf(value)) or isinstance(
                                    value, str):
                                controlnetParams[key] = value
                        if scriptArg.image:
                            controlnetParams['mask'] = None
                            if 'image' in scriptArg.image:
                                pil = Image.fromarray(scriptArg.image['image'])
                                controlnetParams['input_image'] = api.encode_pil_to_base64(pil).decode('utf-8')
                            else:
                                controlnetParams['input_image'] = None

                        execParam['alwayson_scripts']['controlnet']['args'].append(controlnetParams)
                    except Exception as e:
                        print('Exception:', e)

            if scriptTitle == 'ADetailer' and scriptArgs[0] is True:
                dict1 = scriptArgs[1]
                dict1.pop('is_api')
                execParam['alwayson_scripts']['adetailer'] = {
                    "args": [scriptArgs[0], dict1]
                }

        json_str = json.dumps(execParam, indent=4)
        filename = "export-exec-params.json"
        with open(filename, "w") as file:
            file.write(json_str)

        exporterPlugin.is_ran = True

        return p