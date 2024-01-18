import time
import requests
from PIL import Image
from io import BytesIO
import os

from paddlenlp import Taskflow
from pinferencia import Server
import base64

# init
boshland_info = ["产品名称", "客户", "批号", "车号", "规格", "数量", "报告日期",
                 "检验项目", "指标", "检验结果", "检验方法", "结论", "检验人", "审核人"]
boshland = Taskflow("information_extraction", model="uie-x-base", schema=boshland_info, task_path='./checkpoint/model_best')
max_length = 20
y_threshold = 25  # 阈值

def is_base64(s):
    try:
        # 尝试解码，如果成功则为 base64 编码
        base64.b64decode(s)
        return True
    except Exception:
        return False

def is_url(s):
    # 判断是否为 URL
    return s.startswith('http://') or s.startswith('https://')

def image_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        image_binary = image_file.read()
        encoded_image = base64.b64encode(image_binary)
        return encoded_image.decode("utf-8")

def image_url_to_base64(url):
    response = requests.get(url)
    image_binary = BytesIO(response.content).read()
    encoded_image = base64.b64encode(image_binary)
    return encoded_image.decode("utf-8")

def is_base64(data):
    try:
        # Attempt to decode the data
        decoded_data = base64.b64decode(data)
        # If successful, return True
        return True
    except Exception:
        # If decoding fails, return False
        return False

def is_same_row(bbox1, bbox2):
    center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
    center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
    return abs(center1[1] - center2[1]) <= y_threshold


def get_text(item):
    return item[0].get('text') if item else ' '


def predict(data):
    if is_base64(data):
        # 如果已经是 base64 编码，则直接使用
        print("base64")
        img_base64 = data
    elif is_url(data):
        # 如果是网络图片，则将其下载并转换为 base64 编码
        print("url image")
        img_base64 = image_url_to_base64(data)
    else:
        # 如果是本地图片，则读取并转换为 base64 编码
        print("local image")
        img_base64 = image_to_base64(data)
    # 记录推理开始时间
    start_time = time.time()
    results = boshland({"doc": img_base64})
    # 记录推理结束时间
    end_time = time.time()
    # 计算推理时间
    inference_time = end_time - start_time
    processed_results = []

    for data in results:
        product_name = get_text(data.get('产品名称', []))
        customer = get_text(data.get('客户', []))
        batch_number = get_text(data.get('批号', []))
        car_number = get_text(data.get('车号', []))
        specification = get_text(data.get('规格', []))
        quantity = get_text(data.get('数量', []))
        report_date = get_text(data.get('报告日期', []))
        inspector = get_text(data.get('检验人', []))
        reviewer = get_text(data.get('审核人', []))
        conclusion = get_text(data.get('结论', []))

        # 处理 conclusion 字段
        if conclusion and any(keyword in conclusion for keyword in ["pass", "合格", "符合", "通过"]):
            conclusion = "合格"
        else:
            conclusion = ""

        header_row = {
            '产品名称': product_name,
            '客户': customer,
            '批号': batch_number,
            '车号': car_number,
            '规格': specification,
            '数量': quantity,
            '报告日期': report_date,
            '检验人': inspector,
            '审核人': reviewer,
            '结论': conclusion
        }

        item_rows = []
        for item in data.get('检验项目', []):
            corresponding_indicator = [indicator for indicator in data.get('指标', []) if
                                       is_same_row(item['bbox'][0], indicator['bbox'][0])]
            corresponding_result = [result for result in data.get('检验结果', []) if
                                    is_same_row(item['bbox'][0], result['bbox'][0])]
            corresponding_method = [method for method in data.get('检验方法', []) if
                                    is_same_row(item['bbox'][0], method['bbox'][0])]

            inspection_item = item['text']
            indicator_text = get_text(corresponding_indicator)
            result_text = get_text(corresponding_result)
            method_text = get_text(corresponding_method)

            item_row = {
                '检验项目': inspection_item,
                '指标': indicator_text,
                '检验结果': result_text,
                '检验方法': method_text
            }

            item_rows.append(item_row)

        processed_results.append({
            'header': header_row,
            'items': item_rows,
            'time': inference_time,
        })

    return processed_results


service = Server()
service.register(model_name="boshland", model=predict)
