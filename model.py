import os
import requests
from label_studio.core.utils.params import get_env
from label_studio_ml.model import LabelStudioMLBase

os.environ["LABEL_STUDIO_ML_BACKEND_V2"] = "True"
import base64
import json
import cv2
import secrets
import string
from label_studio.core.settings.base import DATA_UNDEFINED_NAME
from label_studio_ml.utils import get_single_tag_keys
from paddlenlp import Taskflow

CONNECTION_TIMEOUT = float(get_env('ML_CONNECTION_TIMEOUT', 10))
HOSTNAME = get_env('HOSTNAME', 'http://localhost:8080')
API_KEY = get_env('API_KEY', "fb789c82ed2ff2b1a5c81d36259208b40e12a738")
HOSTNAME = "http://localhost:8080"
API_KEY = "fb789c82ed2ff2b1a5c81d36259208b40e12a738"

class NewModel(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super(NewModel, self).__init__(**kwargs)
        self.model = None
        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config, 'RectangleLabels', 'Image')
        self.labels_in_config = set(self.labels_in_config)
        
        # save label file
        self.save_labels_config()

    def _init(self):
        """Initialize the model if it is not already initialized."""
        if self.model is None:
            self.model = Taskflow("information_extraction", schema=self.labels_in_config,
                                  task_path='./checkpoint/model_best')
    
    def save_labels_config(self):
        with open('labels_config.txt', 'w') as file:
            for label in self.labels_in_config:
                file.write(label + '\n')

    def generate_random_id(self, length=12):
        """
        Generate a random ID composed of letters, digits, and underscores.
        Args:
            length (int): Length of the generated random ID.
        Returns:
            str: Randomly generated ID.
        """
        characters = string.ascii_letters + string.digits + "_"
        random_id = ''.join(secrets.choice(characters) for _ in range(length))
        return random_id

    def _inference_detector(self, image_path):
        """Inference on a single image using a detector.
        Args:
            image_path (str): Path to the image file.
        Returns:
            tuple: Tuple containing inference results, original image height, and original image width.
        """
        img = cv2.imread(image_path)
        orig_height, orig_width, _ = img.shape
        # Convert the image to Base64 encoding
        _, img_encoded = cv2.imencode(".jpg", img)
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")
        # Pass Base64 encoded image to the inference engine 'model'
        results = self.model({"doc": img_base64})
        return results, orig_height, orig_width

    def _get_image_url(self, task):
        """Get the image URL from the task data.
        Args:
            task (dict): Task data.
        Returns:
            str: Image URL.
        """
        image_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
        return image_url

    def predict(self, tasks, **kwargs):
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :return predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Raw-JSON-format-of-completed-tasks)
        """
        print("predict start......")
        self._init()
        from_name = self.from_name
        to_name = self.to_name
        task = tasks[0]
        # Get image URL and local path
        image_url = self._get_image_url(task)
        image_path = self.get_local_path(image_url, "/images")

        # Run inference on the detector
        model_results, original_height, original_width = self._inference_detector(image_path)

        # Prepare annotation results
        result = []
        predictions = []

        for item_list in model_results:
            for label, annotations in item_list.items():
                for annotation in annotations:
                    # Create result item
                    result_item = {
                        "original_width": original_width,
                        "original_height": original_height,
                        "image_rotation": 0,
                        "value": {
                            "x": (annotation['bbox'][0][0] / original_width) * 100,
                            "y": (annotation['bbox'][0][1] / original_height) * 100,
                            "width": ((annotation['bbox'][0][2] - annotation['bbox'][0][0]) / original_width) * 100,
                            "height": ((annotation['bbox'][0][3] - annotation['bbox'][0][1]) / original_height) * 100,
                            "rotation": 0,
                            "rectanglelabels": [label]
                        },
                        "id": self.generate_random_id(),
                        "from_name": from_name,
                        "to_name": to_name,
                        "type": "rectanglelabels",
                        "origin": "manual"
                    }
                    result.append(result_item)

            # Sort result based on x-coordinate
            result = sorted(result, key=lambda k: k["value"]["x"])

            # Append predictions
            predictions.append({
                'result': result,
                'model_version': 'BOSHLAND V1.0'
            })
        print("predict end  ......")
        return predictions

    def fit(self, annotations, workdir=None, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        """
        print("fit start......")
        project_id = kwargs['data']['project']['id']
        download_url = f"{HOSTNAME}/api/projects/{project_id}/export?exportType=JSON"
        print("url:", download_url)
        response = requests.get(download_url, headers={'Authorization': f'Token {API_KEY}'})
        if response.status_code != 200:
            raise Exception(f"Can't load task data using {download_url}, "
                            f"response status_code = {response.status_code}")

        annotations = response.json()
        with open("./label_studio_update.json", "w", encoding="utf-8") as outfile:
            json.dump(annotations, outfile, ensure_ascii=False)

        # # Run Label Studio for data preprocessing
        # [Parameter details](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/information_extraction/document)
        os.system('python label_studio.py \
                 --label_studio_file ./label_studio_update.json \
                --task_type "ext" \
                --save_dir ./data \
                --splits 0.8 0.1 0.1')
        # Run fine-tuning script
        # [Parameter details](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/information_extraction/document)
        os.system('python finetune.py  \
                         --device cpu \
                         --logging_steps 50 \
                         --save_steps 50 \
                         --eval_steps 50 \
                         --seed 42 \
                         --model_name_or_path uie-x-base \
                         --output_dir ./checkpoint/model_best \
                         --train_path data/train.txt \
                         --dev_path data/dev.txt  \
                         --max_seq_len 512  \
                         --per_device_train_batch_size  1 \
                         --per_device_eval_batch_size 1 \
                         --num_train_epochs 5 \
                         --learning_rate 1e-5 \
                         --do_train \
                         --do_eval \
                         --do_export \
                         --export_model_dir ./checkpoint/model_best \
                         --overwrite_output_dir \
                         --disable_tqdm True \
                         --metric_for_best_model eval_f1 \
                         --load_best_model_at_end  True \
                         --save_total_limit 1')
        print('fit() completed successfully.')
        return {
            'path': workdir
        }
