import json


# slagcar渣土车
# car 小汽车：轿车、跑车、SUV、商务车、小型面包车等四轮小汽车
# tricar 三轮小汽车：如老年代步三轮车，三轮自行车、三轮摩托车和三轮电动车
# motorbike摩托车：含电瓶车、电动车
# bicycle 两轮自行车：不含三轮自行车
# bus公共汽车、客车
# truck大型货车、小型货车
# tractor 拖拉机
def init():
    """Initialize model
        Returns: model
        """
    return {}


def process_image(handle=None, input_image=None, args=None, **kwargs):
    """Do inference to analysis input_image and get output
        Attributes:
            handle: algorithm handle returned by init()
            input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
        Returns: process result
        """
    # Process image here

    fake_result = {
        'objects': []
    }
    fake_result['objects'].append({
        "algorithm_data": {
            "is_alert": False,
            "target_count": 0,
            "target_info": []
        },
        "model_data": {
            "objects": []
        }
    })
    return json.dumps(fake_result, indent=4)
