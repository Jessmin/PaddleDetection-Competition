import json

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
