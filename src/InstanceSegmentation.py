from src.Segmentation import Segmentation


class InstanceSegmentation:
    def __init__(self, image, segmented=False):
        if not segmented:
            segmentation = Segmentation()
            self.mask = segmentation.run()
        else:
            self.mask = image

    def run(self):
        pass
            