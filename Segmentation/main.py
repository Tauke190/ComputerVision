import pixellib
from pixellib.torchbackend.instance import instanceSegmentation

ins = instanceSegmentation()
ins.load_model("pointrend_resnet50.pkl")
results, output = ins.segmentImage("hand.png", show_bboxes=False, output_image_name="output_image.jpg")

print(results["class_names"])