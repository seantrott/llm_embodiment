from src.utils import ViltHandler, BridgetowerHandler, OpenClipHandler

image_paths = ["data/pecher2006/images_processed/apple.png",
               "data/pecher2006/images_processed/applepie.png"]
texts = ["An apple", "An apple pie"]
modalities = ["text", "vision"]


image_paths = [
    "data/pecher2006/images/airplanewheels.jpg",
    "data/pecher2006/images/airplanesky.jpg"

]
vilt_handler = ViltHandler("vilt")
vilt_handler.compare_pair(texts, image_paths, modalities)


bt_handler = BridgetowerHandler("bridgetower")
bt_handler.compare_pair(texts, image_paths, modalities)

b32 = OpenClipHandler("ViT-B-32")
r = b32.compare_pair(texts, image_paths, modalities)

r.tolist()
