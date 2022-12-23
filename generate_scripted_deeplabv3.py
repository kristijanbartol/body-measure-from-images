import torch
from PIL import Image
from torchvision import transforms
from torch.utils.mobile_optimizer import optimize_for_mobile

# use deeplabv3_resnet50 instead of resnet101 to reduce the model size
model = torch.hub.load('pytorch/vision:v0.7.0', 'deeplabv3_resnet50', pretrained=True)
model.eval()

scriptedm = torch.jit.script(model)
optimized_traced_model = optimize_for_mobile(scriptedm)
optimized_traced_model._save_for_lite_interpreter("deeplabv3_scripted_optimized.ptl")


input_image = Image.open("deeplab.jpg")
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)
with torch.no_grad():
    output = model(input_batch)['out'][0]

print(input_batch.shape)
print(output.shape)
