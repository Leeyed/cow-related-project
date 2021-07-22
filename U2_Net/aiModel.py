# classfy
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from seresnet import se_resnet50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_class_names(class_name_path):
    names_list = []
    with open(class_name_path, 'r') as data:
        for ID, name in enumerate(data):
            names_list.append(name.strip('\n'))
    return names_list



class Classfy_model:
    def __init__(self, model_path, class_path, input_size, threshold=0.5):
        self.obj_list = read_class_names(class_path)

        self.classfy_model = se_resnet50(pretrained=None)
        if input_size[1] == 128:
            self.classfy_model.avg_pool = nn.AvgPool2d(4, stride=1)
        elif input_size[1] == 320:
            self.classfy_model.avg_pool = nn.AvgPool2d(10, stride=1)

        self.classfy_model.last_linear = nn.Linear(2048, len(self.obj_list))
        self.classfy_model.load_state_dict(torch.load(model_path))
        self.classfy_model = self.classfy_model.cuda()
        self.classfy_model.eval()
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.input_size = input_size  # [3, 320, 320]
        self.threshold = threshold

    def letterbox_image(self, image, size):
        '''resize image with unchanged aspect ratio using padding'''
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        shift = [(w - nw) // 2, (h - nh) // 2]
        return new_image, scale, shift
    def forward(self, img):
        im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('RGB')
        new_img, _, _ = self.letterbox_image(im, self.input_size[1:])
        # add
        input_data = self.test_transform(new_img)
        input_data = input_data.unsqueeze(0).to(device)
        output = self.classfy_model(input_data)
        output = F.softmax(output, dim=1)
        index = output.cpu().data.numpy().argmax()
        score = float(output.cpu().data.numpy().max())
        if score < self.threshold:
            return "", score
        label = self.obj_list[index]
        return label, score