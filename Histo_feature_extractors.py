import torch
from torchvision import transforms
from torch import nn
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchvision.transforms as T
from torchinfo import summary
from torchvision.models import densenet121, DenseNet121_Weights
from collections import OrderedDict

from timm.models.vision_transformer import VisionTransformer as VTNew
from torchvision.models.resnet import Bottleneck, ResNet

import clip                       # for Clip
import open_clip                  # for Bioclip
import pathologyfoundation as pf  # for PLIP  

from models_utils.ConvNeXt.models.convnext import *
import models_utils.PathDino_small16_5Blocks_512 as PathDino
from models_utils.HistoSSLscaling.rl_benchmarks.models import iBOTViT
from models_utils.multitask_dipath.mtdp import build_model as build_MuDiPath


NORMALIZE_MEAN = IMAGENET_DEFAULT_MEAN
NORMALIZE_STD = IMAGENET_DEFAULT_STD
        
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

output_col = ["input_size","output_size","num_params"]


def get_pretrained_urlDino(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url
def vit_small_sslPathology(pretrained, progress, key, **kwargs):
    patch_size = kwargs.get("patch_size", 16)
    model = VTNew(
        img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0
    )
    if pretrained:
        pretrained_url = get_pretrained_urlDino(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
    return model


class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer
        
        # add a global avg pool after the last block
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # added by me
        x = self.gap(x)
        # flatten the representation
        x = torch.flatten(x, 1)
        
        return x
def get_pretrained_url_resnet(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url
def resnet50SSL(pretrained, progress, key, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_url = get_pretrained_url_resnet(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        print(verbose)
    return model


class fully_connected(nn.Module):
	"""docstring for BottleNeck"""
	def __init__(self, model, num_ftrs, num_classes):
		super(fully_connected, self).__init__()
		self.model = model
		self.fc_4 = nn.Linear(num_ftrs,num_classes)
	def forward(self, x):
		x = self.model(x)
		x = torch.flatten(x, 1)
		return  x

def getKimiaNet():
	model = densenet121(weights=DenseNet121_Weights.DEFAULT)
	for param in model.parameters():
		param.requires_grad = False
	model.features = nn.Sequential(model.features , nn.AdaptiveAvgPool2d(output_size=(1,1)))
	num_ftrs = model.classifier.in_features
	model_final = fully_connected(model.features, num_ftrs, 30)
	weights = torch.load('/mayo_atlas/home/m288756/mayo_ai_backupLast/src/models/weights/KimiaNetPyTorchWeights.pth')
 
	# Remove 'module.' prefix from layer names
	weights = {name.replace("module.", ""): tensor for name, tensor in weights.items()}
	model_final.load_state_dict(weights)
	return model_final


def get_feature_extractor(mdl="resnet50d", headless=True):
    print(f'loading {mdl} model...')
    if mdl == "resnet50d":
        model = timm.create_model('resnet50d', pretrained=True)
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        if headless:
            model.fc = nn.Identity()
        print(transform)
        print("ResNet50 model loaded.")
    elif mdl == "densenet121":
        model = timm.create_model('densenet121', pretrained=True)
        config = resolve_data_config({}, model=model)   
        transform = create_transform(**config)
        print(transform)
        if headless:
            model.classifier = nn.Identity()
            model.head_drop = nn.Identity()
        print("DenseNet121 model loaded.")
        print(transform)
    elif mdl == "mobilenetv3_large_100_miil_in21k": #mobilenetv3_large_100
        model = timm.create_model('mobilenetv3_large_100.miil_in21k', pretrained=True)
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        if headless:
            model.classifier = nn.Identity()
        print("mobilenetv3_large_100_miil_in21k model loaded.")
    elif mdl == "mobilenetv3_rw": #mobilenetv3_large_100
        model = timm.create_model('mobilenetv3_rw', pretrained=True)
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        if headless:
            model.classifier = nn.Identity()
        print("mobilenetv3_rw model loaded.")
    elif mdl == "efficientnet_b0":
        model = timm.create_model('efficientnet_b0', pretrained=True)
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        if headless:
            model.classifier = nn.Identity()
        print("efficientnet_b0 model loaded.")
    elif mdl == "efficientnet_b3":
        model = timm.create_model('efficientnet_b3', pretrained=True)
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        if headless:
            model.classifier = nn.Identity()
        print("efficientnet_b3 model loaded.")
    elif mdl=="efficientnet_b5":
        model = timm.create_model('efficientnet_b5', pretrained=True)
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        print(transform)
        if headless:
            model.classifier = nn.Identity()
        print("efficientnet_b5 model loaded.")
    elif mdl == "vit_base_patch16_224":
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])
        if headless:
            model.head = nn.Identity()
        # summary(model,input_size=(1,3,224,224),col_names=output_col,depth=5)
        print("vit_base_patch16_224 model loaded.")
    elif mdl == "dino_vits16":
        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])
        model = torch.hub.load('facebookresearch/dino:main', mdl)
        # summary(model,input_size=(1,3,224,224),col_names=output_col,depth=5)
        print("dino_vits16 model loaded.")
    elif mdl == "dino_HIPT":
        transform = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])
        model = torch.hub.load('facebookresearch/dino:main', "dino_vits16")

        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.to(device)
        pretrained_weights ='/mayo_atlas/home/m288756/mayo_ai_backupLast/src/models/weights/vit256_small_dino.pth'
        state_dict = torch.load(pretrained_weights, map_location="cpu")['teacher']
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print("HIPT model loaded.")
    elif mdl == "dino_vitb16":
        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        model = torch.hub.load('facebookresearch/dino:main', mdl)
        print("dino_vitb16 model loaded.")
    elif mdl == "dinov2_vitb14":
        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        print("dinov2_vitb14 model loaded.")
    elif mdl == "KimiaNet":
        model = getKimiaNet()
        if headless:
            model.fc_4 = nn.Identity()
        transform = transforms.Compose([
                transforms.Resize(1000),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        print("KimiaNet model loaded.")
    elif mdl == 'swav_resnet50':
        model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        if headless:
            model.fc = nn.Identity()
        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])
        print("SwAV model loaded.")
    elif mdl == 'convnext_base_22k': 
        model = convnext_base(pretrained=True, in_22k=True, num_classes=21841)
        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        model.head = nn.Identity() 
    elif mdl == 'convnext_xlarge_22k': 
        model = convnext_xlarge(pretrained=True, in_22k=True, num_classes=21841)
        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        model.head = nn.Identity()  
    elif 'clip' in mdl: # 'clip_ViT-B/16'  or  'clip_ViT-L/14@336px'
        modelName = mdl.split('_')[1]
        model, transform = clip.load(modelName)
    elif 'BiomedCLIP' in mdl: 
        model, _, transform = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        print("BiomedCLIP model loaded.")
    elif 'PLIP' in mdl: 
        model = pf.model_zoo("PLIP-ViT-B-32", device=device)
        transform = None
        print("PLIP model loaded.")
    elif mdl == 'iBOT_ViT_B':
        weights_path = "/mayo_atlas/home/m288756/mayo_ai/src/modelss/HistoSSLscaling/weights/ibot_vit_base_pancan.pth"
        model = iBOTViT(architecture="vit_base_pancan", encoder="student", weights_path=weights_path)
        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print("iBOT_ViT_B model loaded.")
    elif mdl == 'MuDiPath_densenet101':
        model = build_MuDiPath(arch="densenet121", pretrained="mtdp")
        transform = transforms.Compose([
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print("MuDiPath_densenet101 model loaded.")
    elif mdl == 'MuDiPath_resnet50':
        model = build_MuDiPath(arch="resnet50", pretrained="mtdp")
        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print("MuDiPath_resnet50 model loaded.")
    elif 'DinoSSLPathology' in mdl:
        # get the patch size from the last part of the mdl name 
        patch_size = int(mdl.split('_')[-1])
        if patch_size == 16:
            model = vit_small_sslPathology(pretrained=True, progress=False, key="DINO_p16", patch_size=16)
        else:
            model = vit_small_sslPathology(pretrained=True, progress=False, key="DINO_p8", patch_size=8)
        transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.70322989, 0.53606487, 0.66096631], [0.21716536, 0.26081574, 0.20723464])
            ])
        print("DinoSSLPathology model loaded.")
    elif mdl == 'ResNet_Barlow_Twins':
        model = resnet50SSL(pretrained=True, progress=False, key="BT")
        transform = transforms.Compose([
                transforms.Resize((1024, 768)),
                transforms.CenterCrop((1024, 768)),
                transforms.ToTensor(),
                transforms.Normalize([0.70322989, 0.53606487, 0.66096631], [0.21716536, 0.26081574, 0.20723464])
            ])
        print("ResNet_Barlow_Twins model loaded.")
    elif mdl == 'ResNet_MoCoV2':
        model = resnet50SSL(pretrained=True, progress=False, key="MoCoV2")
        transform = transforms.Compose([
                transforms.Resize((1024, 768)),
                transforms.CenterCrop((1024, 768)),
                transforms.ToTensor(),
                transforms.Normalize([0.70322989, 0.53606487, 0.66096631], [0.21716536, 0.26081574, 0.20723464])
            ])
        print("ResNet_MoCoV2 model loaded.")
    elif mdl == 'ResNet_SwAV':
        model = resnet50SSL(pretrained=True, progress=False, key="SwAV")
        transform = transforms.Compose([
                transforms.Resize((1024, 768)),
                transforms.CenterCrop((1024, 768)),
                transforms.ToTensor(),
                transforms.Normalize([0.70322989, 0.53606487, 0.66096631], [0.21716536, 0.26081574, 0.20723464])
            ])
        print("ResNet_SwAV model loaded.")
    elif mdl == 'PathDino':
        pretrained_weights ='/home/m288756/mayo_ai/src/modelss/PathDino/teacher_PathDino_5Blocks_512_headless.pth'
        model = PathDino.get_pathDino_model(pretrained_weights)
        transform = transforms.Compose([
                transforms.CenterCrop(512),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        print("PathDino model loaded.")
    elif mdl in ["hiera_base_224", "hiera_base_16x224"]:
        if mdl == "hiera_base_224":
            model = torch.hub.load("facebookresearch/hiera", model="hiera_base_224")
            if headless:
                model.head = nn.Identity()
        else:
            model = torch.hub.load("facebookresearch/hiera", model="hiera_base_16x224")
            if headless:
                model.head = nn.Identity()
            
        transform = transforms.Compose([
                    # transforms.Resize(256, interpolation=3),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])       
    else:
        raise ValueError(f"{mdl} feature extractor is not exist.")

    return model, transform
