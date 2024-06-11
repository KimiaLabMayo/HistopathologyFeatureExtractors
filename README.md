<!-- omit in toc -->
# Histopathology Feature Extractors (2024)
<!-- omit in toc -->
<!-- ### [PathDino Paper](https://arxiv.org/pdf/2311.08359.pdf) | [Supplementary](https://arxiv.org/pdf/2311.08359.pdf) | [Website](https://KimiaLabMayo.github.io/PathDino-Page/) | [Demo](https://huggingface.co/spaces/Saghir/PathDino) | [Dataset](https://portal.gdc.cancer.gov/repository?facetTab=files&filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.project.program.name%22%2C%22value%22%3A%5B%22TCGA%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.access%22%2C%22value%22%3A%5B%22open%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.data_format%22%2C%22value%22%3A%5B%22svs%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.experimental_strategy%22%2C%22value%22%3A%5B%22Diagnostic%20Slide%22%5D%7D%7D%5D%7D) -->


This repository is the largest repository that gathers histopathological image feature extractors in one place. 
This code has been used in our works: 

1- [Rotation-Agnostic Image Representation Learning for Digital Pathology](https://KimiaLabMayo.github.io/PathDino-Page/) (CVPR 2024)

2- [Foundation Models for Histopathology — Fanfare or Flair](https://www.sciencedirect.com/science/article/pii/S2949761224000142) (Mayo Clinic Proceedings: Digital Health)
<!-- omit in toc -->
## Contents

- [Models List](#models-list)
- [PathDino Inference on Histopathology Image](#pathdino-inference-on-histopathology-image)
- [Test All Models](#test-all-models)
- [Citation](#citation)


## Models List 
**PathDino** attention maps of its six heads.
<img width="1500" src="./images/Activation-Map-PathDino heads.gif">

This repository gathers most recent histopathology feature extractors. To get embeddings of a specific model, choose one of the following available models:

**Trained on natural data (images/text):**

1-        "resnet50d"

2-        "densenet121"

3-        "mobilenetv3_large_100_miil_in21k"

4-        "mobilenetv3_rw"

5-        "efficientnet_b0"

6-        "efficientnet_b3"

7-        "efficientnet_b5"

8-        "convnext_base_22k"

9-        "convnext_xlarge_22k"

10-        "vit_base_patch16_224"

11-        "dino_vits16"

12-        "dino_vitb16"

13-        "clip_ViT-B/16"

14-        "clip_ViT-L/14@336px"

15-        "dinov2_vitb14"

**Trained on histopathology data (images/text):**

16-        "PathDino"

17-        "iBOT_ViT_B"

18-        "DinoSSLPathology_16"

19-        "DinoSSLPathology_8" 

20-        "ResNet_Barlow_Twins"

21-        "ResNet_MoCoV2"

22-        "ResNet_SwAV"

23-        "dino_HIPT"

24-        "KimiaNet"

25-        "swav_resnet50"

26-        "PLIP"

27-        "MuDiPath_densenet101"

28-        "MuDiPath_resnet50"

29-        "BiomedCLIP"

**Note:** In our experiments we used "student" of iBOT_ViT_B, however, currently, it is recommended to use "teacher" in the [corresponding repository](https://github.com/owkin/HistoSSLscaling). For more details, please refer to the models' original repositories.

## PathDino Inference on Histopathology Image 
**Example: To extract image's embeddings using our model (PathDino):**

```python
import torch
from PIL import Image
from Histo_feature_extractors import get_feature_extractor

def infer_histImage(image_path, mdl):
    # Load the image
    image = Image.open(image_path)
        print(f"processing {mdl}...")
        # Get the model and its transform
        model, transform = get_feature_extractor(mdl)
        with torch.no_grad():
            if mdl.startswith("clip"):
                input_tensor = transform(image).unsqueeze(0).to(device)
                model.eval().to(device)
                embeddings = model.encode_image(input_tensor)
            elif mdl.startswith("BiomedCLIP"):
                input_tensor = transform(image).unsqueeze(0).to(device) 
                model.eval().to(device)
                embeddings = model(input_tensor)[0]
            elif mdl.startswith("PLIP"):
                embeddings = model.embed_images(image, normalize=True)
            else:
                input_tensor = transform(image).unsqueeze(0).to(device) 
                model.eval().to(device)
                embeddings = model(input_tensor)
        
        # Print the output embedding size
        print(f"{mdl} output embedding shape:", embeddings.shape)

model_name = "PathDino"
image_path = "images/img.png"
infer_histImage(image_path, model_name)
```

## Test All Models
To test all models, run the following code:
```
python Histo_embeddings.py
```

## Citation 
```
@article{alfasly2023rotationagnostic,
      title={Rotation-Agnostic Image Representation Learning for Digital Pathology}, 
      author={Saghir Alfasly and Abubakr Shafique and Peyman Nejat and Jibran Khan and Areej Alsaafin and Ghazal Alabtah and H.R. Tizhoosh},
      year={2023},
      eprint={2311.08359},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{ALFASLY2024165,
    title = {Foundation Models for Histopathology—Fanfare or Flair},
    author = {Saghir Alfasly and Peyman Nejat and Sobhan Hemati and Jibran Khan and Isaiah Lahr and Areej Alsaafin and Abubakr Shafique and Nneka Comfere and Dennis Murphree and Chady Meroueh and Saba Yasir and Aaron Mangold and Lisa Boardman and Vijay H. Shah and Joaquin J. Garcia and H.R. Tizhoosh},
    journal = {Mayo Clinic Proceedings: Digital Health},
    volume = {2},
    number = {1},
    pages = {165-174},
    year = {2024},
    issn = {2949-7612},
    doi = {https://doi.org/10.1016/j.mcpdig.2024.02.003}
}
```

**Acknowledgements** 
Some parts of some models' code are taken from their original repositories. We thank the authors for their great work. 

## Disclaimer
This code is intended for research purposes only. Any commercial use is prohibited.
