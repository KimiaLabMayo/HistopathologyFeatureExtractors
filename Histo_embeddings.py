import torch
from PIL import Image
from Histo_feature_extractors import get_feature_extractor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(image_path, model_names):
    # Load the image
    image = Image.open(image_path)
    for i, mdl in enumerate(model_names):
        print(f"{i} - processing {mdl}... ---------------------------------------")
        # Get the model and its transform
        model, transform = get_feature_extractor(mdl)
        with torch.no_grad():
            if mdl.startswith("clip"):
                input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
                model.eval().to(device)
                embeddings = model.encode_image(input_tensor)
            elif mdl.startswith("BiomedCLIP"):
                input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
                model.eval().to(device)
                embeddings = model(input_tensor)[0]
            elif mdl.startswith("PLIP"):
                embeddings = model.embed_images(image, normalize=True)
            else:
                input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
                model.eval().to(device)
                embeddings = model(input_tensor)
        
        # Print the output embedding size
        print(f"{i} - {mdl} output embedding shape:", embeddings.shape)
        

if __name__ == "__main__":
    # Example usage
    image_path = "images/img.png"
    model_names = [
        "resnet50d",
        "densenet121",
        "mobilenetv3_large_100_miil_in21k",
        "mobilenetv3_rw",
        "efficientnet_b0",
        "efficientnet_b3",
        "efficientnet_b5",
        "vit_base_patch16_224",
        "dino_vits16",
        "dino_HIPT",
        "dino_vitb16",
        "dinov2_vitb14",
        "KimiaNet",
        "swav_resnet50",
        "convnext_base_22k",
        "convnext_xlarge_22k",
        "clip_ViT-B/16",
        "clip_ViT-L/14@336px",  
        "PLIP",
        "iBOT_ViT_B",
        "MuDiPath_densenet101",
        "MuDiPath_resnet50",
        "DinoSSLPathology_16",  
        "DinoSSLPathology_8",  
        "ResNet_Barlow_Twins",
        "ResNet_MoCoV2",
        "ResNet_SwAV",
        "PathDino",
        "BiomedCLIP"
    ]
    main(image_path, model_names)
    

