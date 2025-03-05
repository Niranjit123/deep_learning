import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=2):
        """
        Vision Transformer (ViT) model for face classification
        
        Args:
            num_classes (int): Number of output classes (default is 2 for local/non-local)
        """
        super(VisionTransformer, self).__init__()
        
      
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
       
        for param in self.vit.parameters():
            param.requires_grad = False
            
        
     
        if isinstance(self.vit.heads, nn.Sequential):
            
            for module in reversed(list(self.vit.heads.children())):
                if isinstance(module, nn.Linear):
                    in_features = module.in_features
                    break
            else:
           
                in_features = 768 
                
       
            self.vit.heads = nn.Linear(in_features, num_classes)
        else:
           
            in_features = self.vit.heads.in_features
            self.vit.heads = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        """Forward pass through the ViT model"""
        return self.vit(x)
    
    def load_from_checkpoint(self, checkpoint_path):
        """
        Load model weights from checkpoint
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Print checkpoint structure for debugging
            print(f"Checkpoint type: {type(checkpoint)}")
            if isinstance(checkpoint, dict):
                print(f"Checkpoint keys: {checkpoint.keys()}")
                # Check if any keys contain 'heads'
                heads_keys = [k for k in checkpoint.keys() if 'heads' in k]
                print(f"Heads keys: {heads_keys}")
                
            # Create a new state dict that can handle both key formats
            if isinstance(checkpoint, dict) and not any(k.startswith('vit.') for k in checkpoint.keys()):
               
                new_state_dict = {}
                for k, v in checkpoint.items():
           
                    if k == 'heads.weight' or k == 'vit.heads.weight':
                        new_state_dict['vit.heads.weight'] = v
                    elif k == 'heads.bias' or k == 'vit.heads.bias':
                        new_state_dict['vit.heads.bias'] = v
                    else:
                       
                        if not k.startswith('vit.'):
                            new_state_dict[f'vit.{k}'] = v
                        else:
                            new_state_dict[k] = v
                            
                self.load_state_dict(new_state_dict, strict=False)
                print("Custom state dict loaded successfully")
            else:
             
                if isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        self.load_state_dict(checkpoint['model'], strict=False)
                    elif 'state_dict' in checkpoint:
                        self.load_state_dict(checkpoint['state_dict'], strict=False)
                    else:
                        self.load_state_dict(checkpoint, strict=False)
                else:
                    self.load_state_dict(checkpoint, strict=False)
                
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            raise e
        
        return self