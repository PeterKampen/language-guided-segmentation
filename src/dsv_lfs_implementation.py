import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, LlavaForConditionalGeneration, LlavaProcessor
from segment_anything import sam_model_registry, SamPredictor
from argparse import ArgumentParser
from coco_dataset import COCODataset
from peft import LoraConfig, get_peft_model

class ClassSemanticEncoder(nn.Module):
    """
    Class Semantic Encoder Module that adapts general class descriptions to the query image
    using a multimodal LLM (LLaVA architecture) with LoRA fine-tuning and int4 quantization.
    """

    def __init__(self, llm_model_name="llava-hf/llava-1.5-7b", device="cuda", lora_r=16, lora_alpha=32,
                 lora_dropout=0.1):
        super(ClassSemanticEncoder, self).__init__()
        self.device = device

        # Load quantization configuration for int4
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # normalized float 4
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        # Initialize the LLaVA model with quantization
        self.llava = LlavaForConditionalGeneration.from_pretrained(
            llm_model_name,
            quantization_config=quantization_config,
            device_map="auto",  # Automatically determine device mapping
        )
        self.processor = LlavaProcessor.from_pretrained(llm_model_name)

        # Add <SEMprompt> token to the vocabulary
        self.sem_prompt_token_id = len(self.processor.tokenizer)
        self.llava.resize_token_embeddings(len(self.processor.tokenizer) + 1)

        # Freeze all model weights by default
        for param in self.llava.parameters():
            param.requires_grad = False

        # Apply LoRA to relevant layers


        # Define LoRA configuration
        lora_config = LoraConfig(
            r=lora_r,  # Rank of the update matrices
            lora_alpha=lora_alpha,  # Scaling factor
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Target attention modules
            lora_dropout=lora_dropout,  # Dropout probability
            bias="none",  # Don't train bias terms
            task_type="CAUSAL_LM"  # Type of task
        )

        # Apply LoRA adapters to the model
        self.llava = get_peft_model(self.llava, lora_config)
        print(
            f"LoRA applied to LLaVA model. Total trainable parameters: {sum(p.numel() for p in self.llava.parameters() if p.requires_grad)}")

        # MLP projection layer to project LLM embedding to the decoder expected dimensions
        self.proj = nn.Sequential(
            nn.Linear(self.llava.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, query_image, class_description):
        """
        Args:
            query_image: The query image tensor
            class_description: The text description for the target class (W_C)

        Returns:
            sem_prompt: Semantic prompt feature for the decoder
        """
        # Construct input with <SEMprompt> token
        augmented_description = f"{class_description} <SEMprompt>"

        # Process the input
        inputs = self.processor(
            text=augmented_description,
            images=query_image,
            return_tensors="pt"
        ).to(self.device)

        # Get LLM outputs with LoRA adapters
        outputs = self.llava(**inputs, output_hidden_states=True)

        # Extract the last-layer embedding for the <SEMprompt> token
        last_hidden_state = outputs.hidden_states[-1]

        # Find the position of <SEMprompt> token and extract its embedding
        sem_token_pos = (inputs.input_ids == self.sem_prompt_token_id).nonzero()
        if sem_token_pos.shape[0] > 0:
            h_sem = last_hidden_state[sem_token_pos[0, 0], sem_token_pos[0, 1], :]
        else:
            # Fallback to the last token if <SEMprompt> is not found
            h_sem = last_hidden_state[0, -1, :]

        # Project the embedding to the expected dimension
        sem_prompt = self.proj(h_sem)

        return sem_prompt

    def save_lora_weights(self, path):
        """Save only the trained LoRA weights"""
        self.llava.save_pretrained(path)

    def load_lora_weights(self, path):
        """Load trained LoRA weights"""
        from peft import PeftModel
        self.llava = PeftModel.from_pretrained(self.llava, path)


# Helper function to convert a model to int4 quantization after training
def quantize_model_to_int4(model_path, output_path=None):
    """
    Convert a saved model to int4 quantization.

    Args:
        model_path: Path to the saved model
        output_path: Path to save the quantized model (if None, will use model_path + '_int4')

    Returns:
        Path to the quantized model
    """
    from transformers import BitsAndBytesConfig
    import os

    if output_path is None:
        output_path = model_path + "_int4"

    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Load the model with quantization
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Save the quantized model
    model.save_pretrained(output_path)

    print(f"Model quantized and saved to {output_path}")
    return output_path


class DenseMatchingModule(nn.Module):
    """
    Module to generate visual prompt by finding dense correspondence
    between query and support images.
    """
    def __init__(self, sam_model_type="vit_h", sam_checkpoint="path/to/sam_checkpoint.pth"):
        super(DenseMatchingModule, self).__init__()
        
        # Initialize the SAM model as the feature extractor
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        
        # 4D correlation encoder
        self.encoder_4d = Center4DConvEncoder()
        
        # 4D correlation decoder
        self.decoder_4d = Center4DConvDecoder()
    
    def extract_features(self, images):
        """Extract multilevel features from the SAM encoder"""
        features = []
        with torch.no_grad():
            # Extract features from different transformer layers
            vision_outputs = self.sam.image_encoder(images, output_hidden_states=True)
            for layer_idx in [3, 6, 9, 12]:  # Example layer indices
                features.append(vision_outputs.hidden_states[layer_idx])
        return features
    
    def compute_hypercorrelation(self, query_feats, support_feats):
        """Compute 4D hypercorrelation volume between query and support features"""
        hypercorrs = []
        
        for q_feat, s_feat in zip(query_feats, support_feats):
            # Normalize features
            q_feat = F.normalize(q_feat, dim=1)
            s_feat = F.normalize(s_feat, dim=1)
            
            # Compute correlation
            b, c, hq, wq = q_feat.shape
            _, _, hs, ws = s_feat.shape
            
            q_feat = q_feat.view(b, c, -1).permute(0, 2, 1)  # B x HW x C
            s_feat = s_feat.view(b, c, -1)  # B x C x HW
            
            correlation = torch.bmm(q_feat, s_feat)  # B x HW_q x HW_s
            correlation = correlation.view(b, hq, wq, hs, ws)
            
            # Apply ReLU
            correlation = F.relu(correlation)
            
            hypercorrs.append(correlation)
        
        # Stack along the channel dimension
        hypercorr_volume = torch.cat(hypercorrs, dim=1)
        return hypercorr_volume
    
    def forward(self, query_images, support_images, support_masks):
        """
        Args:
            query_images: Batch of query images
            support_images: Batch of support images
            support_masks: Masks for the support images
            
        Returns:
            vis_prompt: Visual prompt for the decoder
        """
        # Extract features from query and support images
        query_feats = self.extract_features(query_images)
        support_feats = self.extract_features(support_images)
        
        # Apply support masks to focus on target regions in support features
        masked_support_feats = []
        for s_feat, mask in zip(support_feats, support_masks):
            # Resize mask to match feature resolution
            mask = F.interpolate(mask, size=s_feat.shape[-2:], mode='bilinear', align_corners=True)
            masked_feat = s_feat * mask
            masked_support_feats.append(masked_feat)
        
        # Compute 4D hypercorrelation volume
        hypercorr = self.compute_hypercorrelation(query_feats, masked_support_feats)
        
        # Process through 4D encoder
        h_4d = self.encoder_4d(hypercorr)
        
        # Generate visual prompt through 4D decoder
        vis_prompt = self.decoder_4d(h_4d)
        
        return vis_prompt


class Center4DConvEncoder(nn.Module):
    """
    Efficient center-pivot 4D convolution encoder for processing hypercorrelation volumes.
    """
    def __init__(self, in_channels=4, mid_channels=16, out_channels=8):
        super(Center4DConvEncoder, self).__init__()
        
        # Simplified implementation of the center-pivot 4D convolutions
        self.encoder = nn.Sequential(
            Center4DConv(in_channels, mid_channels, kernel_size=3),
            nn.ReLU(),
            Center4DConv(mid_channels, mid_channels, kernel_size=3),
            nn.ReLU(),
            Center4DConv(mid_channels, out_channels, kernel_size=3)
        )
    
    def forward(self, x):
        return self.encoder(x)


class Center4DConv(nn.Module):
    """
    Center-pivot 4D convolution implementation.
    Decomposes 4D convolution into two 2D convolutions for efficiency.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Center4DConv, self).__init__()
        
        # First 2D convolution operates on query dimensions
        self.conv_q = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        
        # Second 2D convolution operates on support dimensions
        self.conv_s = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding
        )
    
    def forward(self, x):
        b, c, hq, wq, hs, ws = x.shape
        
        # Reshape to process query dimensions
        x_q = x.permute(0, 1, 4, 5, 2, 3).reshape(b * hs * ws, c, hq, wq)
        x_q = self.conv_q(x_q)
        _, c_out, hq_out, wq_out = x_q.shape
        
        # Reshape to process support dimensions
        x_q = x_q.reshape(b, hs, ws, c_out, hq_out, wq_out).permute(0, 3, 4, 5, 1, 2)
        x_s = x_q.permute(0, 1, 2, 3, 5, 4).reshape(b * hq_out * wq_out, c_out, hs, ws)
        x_s = self.conv_s(x_s)
        _, _, hs_out, ws_out = x_s.shape
        
        # Reshape back to 4D correlation format
        output = x_s.reshape(b, hq_out, wq_out, c_out, hs_out, ws_out).permute(0, 3, 1, 2, 4, 5)
        
        return output


class Center4DConvDecoder(nn.Module):
    """
    Decoder for 4D correlation features to produce visual prompt.
    """
    def __init__(self, in_channels=8, hidden_dim=64, output_dim=256):
        super(Center4DConvDecoder, self).__init__()
        
        # Process 4D features and collapse to 2D
        self.conv_reduce = nn.Sequential(
            Center4DConv(in_channels, in_channels, kernel_size=3),
            nn.ReLU(),
            Center4DConv(in_channels, 1, kernel_size=1, padding=0)
        )
        
        # Process the 2D features to generate the visual prompt
        self.mlp = nn.Sequential(
            nn.Linear(256, hidden_dim),  # 256 is the flattened feature dimension
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        # Reduce 4D correlation to 2D
        x = self.conv_reduce(x)
        
        # Squeeze the channel dimension and support dimensions
        x = x.squeeze(1)  # Remove channel dimension
        x = x.mean(dim=(-2, -1))  # Average over support dimensions
        
        # Flatten and process through MLP
        b = x.shape[0]
        x = x.reshape(b, -1)
        vis_prompt = self.mlp(x)
        
        return vis_prompt


class PromptBasedDecoder(nn.Module):
    """
    SAM-based decoder guided by semantic and visual prompts.
    """
    def __init__(self, sam_model_type="vit_h", sam_checkpoint="path/to/sam_checkpoint.pth"):
        super(PromptBasedDecoder, self).__init__()
        
        # Initialize SAM model
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.mask_decoder = self.sam.mask_decoder
        
        # Additional layers to fuse semantic and visual prompts
        self.prompt_fusion = nn.MultiheadAttention(
            embed_dim=256,  # Match SAM's embedding dimension
            num_heads=8,
            dropout=0.1
        )
        
        # Projections for the prompts
        self.sem_proj = nn.Linear(256, 256)
        self.vis_proj = nn.Linear(256, 256)
    
    def forward(self, image_features, visual_prompt, semantic_prompt):
        """
        Args:
            image_features: Features from the image encoder
            visual_prompt: Visual prompt from the dense matching module
            semantic_prompt: Semantic prompt from the class semantic encoder
            
        Returns:
            masks: Predicted segmentation masks
        """
        # Process prompts
        vis_prompt = self.vis_proj(visual_prompt).unsqueeze(0)  # Add sequence dimension
        sem_prompt = self.sem_proj(semantic_prompt).unsqueeze(0)  # Add sequence dimension
        
        # Concatenate prompts
        prompts = torch.cat([vis_prompt, sem_prompt], dim=0)
        
        # Fuse prompts through self-attention
        fused_prompt, _ = self.prompt_fusion(prompts, prompts, prompts)
        fused_prompt = fused_prompt.mean(dim=0)  # Average over sequence dimension
        
        # Use the fused prompt to guide the mask decoder
        # Adapt to SAM's mask decoder input format
        sparse_embeddings = fused_prompt.unsqueeze(0)  # Add batch dimension
        dense_embeddings = None  # Not using dense prompts
        
        # Generate masks using SAM's mask decoder
        masks = self.mask_decoder(
            image_embeddings=image_features,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        return masks


class DSVLFS(nn.Module):
    """
    DSV-LFS: Unifying LLM-Driven Semantic Cues with Visual Features for Robust Few-Shot Segmentation
    With LoRA fine-tuning and int4 quantization for the LLM.
    """

    def __init__(
            self,
            sam_model_type="vit_h",
            sam_checkpoint="path/to/sam_checkpoint.pth",
            llm_model_name="llava-hf/llava-1.5-7b",
            device="cuda",
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.1
    ):
        super(DSVLFS, self).__init__()

        # Vision backbone (SAM encoder)
        self.sam_encoder = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint).image_encoder

        # Class semantic encoder with LoRA and quantization
        self.class_semantic_encoder = ClassSemanticEncoder(
            llm_model_name=llm_model_name,
            device=device,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )

        # Dense matching module
        self.dense_matching_module = DenseMatchingModule(sam_model_type=sam_model_type, sam_checkpoint=sam_checkpoint)

        # Prompt-based decoder
        self.prompt_decoder = PromptBasedDecoder(sam_model_type=sam_model_type, sam_checkpoint=sam_checkpoint)

        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

        # Loss weights
        self.lambda_text = 0.5
        self.lambda_mask = 0.5
        self.lambda_bce = 0.5
        self.lambda_dice = 0.5

    def forward(self, query_image, support_images, support_masks, class_description=None):
        """
        Forward pass for the DSV-LFS model.

        Args:
            query_image: The query image to segment
            support_images: Support images containing the target object
            support_masks: Masks for the target object in support images
            class_description: Text description of the target class (W_C)

        Returns:
            mask_pred: Predicted segmentation mask for the query image
        """
        # Extract features from query image
        query_features = self.sam_encoder(query_image)

        # Generate semantic prompt from class description and query image
        semantic_prompt = self.class_semantic_encoder(query_image, class_description)

        # Generate visual prompt from query and support images/masks
        visual_prompt = self.dense_matching_module(query_image, support_images, support_masks)

        # Decode the mask using both prompts
        mask_pred = self.prompt_decoder(query_features, visual_prompt, semantic_prompt)

        return mask_pred

    def save_model(self, path):
        """Save the full model and LoRA weights separately"""
        import os

        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Save LoRA weights separately
        lora_path = os.path.join(path, "lora_weights")
        os.makedirs(lora_path, exist_ok=True)
        self.class_semantic_encoder.save_lora_weights(lora_path)

        # Save rest of the model
        torch.save({
            'sam_encoder': self.sam_encoder.state_dict(),
            'dense_matching': self.dense_matching_module.state_dict(),
            'prompt_decoder': self.prompt_decoder.state_dict(),
            'proj': self.class_semantic_encoder.proj.state_dict()
        }, os.path.join(path, "model_weights.pth"))

        print(f"Model saved to {path}")

    def load_model(self, path, device="cuda"):
        """Load the full model including LoRA weights"""
        import os

        # Load the main model weights
        weights = torch.load(os.path.join(path, "model_weights.pth"), map_location=device)
        self.sam_encoder.load_state_dict(weights['sam_encoder'])
        self.dense_matching_module.load_state_dict(weights['dense_matching'])
        self.prompt_decoder.load_state_dict(weights['prompt_decoder'])
        self.class_semantic_encoder.proj.load_state_dict(weights['proj'])

        # Load LoRA weights
        lora_path = os.path.join(path, "lora_weights")
        if os.path.exists(lora_path):
            self.class_semantic_encoder.load_lora_weights(lora_path)

        print(f"Model loaded from {path}")

class DiceLoss(nn.Module):
    """Dice loss for binary segmentation"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten the predictions and targets
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Calculate Dice coefficient and loss
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_coeff
        
        return dice_loss


# Helper function to generate class descriptions using an LLM

class GenerateClassDescription:
    def __init__(self, path_to_class_description):

        import json
        with open(path_to_class_description) as f:
            self.class_description = json.load(f)

    def __call__(self, class_name):
        return self.class_description[class_name]


def train_dsvlfs(model, data_loader, optimizer, device, num_epochs=10, checkpoint_dir=None, eval_loader=None,
                 log_interval=10):
    """
    Training function for the DSV-LFS model with LoRA fine-tuning.

    Args:
        model: The DSV-LFS model
        data_loader: DataLoader for training data
        optimizer: Optimizer for training
        device: Device to train on
        num_epochs: Number of epochs to train for
        checkpoint_dir: Directory to save checkpoints
        eval_loader: DataLoader for evaluation during training
        log_interval: How often to log progress
    """
    import os
    import time
    from tqdm import tqdm

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    model.train()
    class_description_getter = GenerateClassDescription("path/to/class_description.json")
    best_val_miou = 0.0

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start_time = time.time()

        # Training loop
        pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, (query_img, query_mask, support_imgs, support_masks, class_name) in pbar:
            # Move data to device
            query_img = query_img.to(device)
            query_mask = query_mask.to(device)

            if not isinstance(support_imgs, list):
                support_imgs = [support_imgs.to(device)]
                support_masks = [support_masks.to(device)]
            else:
                support_imgs = [img.to(device) for img in support_imgs]
                support_masks = [mask.to(device) for mask in support_masks]

            # Generate class description
            class_description = class_description_getter(class_name[0])

            # Forward pass
            optimizer.zero_grad()

            # Since our model only outputs the mask_pred now
            mask_pred = model(query_img, support_imgs, support_masks, class_description)

            # Compute loss for the mask only
            bce_loss = model.bce_loss(mask_pred, query_mask)
            dice_loss = model.dice_loss(mask_pred, query_mask)
            loss = model.lambda_bce * bce_loss + model.lambda_dice * dice_loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'bce': f"{bce_loss.item():.4f}",
                'dice': f"{dice_loss.item():.4f}"
            })

        # Report epoch metrics
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s - Avg Loss: {avg_loss:.4f}")

        # Validation
        if eval_loader is not None:
            print("Running validation...")
            model.eval()
            metrics = evaluate_model(model, eval_loader, device, k=1)
            model.train()

            val_miou = metrics['miou']
            print(f"Validation mIoU: {val_miou:.4f}")

            # Update learning rate scheduler
            scheduler.step(val_miou)

            # Save checkpoint if best so far
            if checkpoint_dir and val_miou > best_val_miou:
                best_val_miou = val_miou
                print(f"New best mIoU: {best_val_miou:.4f} - Saving checkpoint")
                model.save_model(os.path.join(checkpoint_dir, f"best_model_epoch{epoch + 1}"))

        # Save regular checkpoint
        if checkpoint_dir and (epoch + 1) % 5 == 0:
            model.save_model(os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch + 1}"))

    print("Training completed!")
    if checkpoint_dir:
        model.save_model(os.path.join(checkpoint_dir, "final_model"))

    return model

def evaluate_model(model, data_loader, device, k=1, threshold=0.5, save_visualizations=False, output_dir=None):
    """
    Evaluate the DSV-LFS model on a validation/test dataset.

    Args:
        model: The trained DSV-LFS model
        data_loader: DataLoader for validation/test data
        device: Device to evaluate on
        k: Number of shots to use for inference
        threshold: Threshold value for binary prediction
        save_visualizations: Whether to save visualization of predictions
        output_dir: Directory to save visualizations if enabled

    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    import numpy as np
    from tqdm import tqdm
    import torch.nn.functional as F
    import os

    if save_visualizations and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        from matplotlib import pyplot as plt

    model.eval()

    # Initialize metric trackers
    metrics = {
        'miou': 0.0,  # Mean IoU
        'dice': 0.0,  # Dice coefficient
        'precision': 0.0,  # Precision
        'recall': 0.0,  # Recall
        'f1': 0.0,  # F1 Score
        'accuracy': 0.0  # Pixel accuracy
    }

    # Set dataloader to return k support samples
    if hasattr(data_loader, 'set_k_shot'):
        data_loader.set_k_shot(k)

    class_description_getter = GenerateClassDescription("path/to/class_description.json")
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (query_img, query_mask, support_imgs, support_masks, class_name) in enumerate(
                tqdm(data_loader, desc=f"Evaluating ({k}-shot)")):
            # Move data to device
            query_img = query_img.to(device)
            query_mask = query_mask.to(device)

            # Make sure support images and masks are in the right format
            if not isinstance(support_imgs, list):
                support_imgs = [support_imgs[i:i + 1] for i in range(support_imgs.shape[0])]
                support_masks = [support_masks[i:i + 1] for i in range(support_masks.shape[0])]

            support_imgs = [img.to(device) for img in support_imgs[:k]]
            support_masks = [mask.to(device) for mask in support_masks[:k]]

            # Generate class description
            class_description = class_description_getter(class_name[0])

            if k == 1:
                # Single-shot prediction
                mask_pred = model(query_img, support_imgs, support_masks, class_description)
                pred_mask_binary = (torch.sigmoid(mask_pred) > threshold).float()
            else:
                # K-shot prediction using the method from the model
                pred_mask_binary = model.k_shot_inference(
                    query_img,
                    support_imgs,
                    support_masks,
                    class_description,
                    k=k,
                    threshold=threshold
                )

            # Move tensors to CPU for metric calculation
            pred_mask_np = pred_mask_binary.cpu().numpy().squeeze()
            gt_mask_np = query_mask.cpu().numpy().squeeze()

            # Calculate metrics
            batch_metrics = calculate_segmentation_metrics(pred_mask_np, gt_mask_np)

            # Update running metrics
            for key in metrics:
                metrics[key] += batch_metrics[key]

            num_samples += query_img.size(0)

            # Save visualizations if requested
            if save_visualizations and output_dir is not None:
                save_prediction_visualization(
                    query_img.cpu().numpy().squeeze(),
                    gt_mask_np,
                    pred_mask_np,
                    f"{output_dir}/sample_{batch_idx}.png"
                )

    # Calculate average metrics
    for key in metrics:
        metrics[key] /= num_samples

    # Print evaluation results
    print(f"\n{k}-Shot Evaluation Results:")
    print(f"Mean IoU: {metrics['miou']:.4f}")
    print(f"Dice Coefficient: {metrics['dice']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Pixel Accuracy: {metrics['accuracy']:.4f}")

    return metrics


def calculate_segmentation_metrics(pred_mask, gt_mask):
    """
    Calculate standard segmentation metrics between prediction and ground truth masks.

    Args:
        pred_mask: Binary prediction mask (numpy array)
        gt_mask: Binary ground truth mask (numpy array)

    Returns:
        metrics: Dictionary of computed metrics
    """
    import numpy as np

    # Ensure binary masks
    pred_mask = (pred_mask > 0.5).astype(np.float32)
    gt_mask = (gt_mask > 0.5).astype(np.float32)

    # Calculate intersection and union
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    # Handle edge case of empty masks
    if union == 0:
        iou = 1.0 if intersection == 0 else 0.0
    else:
        iou = intersection / union

    # Calculate true positives, false positives, and false negatives
    tp = intersection
    fp = pred_mask.sum() - tp
    fn = gt_mask.sum() - tp
    tn = pred_mask.shape[0] * pred_mask.shape[1] - tp - fp - fn

    # Calculate precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Calculate Dice coefficient
    dice = 2 * intersection / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) > 0 else 0.0

    # Calculate pixel accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Compile metrics
    metrics = {
        'miou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }

    return metrics


def save_prediction_visualization(image, gt_mask, pred_mask, save_path):
    """
    Create and save a visualization of the prediction vs ground truth.

    Args:
        image: Original query image
        gt_mask: Ground truth mask
        pred_mask: Predicted mask
        save_path: Path to save the visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Normalize image if needed
    if image.max() > 1.0:
        image = image / 255.0

    # If image has channels first, transpose to channels last for plotting
    if image.shape[0] == 3:  # Channels first (C, H, W)
        image = np.transpose(image, (1, 2, 0))

    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title('Query Image')
    axes[0].axis('off')

    # Plot ground truth mask
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    # Plot predicted mask
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def evaluate_across_k_shots(model, data_loader, device, k_values=[1, 5, 10], save_dir=None):
    """
    Evaluate the model across different k-shot scenarios.

    Args:
        model: The DSV-LFS model
        data_loader: DataLoader for validation/test data
        device: Device to evaluate on
        k_values: List of k values to evaluate
        save_dir: Directory to save results and visualizations

    Returns:
        all_metrics: Dictionary containing evaluation metrics for each k value
    """
    import os
    import json

    all_metrics = {}

    for k in k_values:
        print(f"\nEvaluating with k={k} shots...")

        if save_dir:
            k_save_dir = os.path.join(save_dir, f"k{k}")
            os.makedirs(k_save_dir, exist_ok=True)
        else:
            k_save_dir = None

        metrics = evaluate_model(
            model,
            data_loader,
            device,
            k=k,
            save_visualizations=(save_dir is not None),
            output_dir=k_save_dir
        )

        all_metrics[f'k{k}'] = metrics

    # Save metrics to json if save_dir is provided
    if save_dir:
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(all_metrics, f, indent=4)

    return all_metrics


# Example usage in your main function:
def infer(model, data_loader, device):
    """
    Inference function with evaluation for the DSV-LFS model.

    Args:
        model: The DSV-LFS model
        data_loader: DataLoader for validation/test data
        device: Device to evaluate on
    """
    import os

    # Create output directory
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)

    # Evaluate across multiple k-shot scenarios
    metrics = evaluate_across_k_shots(
        model,
        data_loader,
        device,
        k_values=[1, 3, 5],
        save_dir=output_dir
    )

    # Print summary of results
    print("\nEvaluation Summary:")
    for k, k_metrics in metrics.items():
        print(f"\n{k}-shot results:")
        print(f"Mean IoU: {k_metrics['miou']:.4f}")
        print(f"Dice: {k_metrics['dice']:.4f}")
        print(f"F1: {k_metrics['f1']:.4f}")

    return metrics


def main(args):
    # Initialize the DSV-LFS model
    model = DSVLFS(
        sam_model_type="vit_h",
        sam_checkpoint=args.sc,
        llm_model_name="llava-hf/llava-1.5-7b-hf",
        device="cuda"
    ).to("cuda")

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Create data loaders
    train_loader = COCODataset(args.data_dir, split='train', batch_size=2)
    val_loader = COCODataset(args.data_dir, split='validation', batch_size=2)

    # Train the model
    train_dsvlfs(model, train_loader, optimizer, "cuda", num_epochs=3)

    # Evaluate the model with different k-shot configurations
    metrics = evaluate_across_k_shots(
        model,
        val_loader,
        "cuda",
        k_values=[1, 3, 5],
        save_dir=args.outdir
    )

    # Save model checkpoint
    torch.save(model.state_dict(), os.path.join(args.outdir, "dsv_lfs_model.pth"))

    return metrics


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--sc', type=str, required=True, help='Path to SAM checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory for results and checkpoints')
    args = parser.parse_args()
    main(args)
