import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).parents[1]))
import argparse
import json
import deepspeed
import wandb
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import EvalPrediction, AutoTokenizer, BitsAndBytesConfig

# Import PEFT modules for parameter-efficient fine-tuning
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

# Import the specific sliced model and its loss function
from models.SlicedLlama3_2 import SlicedLlama3_2, compute_loss
from utils.preprocessing import tokenize, train_val_test_split, subset_dataset
from utils.trainer import compute_metrics # For validation metrics

NUM_EPOCH = 3
MODEL_NAME = "meta-llama/Llama-3.2-1B" # Specify base model

def main():
    parser = argparse.ArgumentParser(description='Sliced Llama-3.2 model experiment')
    parser.add_argument("--dataset", choices=['imdb', 'yelp'], default='imdb', help="Dataset to use")
    parser.add_argument("--subset_yelp", action='store_true', help='Whether to subset the yelp dataset')
    parser.add_argument("--subset_size", type=int, default=25000, help="Size for subsetting train/val/test")
    parser.add_argument("--ds_config", type=str, default="ds_config_llama3_2.json", help="DeepSpeed config file")
    parser.add_argument("--val_interval", type=int, default=500, help="Steps between validation checks")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Base model identifier")
    
    # Add quantization and LoRA parameters
    parser.add_argument("--quantize", choices=['none', '8bit', '4bit'], default='none', 
                        help="Whether to quantize the model to 8-bit or 4-bit precision")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout value")

    # for deepspeed
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    args = parser.parse_args()

    print(args)

    # --- Set up Dataset ---
    input_col_name = "text"
    if args.dataset == 'imdb':
        dataset = load_dataset("imdb")
        num_labels = 2
    elif args.dataset == 'yelp':
        dataset = load_dataset("yelp_review_full")
        num_labels = 5
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported.")

    # --- Load Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad token to EOS token: {tokenizer.pad_token}")
    # Llama uses last token, ensure left padding for consistency if batching
    tokenizer.padding_side = 'left'
    print(f"Set padding side to left")

    # --- Configure quantization if requested ---
    if args.quantize != 'none':
        print(f"Applying {args.quantize} quantization for Llama model")
        if args.quantize == '8bit':
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
        elif args.quantize == '4bit':
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,  # Use float16 for better compatibility
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
    else:
        quantization_config = None

    # --- Set up Sliced Model ---
    print(f"Initializing SlicedLlama3_2 with base model: {args.model_name}")
    model = SlicedLlama3_2(model_name=args.model_name, num_labels=num_labels)
    
    # Apply LoRA if using quantization
    use_lora = args.quantize != 'none'
    if use_lora:
        print(f"Applying LoRA with r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
        
        # Prepare model for k-bit training if quantized
        if quantization_config:
            model.llama = prepare_model_for_kbit_training(model.llama)
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # We're using it for feature extraction
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            inference_mode=False,
        )
        
        # Apply LoRA to llama model
        model.llama = get_peft_model(model.llama, peft_config)
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable_params} ({100 * trainable_params / all_params:.2f}% of {all_params})")

    # --- DeepSpeed Initialization ---
    print(f"Loading DeepSpeed config from: {args.ds_config}")
    with open(args.ds_config, "r") as f:
        df_config = json.load(f)

    # Filter model parameters to train only classification layers and LoRA parameters
    trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=trainable_params, # Pass only trainable params
        config=df_config
    )
    print(f"DeepSpeed initialized on rank {model_engine.local_rank}")


    # --- Prepare Datasets ---
    tokenized_datasets = tokenize(dataset, tokenizer, input_col_name=input_col_name)
    train_dataset, val_dataset, test_dataset = train_val_test_split(tokenized_datasets)

    if args.dataset == 'yelp' and args.subset_yelp:
        print(f"Subsetting Yelp dataset to size: {args.subset_size}")
        train_dataset = subset_dataset(train_dataset, size=args.subset_size, seed=42)
        val_dataset = subset_dataset(val_dataset, size=args.subset_size, seed=42)
        test_dataset = subset_dataset(test_dataset, size=args.subset_size, seed=42)

    # --- Set up Dataloaders ---
    batch_size = df_config.get("train_micro_batch_size_per_gpu", df_config["train_batch_size"])
    print(f"Using batch size: {batch_size}")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # --- W&B Tracking ---
    run_name = f"Sliced-{args.model_name.split('/')[-1]}-{args.dataset}"
    if args.subset_yelp:
         run_name += f"_subset{args.subset_size}"
    if args.quantize != 'none':
        run_name += f"-{args.quantize}-LoRA"

    if model_engine.global_rank == 0:
        wandb.init(
            project="text-sentiment-analysis-sliced",
            entity="sc4001", # Replace with your entity
            name=run_name,
            config=df_config
        )
        wandb.config.update(vars(args))
        wandb.config.update({"num_layers": model.num_layers, "hidden_dim": model.hidden_dimension})

    # --- Training Loop ---
    global_step = 0
    for epoch in range(NUM_EPOCH):
        if model_engine.global_rank == 0:
            print(f"Epoch {epoch + 1}/{NUM_EPOCH}")

        model_engine.train()
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", disable=(model_engine.global_rank != 0))
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(model_engine.local_rank)
            attention_mask = batch['attention_mask'].to(model_engine.local_rank)
            labels = batch['label'].to(model_engine.local_rank)

            # Forward pass
            all_output_logits = model_engine(input_ids=input_ids,
                                             attention_mask=attention_mask)

            # Compute loss
            mean_loss, all_layer_loss = compute_loss(
                all_layer_logits=all_output_logits,
                labels=labels,
                num_labels=model.num_labels,
                num_layers=model.num_layers
            )

            # Backward propagation
            model_engine.backward(mean_loss)

            # Weight update
            model_engine.step()

            # --- Logging ---
            wandb_log = {"train/mean_loss": mean_loss.item()}
            for i in range(model.num_layers):
                wandb_log[f"train_loss/layer_{i+1}"] = all_layer_loss[i].item()

            pbar.set_postfix({"Loss": mean_loss.item()})

            ########### Validation ###########
            if global_step > 0 and global_step % args.val_interval == 0:
                model_engine.eval()
                all_labels_val = []
                all_layer_logits_val = []

                with torch.no_grad():
                    for val_batch in val_dataloader:
                        input_ids_val = val_batch['input_ids'].to(model_engine.local_rank)
                        attention_mask_val = val_batch['attention_mask'].to(model_engine.local_rank)
                        labels_val = val_batch['label']

                        logits_val = model_engine(input_ids=input_ids_val,
                                                  attention_mask=attention_mask_val)

                        all_labels_val.append(labels_val)
                        all_layer_logits_val.append(logits_val.cpu())

                all_labels_tensor = torch.cat(all_labels_val, dim=0)
                all_layer_logits_tensor = torch.cat(all_layer_logits_val, dim=1)

                _, all_layer_val_loss = compute_loss(
                    all_layer_logits=all_layer_logits_tensor,
                    labels=all_labels_tensor,
                    num_labels=model.num_labels,
                    num_layers=model.num_layers
                )

                for i in range(model.num_layers):
                    wandb_log[f"eval_loss/layer_{i+1}_loss"] = all_layer_val_loss[i].item()

                for i in range(model.num_layers):
                    pred = EvalPrediction(predictions=all_layer_logits_tensor[i].numpy(),
                                          label_ids=all_labels_tensor.long().numpy())
                    metrics = compute_metrics(pred=pred)
                    for key, value in metrics.items():
                        wandb_log[f"eval_{key}/layer_{i+1}"] = value

                model_engine.train()

            # Log to wandb
            if model_engine.global_rank == 0:
                 wandb.log(wandb_log, step=global_step)

            global_step += 1

    # --- End W&B Tracking ---
    if model_engine.global_rank == 0:
        wandb.finish()
    print("Training finished.")


if __name__ == "__main__":
    main()