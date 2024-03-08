# 2S-ODIS: Two-Stage Omni-Directional Image Synthesis by Geometric Distortion Correction

![proposed_method.jpg](assets%2Fproposed_method.jpg)

## Requirements

```bash
conda create -n 2S-ODIS python=3.10
conda activate 2S-ODIS
pip install -r requirements.txt
```

Download other required files (VQGAN):
```bash
sh setup_vqgan.sh
```
Check if `vqvae_models.pth` is generated.

## Dataset Creation for Training

Place the SUN360 outdoor data one level up (in the same directory as the 2S-ODIS folder).

```bash
sh generate_dataset.sh
```
Leave it for about a day, and the dataset will be created automatically.

## Training Low/High-Resolution Models

```bash
python train_first_stage.py
python train_second_stage.py --bf16
```
Train each for two days to save the trained models on mlflow. 
Please enable `bf16` for the second stage.

Ensure to note down the path where weights are saved on mlflow for inference.
To view experiment logs in mlflow, navigate to the `2S-ODIS` folder and run:
```bash
mlflow ui
```

## Inference

```bash
python image_generation.py --lowres_base_path="path_to_lowres" --highres_base_path="path_to_highres"
```
Example:
```bash
python image_generation.py --lowres_base_path=mlruns/1/0910882ab5ad4faeadbbd2288b176980/artifacts/ --highres_base_path=mlruns/2/7e71d11321ce4f0da773f8723861c077/artifacts/
```
Generating 5000 images takes approximately 2 hours in RTX3090. 
Each image is created in about 1.5s.

To evaluate, navigate to the `eval` folder and run:
```bash
python evaluation.py
```
Evaluation results will be computed in `eval/results`.

## Development environment
- NVIDIA RTX 3090
- CUDA 11.8

## License
Apache License 2.0

## BibTeX
