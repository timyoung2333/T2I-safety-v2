import os, torch, pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
import argparse


class CocoGenDataset(Dataset):
    def __init__(self, img_dir, csv_path):
        self.img_dir = img_dir
        df = pd.read_csv(csv_path)
        self.data = df.head(1000)
        self.images = []
        self.texts = []
        for _, row in self.data.iterrows():
            fname = f'COCO_val2014_{int(row["image_id"]):012}.png'
            fpath = os.path.join(img_dir, fname)
            if os.path.exists(fpath):
                self.images.append(fpath)
                self.texts.append(row['text'])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {'image': self.images[idx], 'text': self.texts[idx]}


def compute_clip_score(dataloader, device='cuda'):
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')

    total_score = 0
    count = 0
    for batch in tqdm(dataloader, desc='Computing CLIP score'):
        images_feats = processor(images=[Image.open(img) for img in batch['image']], return_tensors="pt").to(device)
        images_out = model.get_image_features(**images_feats)
        images_feats = images_out if isinstance(images_out, torch.Tensor) else images_out.pooler_output

        texts_feats = tokenizer(list(batch['text']), padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        texts_out = model.get_text_features(**texts_feats)
        texts_feats = texts_out if isinstance(texts_out, torch.Tensor) else texts_out.pooler_output

        images_feats = images_feats / images_feats.norm(dim=1, p=2, keepdim=True)
        texts_feats = texts_feats / texts_feats.norm(dim=1, p=2, keepdim=True)
        score = (images_feats * texts_feats).sum(-1)
        total_score += score.sum().item()
        count += len(batch['text'])

    return total_score / count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--csv_path', type=str, default='/scratch/gilbreth/yang1683/projects/vlm_safety/T2I-safety-v2/code/eval/data/mscoco.csv')
    parser.add_argument('--batch_size', type=int, default=10)
    args = parser.parse_args()

    dataset = CocoGenDataset(args.img_dir, args.csv_path)
    print(f'Found {len(dataset)} images')
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    score = compute_clip_score(dataloader)
    print(f'CLIP Score: {score * 100:.2f}')
