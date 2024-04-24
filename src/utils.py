from scipy.spatial.distance import cosine
from transformers import AutoProcessor, ClapModel
import torchaudio
from open_clip.transformer import text_global_pool
import os

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageOps, ImageDraw
from scipy.spatial.distance import cdist, cosine
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import open_clip

from transformers import (ViltProcessor, ViltForImageAndTextRetrieval,
                          BridgeTowerProcessor,
                          BridgeTowerForImageAndTextRetrieval,
                          GPT2Model, GPT2Tokenizer, AutoTokenizer, AutoModel)

"""
Setup Models
"""
model_config = {
    "ViT-B-32": {"pretrained": "openai", "model_name": "ViT-B-32"},
    "ViT-L-14-336": {"pretrained": "openai", "model_name": "ViT-L-14-336"},
    "ViT-H-14": {"pretrained": "laion2b_s32b_b79k", "model_name": "ViT-H-14"},
    "ViT-H-14-no-proj": {"pretrained": "laion2b_s32b_b79k", "no_proj": True, "model_name": "ViT-H-14"},
    "ViT-g-14": {"pretrained": "laion2b_s34b_b88k", "model_name": "ViT-g-14"},
    "ViT-bigG-14": {"pretrained": "laion2b_s39b_b160k", "model_name": "ViT-big-G-14"},
    "ViT-L-14": {"pretrained": "openai", "model_name": "ViT-L-14"},
    "imagebind": {"pretrained": True, "custom_model": True, "model_name": "imagebind"},
    "vilt": {"pretrained": True, "custom_model": True, "model_name": "vilt"},
    "bridgetower": {"pretrained": True, "custom_model": True, "model_name": "bridgetower"},
    "gpt2": {"pretrained": True, "model_name": "gpt2"},
    "gpt2-medium": {"pretrained": True, "model_name": "gpt2-medium"},
    "gpt2-large": {"pretrained": True, "model_name": "gpt2-large"},
    "gpt2-xl": {"pretrained": True, "model_name": "gpt2-xl"},
    "clap": {"pretrained": True, "model_name": "laion/clap-htsat-unfused"},
    "bert-base-uncased": {"pretrained": True, "model_name": "bert-base-uncased"},
    "bert-large-uncased": {"pretrained": True, "model_name": "bert-large-uncased"}
}


class BaseModelHandler:
    def __init__(self, model_key):
        self.config = model_config.get(model_key)

        if not self.config:
            raise ValueError("Model not implemented")

        self.model_name = self.config["model_name"]
        self.model, self.preprocess, self.tokenizer, self.device = self.setup_model()

    def setup_model(self):
        raise NotImplementedError
    
    def compare_pair(self, pair_1, pair_2, modalities, metric="softmax"):
        raise NotImplementedError

    def encode_media(self, data, modality='vision'):
        # Unified method to handle all modalities including text
        pass

class OpenClipHandler(BaseModelHandler):
    def __init__(self, model_name):
        super().__init__(model_name)

    def setup_model(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.config["pretrained"])
        tokenizer = open_clip.get_tokenizer(self.model_name) if isinstance(
            model, open_clip.model.CLIP) else None
        
        if self.config.get("no_proj"):
            model.text_projection = None

        model.eval()
        model.to(device)
        return model, preprocess, tokenizer, device

    def encode_text(self, text_list):
        text_inputs = self.tokenizer(text_list)
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
        return text_features

    def encode_image(self, image_paths):
        image_inputs = [self.preprocess(Image.open(path)).unsqueeze(
            0).to(self.device) for path in image_paths]
        with torch.no_grad():
            image_features = torch.stack(
                [self.model.encode_image(img) for img in image_inputs]
                ).squeeze()
        return image_features

    def encode_media(self, data, modality='vision'):
        if modality == 'vision':
            return self.encode_image(data)
        elif modality == 'text':
            return self.encode_text(data)
        else:
            raise ValueError("Modality must be either 'vision' or 'text'")
        
    def compare_pair(self, pair_1, pair_2, modalities, metric="softmax"):

        pair_1_features = self.encode_media(
            pair_1, modalities[0])
        pair_2_features = self.encode_media(
            pair_2, modalities[1])
        
        if metric == "cosine":
            return Util.calculate_cosine_similarity(
                pair_1_features, pair_2_features)
        elif metric == "softmax":
            return Util.calculate_softmax_probability(
                pair_1_features, pair_2_features)
        elif metric == "logits":
            return pair_1_features @ pair_2_features.T
        else:
            raise ValueError("Metric must be either 'cosine' or 'softmax'")
        
class GPT2Handler(BaseModelHandler):
    def __init__(self, model_name):
        super().__init__(model_name)

    def setup_model(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = GPT2Model.from_pretrained(self.model_name, output_hidden_states=True)
        tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        model.to(device)
        return model, None, tokenizer, device

    def encode_text(self, text_list):
        inputs = self.tokenizer(
            text_list, return_tensors='pt',
            padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.hidden_states
        layer = hidden_states[-1]
        # print(layer.shape)
        layer_embeddings = layer.mean(dim=1).squeeze().numpy()
        # print(layer_embeddings.shape)
        return layer_embeddings

    def encode_image(self, image_paths):
        raise NotImplementedError

    def encode_media(self, data, modality='text'):
        if modality == 'text':
            return self.encode_text(data)
        else:
            raise ValueError("Modality must be 'text' for GPT2")

    def compare_pair(self, pair_1, pair_2, modalities, metric="softmax"):
        pair_1_features = self.encode_media(
            pair_1, modalities[0])
        pair_2_features = self.encode_media(
            pair_2, modalities[1])
        
        if metric == "cosine":
            return 1 - cdist(pair_1_features, pair_2_features, metric="cosine")
        elif metric == "softmax":
            return Util.calculate_softmax_probability(
                pair_1_features, pair_2_features)
        else:
            raise ValueError("Metric must be either 'cosine' or 'softmax'")


class BERTHandler(BaseModelHandler):
    def __init__(self, model_name):
        super().__init__(model_name)

    def setup_model(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = AutoModel.from_pretrained(
            self.model_name, output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model.eval()
        model.to(device)
        return model, None, tokenizer, device

    def encode_text(self, text_list, scope="cls", add_special_tokens=True):
        inputs = self.tokenizer(
            text_list, return_tensors='pt', padding=True,
            truncation=True, max_length=512, add_special_tokens=add_special_tokens
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.hidden_states

        attention_masks = inputs['attention_mask']
        token_masks = attention_masks.unsqueeze(
            -1).expand_as(hidden_states[0]).clone()

        if scope == "target" and add_special_tokens:
            for i in range(token_masks.size(0)):
                token_masks[i, 0, :] = 0  # Zero out [CLS]
                last_token_index = torch.sum(attention_masks[i]) - 1
                token_masks[i, last_token_index, :] = 0  # Zero out [SEP]
            layer_embeddings = hidden_states[-1] * token_masks
            layer_embeddings = layer_embeddings.mean(dim=1).squeeze()

        if scope == "cls":
            layer_embeddings = hidden_states[-1][:, 0, :]
            layer_embeddings = layer_embeddings.squeeze()

        else:
            layer_embeddings = hidden_states[-1] * token_masks
            layer_embeddings = layer_embeddings.mean(dim=1).squeeze()
        # layer_embeddings = []
        # for layer in hidden_states:
        #     layer = layer * token_masks
        #     layer_embeddings.append(layer.mean(dim=1).squeeze())
        # print(layer_embeddings.shape)

        return layer_embeddings

    def encode_image(self, image_paths):
        raise NotImplementedError

    def encode_media(self, data, modality='text', scope="cls", add_special_tokens=True):
        if modality == 'text':
            return self.encode_text(data, scope, add_special_tokens)
        else:
            raise ValueError("Modality must be 'text' for BERT")

    def compare_pair(self, pair_1, pair_2, modalities, metric="softmax", scope="cls", add_special_tokens=True):
        pair_1_features = self.encode_media(
            pair_1, modalities[0], scope, add_special_tokens)
        pair_2_features = self.encode_media(
            pair_2, modalities[1], scope, add_special_tokens)

        if metric == "cosine":
            return Util.calculate_cosine_similarity(
                pair_1_features, pair_2_features)
        elif metric == "softmax":
            return Util.calculate_softmax_probability(
                pair_1_features, pair_2_features)
        else:
            raise ValueError("Metric must be either 'cosine' or 'softmax'")


class ImageBindHandler(BaseModelHandler):
    def __init__(self, model_name):
        super().__init__(model_name)

    def setup_model(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = imagebind_model.imagebind_huge(
            pretrained=self.config["pretrained"])
        model.eval()
        model.to(device)
        return model, None, None, device

    def encode_text(self, text_list):
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(
                text_list, self.device),
        }
        with torch.no_grad():
            embeddings = self.model(inputs)
        return embeddings[ModalityType.TEXT]

    def encode_image(self, image_paths):
        inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data(
                image_paths, self.device),
        }
        with torch.no_grad():
            embeddings = self.model(inputs)
        return embeddings[ModalityType.VISION]
    
    def encode_audio(self, audio_paths):
        inputs = {
            ModalityType.AUDIO: data.load_and_transform_audio_data(
                audio_paths, self.device),
        }
        with torch.no_grad():
            embeddings = self.model(inputs)
        return embeddings[ModalityType.AUDIO]

    def encode_media(self, data, modality='vision'):
        if modality == 'vision':
            return self.encode_image(data)
        elif modality == 'text':
            return self.encode_text(data)
        elif modality == 'audio':
            return self.encode_audio(data)
        else:
            raise ValueError(
                "Modality must be either 'vision', 'text', or 'audio'"
            )
        
    def compare_pair(self, pair_1, pair_2, modalities, metric="softmax"):
        pair_1_features = self.encode_media(
            pair_1, modalities[0])
        pair_2_features = self.encode_media(
            pair_2, modalities[1])
        
        if metric == "cosine":
            return Util.calculate_cosine_similarity(
                pair_1_features, pair_2_features)
        elif metric == "softmax":
            return Util.calculate_softmax_probability(
                pair_1_features, pair_2_features)
        else:
            raise ValueError("Metric must be either 'cosine' or 'softmax'")


class CLAPHandler(BaseModelHandler):
    def __init__(self, model_name):
        super().__init__(model_name)

    def setup_model(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = ClapModel.from_pretrained(self.model_name)
        processor = AutoProcessor.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model.eval()
        model.to(device)
        return model, processor, tokenizer, device

    def encode_text(self, text_list):
        with torch.no_grad():
            inputs = self.tokenizer(
                text_list, return_tensors="pt", padding=True)
            outputs = self.model.get_text_features(**inputs)
            text_embeddings = outputs
        return text_embeddings

    def encode_audio(self, audio_list):
        audio_samples = [torchaudio.load(
            audio_file)[0].squeeze() for audio_file in audio_list]
        audio_embeddings = []
        with torch.no_grad():
            for audio_sample in audio_samples:
                inputs = self.preprocess(
                    audios=audio_sample, return_tensors="pt", padding=True)
                outputs = self.model.get_audio_features(**inputs)
                audio_embeddings.append(outputs.squeeze())
        audio_embeddings = torch.stack(audio_embeddings)
        return audio_embeddings

    def encode_media(self, data, modality='audio'):
        if modality == 'audio':
            return self.encode_audio(data)
        elif modality == 'text':
            return self.encode_text(data)
        else:
            raise ValueError("Modality must be either 'audio' or 'text'")

    def compare_pair(self, pair_1, pair_2, modalities, metric="softmax"):
        pair_1_features = self.encode_media(
            pair_1, modalities[0])
        pair_2_features = self.encode_media(
            pair_2, modalities[1])
        
        print(pair_1_features.shape, pair_2_features.shape)

        if metric == "cosine":
            return Util.calculate_cosine_similarity(
                pair_1_features, pair_2_features)
        elif metric == "softmax":
            return Util.calculate_softmax_probability(
                pair_1_features, pair_2_features)
        else:
            raise ValueError("Metric must be either 'cosine' or 'softmax'")


class ViltHandler(BaseModelHandler):
    def __init__(self, model_name):
        super().__init__(model_name)

    def setup_model(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = ViltForImageAndTextRetrieval.from_pretrained(
            "dandelin/vilt-b32-finetuned-coco")
        processor = ViltProcessor.from_pretrained(
            "dandelin/vilt-b32-finetuned-coco")
        model.eval()
        model.to(device)
        return model, processor, None, device
    
    def compare_pair(self, pair_1, pair_2, modalities, metric="softmax"):

        assert modalities == ["text", "vision"]
        images = [Image.open(path).convert("RGB") for path in pair_2]

        results = []
        for text in pair_1:
            results.append([])
            for image in images:
                encoding = self.preprocess(image, text, return_tensors="pt")
                outputs = self.model(**encoding)
                results[-1].append(outputs.logits[0, :])

        if metric == "logits":
            pass
        elif metric == "softmax":
            results = torch.softmax(torch.tensor(results), dim=1)
        else:
            raise ValueError("Metric must be either 'logits' or 'softmax'")
        
        return results


class BridgetowerHandler(BaseModelHandler):
    def __init__(self, model_name):
        super().__init__(model_name)

    def setup_model(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        processor = BridgeTowerProcessor.from_pretrained(
            "BridgeTower/bridgetower-large-itm-mlm-itc")
        model = BridgeTowerForImageAndTextRetrieval.from_pretrained(
            "BridgeTower/bridgetower-large-itm-mlm-itc")
        model.eval()
        model.to(device)
        return model, processor, None, device
    
    def compare_pair(self, pair_1, pair_2, modalities, metric="softmax"):

        assert modalities == ["text", "vision"]
        images = [Image.open(path).convert("RGB") for path in pair_2]

        results = []
        for text in pair_1:
            results.append([])
            for image in images:
                encoding = self.preprocess(image, text, return_tensors="pt")
                outputs = self.model(**encoding)
                results[-1].append(outputs.logits[0, 1])

        if metric == "logits":
            # Keep logits as-is
            pass
        elif metric == "softmax":
            results = torch.softmax(torch.tensor(results), dim=1)
        else:
            raise ValueError("Metric must be either 'logits' or 'softmax'")
        
        return results


class Util:
    @staticmethod
    def calculate_cosine_similarity(features1, features2):
        result = 1 - cdist(
            features1.detach().numpy(),
            features2.detach().numpy(),
            metric='cosine'
        )
        result = result.tolist()
        return result
    
    @staticmethod
    def calculate_softmax_probability(features1, features2):
        return torch.softmax(
            features1 @ features2.T, dim=-1).tolist()


class DataHandler:
    def __init__(self, model_name, dataset,
                 columns=["sentence", "media"],
                 modalities=["text", "vision"],
                 metric="softmax"):
        self.model_name = model_name
        self.dataset = dataset
        self.metric = metric
        self.columns = columns
        self.modalities = modalities
        self.model = self.setup_model_handler(model_name)

        self.csv_path = f"data/{dataset}/items.csv"
        self.img_folder = f"data/{dataset}/images"
        
        self.df = self.load_csv_data(self.csv_path)

        # Add full image path if second modality is not text
        if self.modalities[1] != 'text':
            columns = [f"{self.columns[1]}_a", f"{self.columns[1]}_b"]
            for c in columns:
                self.df[c] = self.df[c].apply(
                    lambda x: os.path.join(self.img_folder, x))

    def setup_model_handler(self, model_name):
        # TODO: move to config
        if model_name == "imagebind":
            return ImageBindHandler(model_name)
        elif model_name == "vilt":
            return ViltHandler(model_name)
        elif model_name == "bridgetower":
            return BridgetowerHandler(model_name)
        elif model_name.startswith("gpt2"):
            return GPT2Handler(model_name)
        elif model_name.startswith("clap"):
            return CLAPHandler(model_name)
        elif model_name.startswith("bert"):
            return BERTHandler(model_name)
        else:
            return OpenClipHandler(model_name)

    def load_csv_data(self, csv_path):
        return pd.read_csv(csv_path)

    def analyze_data(self):
        all_results = []

        pair_1_cols = [f"{self.columns[0]}_a", f"{self.columns[0]}_b"]
        pair_2_cols = [f"{self.columns[1]}_a", f"{self.columns[1]}_b"]
        modality_1, modality_2 = self.modalities

        for index, item in tqdm(self.df.iterrows(), total=len(self.df)):

            pair_1_data = item[pair_1_cols].tolist()
            pair_2_data = item[pair_2_cols].tolist()

            # TODO: Shift logic to model handler
            # model.compare_pair(pair_1, pair_2, modalities, method="cosine")

            results = self.model.compare_pair(
                pair_1_data, pair_2_data, self.modalities, metric=self.metric)

            all_results.append({
                'match_a': results[0][0],
                'mismatch_a': results[0][1],
                'match_b': results[1][1],
                'mismatch_b': results[1][0],
                'object': item['object'],
                'item': item['item'],
                'item_type': item['item_type']
            })

        return pd.DataFrame(all_results)

    @staticmethod
    def format_results(df, model_name, dataset):
        melted_df = pd.melt(df, id_vars=['object', 'item', 'item_type'])
        melted_df['sent_condition'] = melted_df['variable'].apply(
            lambda x: x.split('_')[-1])
        melted_df['match'] = melted_df['variable'].apply(
            lambda x: x.split('_')[0])
        melted_df['media_condition'] = melted_df.apply(
            lambda x: x["sent_condition"] if x["match"] == "match" else 
            {"a":"b", "b":"a"}[x["sent_condition"]],
            axis=1
        )
        melted_df = melted_df.rename(
            columns={'value': 'similarity'}).drop(columns=['variable'])
        melted_df = melted_df[
            [
                "item", "object", "item_type", "sent_condition",
                "media_condition",
                "match", "similarity"]
            ]
        melted_df["model"] = model_name
        melted_df["dataset"] = dataset
        return melted_df
    
    @staticmethod
    def results_summary(df):
        summary = df[["match", "similarity"]].groupby(["match"]).mean()
        return summary

    @staticmethod
    def ttest(df):
        from scipy.stats import ttest_ind
        match = df[df["match"] == "match"]["similarity"]
        mismatch = df[df["match"] == "mismatch"]["similarity"]
        t, p = ttest_ind(match, mismatch)
        return t, p

    @staticmethod
    def plot_results(df, save_path=None):
        sns.pointplot(data=df, x="match",
                    y="similarity", hue="sent_condition")

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
