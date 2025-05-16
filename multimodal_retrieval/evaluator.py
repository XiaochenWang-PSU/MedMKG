import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
#def evaluate(model, test_loader, device):
#    model.eval()
#    image_embeddings = []
#    text_embeddings = []
#    text_to_image_map = []  # This will store the mapping from text indices to image indices
#
#    for batch in tqdm(test_loader):
#        images = batch['image'].to(device)
#        text_tokens = batch['text_tokens'].to(device)
#        image_indices_for_texts = batch['image_indices_for_texts']
#
#        with torch.no_grad():
#            img_embed, txt_embed, _, _ = model({'image': images, 'text_tokens': text_tokens})
#        
#        image_embeddings.extend(img_embed.cpu().numpy())
#        text_embeddings.extend(txt_embed.cpu().numpy())
#        text_to_image_map.extend(image_indices_for_texts)
#
#    image_embeddings = np.array(image_embeddings)
#    text_embeddings = np.array(text_embeddings)
#    image_embeddings /= np.linalg.norm(image_embeddings, axis=1, keepdims=True)
#    text_embeddings /= np.linalg.norm(text_embeddings, axis=1, keepdims=True)
#
#    similarity_all = np.dot(text_embeddings, image_embeddings.T)
#    precision_at_k = []
#    recall_at_k = []
#
#    for k in [1, 5, 10, 20, 50, 100]:
#        precisions = []
#        recalls = []
#        for text_idx, image_indices in enumerate(text_to_image_map):
#            similarity = similarity_all[text_idx]
#            topk_image_ids = np.argsort(-similarity)[:k]
#            target_image_ids = set(image_indices)
#
#            precision = len(set(topk_image_ids) & target_image_ids) / k
#            recall = len(set(topk_image_ids) & target_image_ids) / len(target_image_ids)
#            precisions.append(precision)
#            recalls.append(recall)
#
#        precision_at_k.append(np.mean(precisions))
#        recall_at_k.append(np.mean(recalls))
#
#    print(f"Precision at k: {precision_at_k}")
#    print(f"Recall at k: {recall_at_k}")
#
#    results = {
#        "precision_at_k": precision_at_k,
#        "recall_at_k": recall_at_k
#    }
#    return results

#
#def evaluate(model, test_loader, device):
#    model.eval()
#    image_embeddings = []
#    text_embeddings = []
#    text_to_image_map = {}  # Map text index to original text for grouping
#
#    with torch.no_grad():
#        for batch in tqdm(test_loader):
#            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
#                for k, v in batch.items()}
##            images = batch['image'].to(device).squeeze(1)
##            text_tokens = batch['text_tokens'].to(device).squeeze(1)
#            texts = batch['text']
#            result = model(batch)
#            if type(result) == tuple:
#                similarity_itc = result[0]
#            else:
#                similarity_itc = result
#            # similarity_itc, _ = model(batch)
#            if type(batch['text_tokens']) == dict:
#                text_tokens = batch['text_tokens']
#                texts = batch['text']
#                img_emb = model.encode_image(batch['image'].squeeze(1))
#                txt_emb = model.encode_text(text_tokens)
#            else:
#            
#                text_tokens = batch['text_tokens'].squeeze(1)
#                texts = batch['text']
#                img_emb = model.encode_image(batch['image'].squeeze(1))
#                txt_emb = model.encode_text(text_tokens).squeeze(1)
#            # img_emb = model.encode_image(batch['image'].squeeze(1))
#            # txt_emb = model.encode_text(batch['text_tokens'].squeeze(1))
#           
#           # Normalize embeddings
#            img_emb = F.normalize(img_emb, dim=-1)
#            txt_emb = F.normalize(txt_emb, dim=-1)
#           
#            image_embeddings.extend(img_emb.cpu().numpy())
#            text_embeddings.extend(txt_emb.cpu().numpy())
#            
#            # Group same texts together
#            for i, text in enumerate(texts):
#                if text not in text_to_image_map:
#                    text_to_image_map[text] = []
#                text_to_image_map[text].append(len(image_embeddings) - 1)
#
#    # Calculate metrics
#    image_embeddings = np.array(image_embeddings)
#    text_embeddings = np.array(text_embeddings)
#    similarity_all = np.dot(text_embeddings, image_embeddings.T)
#    
#    metrics = {"precision": [], "recall": []}
#    for k in [1, 5, 10, 20, 50, 100]:
#        precisions = []
#        recalls = []
#        
#        for text, image_indices in text_to_image_map.items():
#            similarity = similarity_all[image_indices[0]]  # Use first occurrence
#            topk_indices = np.argsort(-similarity)[:k]
#            relevant = set(image_indices)
#            
#            precision = len(set(topk_indices) & relevant) / k
#            recall = len(set(topk_indices) & relevant) / len(relevant)
#            
#            precisions.append(precision)
#            recalls.append(recall)
#            
#        metrics["precision"].append(np.mean(precisions))
#        metrics["recall"].append(np.mean(recalls))
#    
#    return metrics




def evaluate(model, test_loader, device):
    model.eval()
    image_embeddings = []
    text_embeddings = []
    text_to_image_map = {}  # Map text to associated image indices
    text_to_index = {}      # Map text to the corresponding text embedding index

    current_text_idx = 0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Handle different batch structures
            if isinstance(batch['text_tokens'], dict):
                text_tokens = batch['text_tokens']
            else:
                text_tokens = batch['text_tokens'].squeeze(1)

            texts = batch['text']
            images = batch['image'].squeeze(1)

            img_emb = model.encode_image(images)
            txt_emb = model.encode_text(text_tokens)

            img_emb = F.normalize(img_emb, dim=-1)
            txt_emb = F.normalize(txt_emb, dim=-1)

            image_embeddings.extend(img_emb.cpu().numpy())
            text_embeddings.extend(txt_emb.cpu().numpy())

            for i, text in enumerate(texts):
                if text not in text_to_image_map:
                    text_to_image_map[text] = []
                    text_to_index[text] = current_text_idx
                text_to_image_map[text].append(len(image_embeddings) - len(texts) + i)
                current_text_idx += 1

    # Stack embeddings
    image_embeddings = np.array(image_embeddings)
    text_embeddings = np.array(text_embeddings)

    # Compute full similarity matrix
    similarity_all = np.dot(text_embeddings, image_embeddings.T)

    # Initialize metrics
    metrics = {"precision": [], "recall": []}

    for k in [1, 5, 10, 20, 50, 100]:
        precisions = []
        recalls = []

        for text, image_indices in text_to_image_map.items():
            text_idx = text_to_index[text]
            similarity = similarity_all[text_idx]

            topk_indices = np.argsort(-similarity)[:k]
            relevant = set(image_indices)

            precision = len(set(topk_indices) & relevant) / k
            recall = len(set(topk_indices) & relevant) / len(relevant)

            precisions.append(precision)
            recalls.append(recall)

        metrics["precision@{}".format(k)] = np.mean(precisions)
        metrics["recall@{}".format(k)] = np.mean(recalls)

    return metrics