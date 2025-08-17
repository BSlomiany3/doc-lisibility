## imports

import os
import cv2
import shutil
import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F


#### Segmentation functions

def _load_binarize(file_path):
    file = Image.open(file_path).convert("L")
    file_array = np.array(file)

    file_binary = np.zeros_like(file_array, dtype=np.uint8)
    file_binary[file_array > 100] = 1
    file_binary[file_binary != 1] = 0

    return file_binary

def _characters_extraction(doc_binary, threshold=0.999):
    line_breaks = []
    block = 1
    writings_flag = False

    for i, row in enumerate(doc_binary):
        white_ratio = row.mean()
        if white_ratio > threshold:
            line_breaks.append(0)
            if writings_flag:
                block += 1
                writings_flag = False
        else:
            writings_flag = True
            line_breaks.append(block)

    new_rows = [doc_binary[i, :] for i, seg in enumerate(line_breaks) if seg != 0]
    writings = np.array(new_rows)

    character_list = []
    unique_lines = sorted(set(line_breaks) - {0})
    for line_id in unique_lines:
        line_indices = [i for i, seg in enumerate(line_breaks) if seg == line_id]
        sub_writing = doc_binary[line_indices, :]

        in_letter = False
        start_col = 0
        for col in range(sub_writing.shape[1]):
            if sub_writing[:, col].min() == 0 and not in_letter:
                in_letter = True
                start_col = col
            elif sub_writing[:, col].min() != 0 and in_letter:
                letter_img = sub_writing[:, start_col:col]
                character_list.append((letter_img != 0).astype(np.uint8))
                in_letter = False

        if in_letter:
            letter_img = sub_writing[:, start_col:]
            character_list.append((letter_img != 0).astype(np.uint8))

    return line_breaks, writings, character_list

def _padding(chars_list):
    padded_chars = []
    for char_roi in chars_list:
        h, w = char_roi.shape
        if h > 28 or w > 28:
            scale = 28 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            new_h = max(1, new_h)
            new_w = max(1, new_w)
            char_roi = cv2.resize(char_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w

        padded = np.ones((28, 28), dtype=np.uint8)
        pad_x = (28 - w) // 2
        pad_y = (28 - h) // 2
        padded[pad_y:pad_y + h, pad_x:pad_x + w] = char_roi

        prep_nn_char = (padded == 0).astype(np.uint8)
        padded_chars.append(prep_nn_char)

    return padded_chars


#### Best model class

class VGGLikeNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, log_activations=False):
        activations = {}

        x = self.block1(x)
        if log_activations:
            activations['block1_out'] = x

        x = self.block2(x)
        if log_activations:
            activations['block2_out'] = x

        logits = self.classifier(x)
        if log_activations:
            activations['logits'] = logits
            return logits, activations
        
        return logits
    

#### Document scoring

def document_confidence(model, list_chars):
    model.eval()
    prob_sum = 0
    
    for i in range(len(list_chars)):
        with torch.no_grad():
            input_tensor = torch.tensor(list_chars[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            logits = model(input_tensor)

            if isinstance(logits, tuple):
                logits = logits[0]

            probs = F.softmax(logits, dim=1)
            prob, _ = probs.max(dim=1)
            prob_sum += prob

    return prob_sum / len(list_chars)


def pipeline(folder, file, best_model, output_folder):
    file_path = os.path.join(folder, file)
    bin_img = _load_binarize(file_path)
    _, _, chars = _characters_extraction(bin_img)
    list_chars = _padding(chars)
    score = float(document_confidence(best_model, list_chars).numpy())

    base_name, ext = os.path.splitext(file)
    new_file_name = f"score__{score:.2f}__{base_name}{ext}"
            
    destination = output_folder
    shutil.move(file_path, os.path.join(destination, new_file_name))