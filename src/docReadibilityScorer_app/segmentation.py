# imports
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from docReadibilityScorer_app.utils import (_DEFAULT_BINARIZE_THRESHOLD, 
                  _DEFAULT_LINE_WHITE_RATIO, 
                  _CHAR_SIZE)


def _load_binarize(file_path: Union[str, Path], threshold: int = _DEFAULT_BINARIZE_THRESHOLD) -> np.ndarray:
    """
    Load image, convert to grayscale and binarize into a uint8 array with values {0, 1}.

    Parameters
    ----------
    file_path:
        Path or string pointing to the image file.
    threshold:
        Grayscale threshold (0-255). Pixels with value > threshold become 1, otherwise 0.

    Returns
    -------
    np.ndarray
        2D uint8 array where background pixels are 1 and ink/foreground pixels are 0.
    """
    path = Path(file_path)
    with Image.open(path) as img:
        gray = img.convert("L")
        arr = np.asarray(gray)

    file_binary = (arr > threshold).astype(np.uint8)
    return file_binary


def _characters_extraction(doc_binary: np.ndarray, threshold: float = _DEFAULT_LINE_WHITE_RATIO) -> Tuple[List[int], np.ndarray, List[np.ndarray]]:
    """
    Extract lines and character ROIs from a binary document image.

    Parameters
    ----------
    doc_binary:
        2D numpy array (uint8) where background pixels are 1 and ink pixels are 0.
    threshold:
        Ratio of white (background) in a row above which the row is considered a line break.

    Returns
    -------
    line_breaks:
        List of integers marking segmentation blocks (0 for blank rows, >0 for block id).
    writings:
        2D numpy array consisting of the rows that are not blank (i.e. belong to some block).
    character_list:
        List of 2D numpy uint8 arrays for each detected character (foreground is 1 in returned arrays).
    """
    line_breaks: List[int] = []
    block = 1
    writings_flag = False

    for i, row in enumerate(doc_binary):
        white_ratio = float(row.mean())
        if white_ratio > threshold:
            line_breaks.append(0)
            if writings_flag:
                block += 1
                writings_flag = False
        else:
            writings_flag = True
            line_breaks.append(block)

    new_rows = [doc_binary[i, :] for i, seg in enumerate(line_breaks) if seg != 0]
    writings = np.array(new_rows) if new_rows else np.zeros((0, doc_binary.shape[1]), dtype=doc_binary.dtype)

    character_list: List[np.ndarray] = []
    unique_lines = sorted(set(line_breaks) - {0})
    for line_id in unique_lines:
        
        line_indices = [i for i, seg in enumerate(line_breaks) if seg == line_id]
        sub_writing = doc_binary[line_indices, :]

        in_letter = False
        start_col = 0
        for col in range(sub_writing.shape[1]):
            
            col_has_ink = (sub_writing[:, col].min() == 0)
            if col_has_ink and not in_letter:
                in_letter = True
                start_col = col
            elif (not col_has_ink) and in_letter:
                
                letter_img = sub_writing[:, start_col:col]
                
                character_list.append((letter_img != 0).astype(np.uint8))
                in_letter = False

        if in_letter:
            letter_img = sub_writing[:, start_col:]
            character_list.append((letter_img != 0).astype(np.uint8))

    return line_breaks, writings, character_list


def _padding(chars_list: List[np.ndarray], size: int = _CHAR_SIZE) -> List[np.ndarray]:
    """
    Pad / resize extracted character ROIs into fixed-size images suitable for a neural network.

    Parameters
    ----------
    chars_list:
        List of 2D numpy arrays with foreground==1, background==0.
    size:
        Target square size (default 28).

    Returns
    -------
    List[np.ndarray]
        List of uint8 arrays of shape (size, size) where foreground is 1 and background is 0,
        matching the original function's final format (prep_nn_char).
    """
    padded_chars: List[np.ndarray] = []
    for char_roi in chars_list:
        h, w = char_roi.shape

        if h > size or w > size:
            scale = size / max(h, w)
            new_h = max(1, int(h * scale))
            new_w = max(1, int(w * scale))
            char_roi = cv2.resize(char_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w

        padded = np.ones((size, size), dtype=np.uint8)
        pad_x = (size - w) // 2
        pad_y = (size - h) // 2
        padded[pad_y:pad_y + h, pad_x:pad_x + w] = char_roi

        prep_nn_char = (padded == 0).astype(np.uint8)
        padded_chars.append(prep_nn_char)

    return padded_chars
