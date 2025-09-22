import torch
import torch.nn.functional as F


def document_confidence(model: torch.nn.Module, list_chars: list[torch.Tensor]) -> torch.Tensor:
    """
    Calcule la confiance moyenne du modèle sur une liste de caractères.

    Chaque caractère est supposé être une image binaire/float (H, W).
    La fonction applique un softmax sur les logits et récupère la probabilité
    maximale par caractère, puis retourne la moyenne sur tous les caractères.

    Paramètres
    ----------
    model : torch.nn.Module
        Modèle de classification prenant en entrée (N, 1, H, W).
    list_chars : list[torch.Tensor] | list[np.ndarray]
        Liste de caractères sous forme de tenseurs ou arrays 2D (H, W).

    Retour
    ------
    torch.Tensor
        Confiance moyenne du modèle (scalaire tensor).
    """
    model.eval()
    confidences = []

    with torch.no_grad():
        for char in list_chars:
            if not isinstance(char, torch.Tensor):
                char = torch.tensor(char, dtype=torch.float32)
            else:
                char = char.to(dtype=torch.float32)

            input_tensor = char.unsqueeze(0).unsqueeze(0)

            logits = model(input_tensor)
            if isinstance(logits, tuple):
                logits = logits[0]

            probs = F.softmax(logits, dim=1)
            prob, _ = probs.max(dim=1)
            confidences.append(prob.item())

    return torch.tensor(confidences).mean()
