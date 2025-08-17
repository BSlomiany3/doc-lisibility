## Expérimentations de classification de caractères

Entraînement sur un petit nombre d'epochs de différentes architectures (petits modèles, de manière à minimiser le temps d’entraînement) afin d’identifier la plus prometteuse.

* Classes du dataset d’entraînement équilibrées → utilisation de l’exactitude globale de classification sur jeu de validation comme critère d’évaluation.
* Entraînement complet d’un modèle plus complexe basé sur l’architecture la plus prometteuse (augmentation jusqu’à overfitting puis techniques de régularisation).
* Évaluation de la performance finale sur ensemble de test.

---

### Résumé des résultats

| Expérience   | Modèle                | Logs TensorBoard                      | ValAcc  |
| ------------ | --------------------- | ------------------------------------- | ------- |
| **Baseline** | `baseline_cnn0.pth`   | `runs/baseline_train0`                | 79.9 %  |
| 1            | `vgglikeNet0.pth`     | `runs/vgglike_train0`                 | 82.05 % |
| 2            | `resnetlikeNet0.pth`  | `runs/resnetlike_train0`              | 78.9 %  |
| 3            | `resneXtlikeNet0.pth` | `runs/resneXtlike_train0`             | 80.5 %  |
| 4            | `InceptionNet0.pth`   | `runs/InceptionNetlike_train0`        | 79.5 %  |
| 5            | `rnn_model0.pth`      | `runs/rnn_squencerTrain0`             | 24.9 %  |
| 6            | `gru_model0.pth`      | `runs/gru_squencerTrain0`             | 80.0 %  |
| 7            | `lstm_model0.pth`     | `runs/lstm_squencerTrain0`            | 81.2 %  |
| 8            | `lstmAttn_model0.pth` | `runs/lstmAttn_squencerTrain0`        | 80.8 %  |
| 9            | `HybridNet0.pth`      | `runs/HybridNetTrain0`                | 80.6 %  |
| 10           | `transformer0.pth`    | `runs/transformer_fromscratch_train0` | 32.9 %  |

> **Conclusion**
> Meilleure invariance à la data augmentation des modèles récurrents, mais, puisque l’on prévoit de centrer les caractères avant passage dans le réseau neuronal, l’architecture **VGGLikeNN** présente la meilleure `ValAcc` et donc le plus d’intérêt.

---

## Détails des expériences

### Expérience « Baseline »

**Fichier modèle :** `baseline_cnn0.pth`
**Logs TensorBoard :** `runs/baseline_train0`

#### Architecture (SimpleBaselineCNN)

```python
class SimpleBaselineCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, num_classes)
        )

    def forward(self, x, log_activations=False):
        x = self.features(x)
        return self.classifier(x)
```

**Training :**

* **Split** : stratified 65 / 15 / 20 (train/val/test)
* **Prétraitement** : Resize 28×28 → ToTensor → binarize (x < 0.7)
* **Augmentation** : flip (p=0.2), rotate ±10°, affine (±10% translation, 0.9–1.1 scale)
* **Optimiseur** : AdamW (lr=1e-2)
* **Loss** : CrossEntropy
* **Epochs** : 5

**Résultats :** ValAcc = 79.9 %

---

### Expérience 1

**Fichier modèle :** `vgglikeNet0.pth`
**Logs TensorBoard :** `runs/vgglike_train0`

#### Architecture (VGGLikeNN)

```python
class VGGLikeNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
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
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_classes)
        )

    def forward(self, x, log_activations=False):
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)
```

**Training :** même protocole que l’expérience Baseline.
**Résultats :** ValAcc = 82.05 %

---

### Expérience 2

**Fichier modèle :** `resnetlikeNet0.pth`
**Logs TensorBoard :** `runs/resnetlike_train0`

#### Architecture (ResNet)

```python
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.link = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1)
        )

    def forward(self, x):
        Y = self.block(x)
        X = self.link(x)

        return F.max_pool2d(F.relu(Y + X), 2, 2)

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.block1 = ResBlock(1, 16)
        self.block2 = ResBlock(16, 64)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_classes)
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
```

**Résultats :** ValAcc = 78.9 %

---

### Expérience 3

**Fichier modèle :** `resneXtlikeNet0.pth`
**Logs TensorBoard :** `runs/resneXtlike_train0`

#### Architecture (ResNeXt)

```python
class ResXBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True)
        )
        self.link = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, 1, 1)
        )

    def forward(self, x):
        Y = self.block(x)
        X = self.link(x)

        return F.max_pool2d(F.relu(torch.concat([X, Y], 1)), 2, 2)

class ResNeXt(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.block1 = ResXBlock(1, 16)
        self.block2 = ResXBlock(16, 64)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_classes)
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
```

**Training :** même protocole que précédemment.
**Résultats :** ValAcc = 80.5 %

---

### Expérience 4

**Fichier modèle :** `InceptionNet0.pth`
**Logs TensorBoard :** `runs/InceptionNetlike_train0`

#### Architecture (InceptionNet)

```python
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 4, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, out_channels//4, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels, 4, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, out_channels//4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels, 4, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, out_channels//4, 5, 1, 2, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels, 4, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, out_channels//4, 7, 1, 3, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        Y1 = self.block1(x)
        Y2 = self.block2(x)
        Y3 = self.block3(x)
        Y4 = self.block4(x)

        return F.max_pool2d(F.relu(torch.concat([Y1, Y2, Y3, Y4], 1)), 2, 2)

class InceptionNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.block1 = InceptionBlock(1, 16)
        self.block2 = InceptionBlock(16, 64)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_classes)
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
```

**Training :** même protocole que précédemment.
**Résultats :** ValAcc = 79.5 %

---

### Expérience 5

**Fichier modèle :** `rnn_model0.pth`
**Logs TensorBoard :** `runs/rnn_squencerTrain0`

#### Architecture (RNNClassifier)

```python
class RNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, cls_dim):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, 2, bidirectional=True, batch_first=True)
        self.out = nn.Linear(hidden_dim * 4, cls_dim)

    def forward(self, x):
        output_sequence, final_hidden_state = self.rnn(x.squeeze(1))
        y = final_hidden_state.permute(1, 0, 2).flatten(start_dim=1)

        return self.out(y)
```

**Training :** même protocole que précédemment.
**Résultats :** ValAcc = 24.9 %

---

### Expérience 6

**Fichier modèle :** `gru_model0.pth`
**Logs TensorBoard :** `runs/gru_squencerTrain0`

#### Architecture (GRUClassifier)

```python
class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, cls_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, 2, bidirectional=True, batch_first=True)
        self.out = nn.Linear(hidden_dim * 4, cls_dim)

    def forward(self, x):
        output_sequence, final_hidden_state = self.gru(x.squeeze(1))
        y = final_hidden_state.permute(1, 0, 2).flatten(start_dim=1)

        return self.out(y)
```

**Training :** même protocole que précédemment.
**Résultats :** ValAcc = 80.0 %

---

### Expérience 7

**Fichier modèle :** `lstm_model0.pth`
**Logs TensorBoard :** `runs/lstm_squencerTrain0`

#### Architecture (LSTMClassifier)

```python
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, cls_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, 
                            bidirectional=True, batch_first=True)

        self.out = nn.Linear(hidden_dim * 2, cls_dim)


    def forward(self, x):
        output_sequence, (h_n, c_n) = self.lstm(x.squeeze(1))
        final_layer_h_n = h_n[-2:, :, :]
            
        y = final_layer_h_n.permute(1, 0, 2).flatten(start_dim=1)
        return self.out(y)
```

**Training :** même protocole que précédemment.
**Résultats :** ValAcc = 81.2 %

---

### Expérience 8

**Fichier modèle :** `lstmAttn_model0.pth`
**Logs TensorBoard :** `runs/lstmAttn_squencerTrain0`

#### Architecture (BiLSTMAttnClassifier)

```python
class BiLSTMAttnClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, cls_dim):
        super().__init__()
        self.lstm_cell = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        self.attn = nn.Parameter(torch.randn(hidden_dim * 2))
        self._out = nn.Linear(hidden_dim * 2, cls_dim)

    def forward(self, inputs):
        outputs, (hn, c_n) = self.lstm_cell(inputs.squeeze(1))

        attn_scores = torch.matmul(outputs, self.attn)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)

        context = torch.sum(attn_weights * outputs, dim = 1)

        logits = self._out(context)
        return logits
```

**Training :** même protocole que précédemment.
**Résultats :** ValAcc = 80.8 %

---

### Expérience 9

**Fichier modèle :** `HybridNet0.pth`
**Logs TensorBoard :** `runs/HybridNetTrain0`

#### Architecture (HybridNet)

```python
class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//4, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
    
    def forward(self, x):
        return self.block(x)
    
class HybridNet(nn.Module):
    def __init__(self, hidden_dim, cls_dim):
        super().__init__()
        self.bloc1 = VGGBlock(1, 16)
        self.bloc2 = VGGBlock(16, 64)

        self.lstm = nn.LSTM(64, hidden_dim, 1, batch_first=True, bidirectional=True)

        self.attn = nn.Parameter(torch.randn(hidden_dim*2))
        self._out = nn.Linear(hidden_dim * 2, cls_dim)

    def forward(self, x):
        x = self.bloc2(self.bloc1(x))
        x = torch.flatten(x, 2).permute(0, 2, 1)
        outputs, (h_n, c_n) = self.lstm(x)

        attn_scores = torch.matmul(outputs, self.attn)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)

        context = torch.sum(attn_weights * outputs, dim = 1)

        logits = self._out(context)
        return logits
```

**Training :** même protocole que précédemment.
**Résultats :** ValAcc = 80.6 %

---

### Expérience 10

**Fichier modèle :** `transformer0.pth`
**Logs TensorBoard :** `runs/transformer_fromscratch_train0`

#### Architecture (VisionTransformer)

```python
class SimpleViT(nn.Module):
    def __init__(self, img_size=28, patch_size=4, num_classes=63, dim=8, depth=2, heads=2):
        super().__init__()

        num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.att_token = nn.Parameter(torch.randn(1, 1, dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=64, dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.patch_size = patch_size
        self.dim = dim

    def forward(self, x):
        batch, canal, height, width = x.shape
        p = self.patch_size

        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(batch, canal, -1, p, p)
        x = x.permute(0, 2, 1, 3, 4).flatten(2)

        x = self.patch_embed(x)

        att_tokens = self.att_token.expand(batch, -1, -1)
        x = torch.cat((att_tokens, x), dim=1)

        x = x + self.pos_embedding[:, :x.size(1), :]

        x = self.transformer(x)

        cls_output = x[:, 0]
        out = self.mlp_head(cls_output)

        return out
```

**Training :** même protocole que précédemment.
**Résultats :** ValAcc = 32.9 %

*Note : Temps d'entrainement trop court/trop peu de données pour des performances convaincantes avec cette architecture -- TODO : pretraining*
