import torch
import torch.nn as nn
from typing import Optional

# CNNEncoder 클래스를 임포트합니다.
from models.st_mem_1dcnn import CNNEncoder 


class PairDiffCNN(nn.Module):
    """
    1D CNN 인코더를 공유하며, 각 시점의 특징(256D)을 스칼라 점수로 압축한 후,
    그 스칼라 점수의 차이를 기반으로 분류하는 모델입니다.
    """

    def __init__(self, num_classes=2, embed_dim=256, **kwargs):
        super().__init__()

        D_out = embed_dim  # 256

        # 1. Base 인코더 로드 (CNNEncoder를 로드)
        # Note: CNNEncoder는 head를 포함하며, forward_features를 사용해야 합니다.
        self.encoder_body = CNNEncoder(num_classes=num_classes, embed_dim=embed_dim, **kwargs)

        # 📌 수정: Score Head -> self.diff_head
        self.diff_head = nn.Sequential(
            nn.Linear(D_out, D_out // 2),  # 256 -> 128
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(D_out // 2, 1)  # 128 -> 1 (단일 스칼라 점수)
        )

        # 📌 수정: Final Head -> self.diff_head2
        self.diff_head2 = nn.Sequential(
            nn.Linear(256, 512),  # 1 -> 64
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # 64 -> 2 (호전/그 외 로짓)
        )
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        
    def forward(self, x_pre: torch.Tensor, x_post: torch.Tensor) -> torch.Tensor:
        # CNNEncoder의 forward_features를 사용해야 합니다. (이 로직은 별도 파일에 추가되어야 함)
        # 1. 인코더 공유 및 특징 벡터 추출 (256D)
        z_pre = self.encoder_body.forward_features(x_pre)  # 특징 벡터 [B, 256]
        z_post = self.encoder_body.forward_features(x_post)  # 특징 벡터 [B, 256]
        
        query = z_post.unsqueeze(1) # [B, 1, 256]
        key = z_pre.unsqueeze(1)    # [B, 1, 256]
        value = z_pre.unsqueeze(1)  # [B, 1, 256]
        # 2. 어텐션 연산
        attn_output, _ = self.attn(query, key, value)

        # 3. 차원 복구 및 잔차 연결 (선택사항이지만 보통 원본을 더해줌)
        D = attn_output.squeeze(1) + z_post
        # 4. 최종 MLP 분류
        
        logits = self.diff_head2(D) # self.diff_head2 사용

        return logits
'''import torch
import torch.nn as nn
# 위에서 정의하신 TCNEncoder를 사용합니다.
from models.st_mem_tcn import TCNEncoder 

class PairDiffTCN(nn.Module):
    """
    Pretrained TCN Encoder를 사용하여 두 시점의 차이를 분석하는 모델
    """
    def __init__(self, num_classes=2, embed_dim=256, **kwargs):
        super().__init__()
        
        # 1. TCN Encoder 로드
        self.encoder_body = TCNEncoder(embed_dim=embed_dim, **kwargs)
        
        # 📌 추가된 부분: Global Average Pooling
        # TCN은 시간 축(L)이 살아있는 [B, C, L]을 출력하므로, 
        # 이를 하나의 벡터 [B, C]로 압축해야 합니다.
        self.gap = nn.AdaptiveAvgPool1d(1)

        D_out = embed_dim  # 256

        # 2. Score Head (특징 벡터 -> 스칼라 점수)
        self.diff_head = nn.Sequential(
            nn.Linear(D_out, D_out // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(D_out // 2, 1)  # 스칼라 점수 출력
        )

        # 3. Final Head (점수 차이 -> 호전 여부 로짓)
        self.diff_head2 = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_pre, x_post):
        # 1. TCN Feature Extraction: [B, 256, L]
        f_pre = self.encoder_body.forward_features(x_pre)
        f_post = self.encoder_body.forward_features(x_post)

        # 2. Pooling: [B, 256, L] -> [B, 256, 1] -> [B, 256]
        z_pre = self.gap(f_pre).squeeze(-1)
        z_post = self.gap(f_post).squeeze(-1)

        # 3. Score Calculation
        score_pre = self.diff_head(z_pre)
        score_post = self.diff_head(z_post)

        # 4. Difference & Classification
        D_scalar = score_post - score_pre
        logits = self.diff_head2(D_scalar)

        return logits'''
import torch
import torch.nn as nn
from models.st_mem_tcn import TCNEncoder # 기존 TCN 인코더 사용

class PairDiffTCN_Feature(nn.Module):
    """
    TCN Encoder를 사용하여 Feature Vector를 추출하고,
    (Post Feature - Pre Feature)의 차이 벡터를 입력으로 받아 분류하는 모델
    """
    def __init__(self, num_classes=2, embed_dim=256, **kwargs):
        super().__init__()
        
        # 1. TCN Encoder 로드 (Weights 공유)
        # kwargs로 kernel_size, dropout 등을 전달받습니다.
        self.encoder_body = TCNEncoder(embed_dim=embed_dim, **kwargs)
        
        # 2. Global Average Pooling
        # [B, 256, L] -> [B, 256, 1]
        self.gap = nn.AdaptiveAvgPool1d(1)

        # 3. Classifier Head (Feature Difference -> Class Logits)
        # 입력 차원이 '1'이 아니라 'embed_dim(256)'이 됩니다.
        self.diff_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), # 256 -> 128
            nn.BatchNorm1d(embed_dim // 2),       # 안정적인 학습을 위한 BN 추가
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(embed_dim // 2, 64),        # 128 -> 64
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(64, num_classes)            # 64 -> 2 (0 or 1)
        )

        # 📌 [핵심] 초기화 꿀팁 적용 (Bias 0)
        # 초기 출력을 0 근처로 만들어 50:50 확률에서 시작하게 함
        nn.init.constant_(self.diff_head[-1].bias, 0.0)
        nn.init.normal_(self.diff_head[-1].weight, std=0.01)

    def forward(self, x_pre, x_post):
        # 1. TCN Feature Extraction: [B, 256, L]
        # weights sharing (샴 네트워크 구조)
        f_pre = self.encoder_body.forward_features(x_pre)
        f_post = self.encoder_body.forward_features(x_post)

        # 2. Pooling: [B, 256, L] -> [B, 256]
        z_pre = self.gap(f_pre).squeeze(-1)
        z_post = self.gap(f_post).squeeze(-1)

        # 3. Feature Difference (벡터 빼기)
        # 스칼라가 아니라 256차원 벡터끼리의 차이입니다.
        diff_vector = z_post - z_pre  # [B, 256]

        # 4. Classification
        logits = self.diff_head(diff_vector)

        return logits