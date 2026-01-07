import torch
import torch.nn as nn
from typing import Optional

# 기존 ST_MEM_ViT 클래스 (encoder/st_mem_vit.py에 있다고 가정)를 임포트합니다.
# Base 모델을 반환하는 함수도 임포트합니다.
from .st_mem_vit import st_mem_vit_base 


class PairDiffPredictor(nn.Module):
    """
    ST-MEM ViT Base 인코더를 공유하며, 두 입력(pre/post)의 차이 벡터를 기반으로
    호전(1) 여부를 예측하는 모델입니다.
    """
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        
        # Base 모델의 특징 벡터 차원 (width)
        D_out = 768 
        
        # 1. Base 인코더 로드 (num_classes=None으로 설정하여 최종 Linear Head는 Identity로 만듦)
        # **kwargs는 seq_len, patch_size, num_leads 등을 전달합니다.
        # 인코더는 Base 모델을 사용합니다.
        self.encoder_body = st_mem_vit_base(num_classes=None, **kwargs)
        
        # 2. 차이 벡터를 위한 새로운 MLP 분류 헤드 (Diff Head)
        # 입력 차원: D_out (768)
        self.diff_head = nn.Sequential(
            nn.Linear(D_out, D_out // 2),  # 768 -> 384
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(D_out // 2, 1) # 384 -> 1개의 스칼라 값으로.
        )
        self.diff_head2 = nn.Sequential(
            nn.Linear(768, 512),  # 768 -> 384
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2) # 2 class 분류
        )
        self.attn = nn.MultiheadAttention(embed_dim=768, num_heads=4, batch_first=True)
        
    def forward(self, x_pre: torch.Tensor, x_post: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_pre: 퇴원 전 ECG 데이터 텐서 [B, C, L]
            x_post: 퇴원 후 ECG 데이터 텐서 [B, C, L]
        """
        
        # 3. 인코더 공유 및 특징 벡터 추출
        # self.encoder_body는 ST_MEM_ViT의 인스턴스이며, forward_encoding을 사용합니다.
        z_pre = self.encoder_body.forward_encoding(x_pre)  # 특징 벡터 [B, 768]
        z_post = self.encoder_body.forward_encoding(x_post) # 특징 벡터 [B, 768]

        query = z_post.unsqueeze(1) # [B, 1, 256]
        key = z_pre.unsqueeze(1)    # [B, 1, 256]
        value = z_pre.unsqueeze(1)  # [B, 1, 256]

# 2. 어텐션 연산
        attn_output, _ = self.attn(query, key, value)

# 3. 차원 복구 및 잔차 연결 (선택사항이지만 보통 원본을 더해줌)
        D = attn_output.squeeze(1) + z_post
        logits = self.diff_head2(D)
        
        return logits