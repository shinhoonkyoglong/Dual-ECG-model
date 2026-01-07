import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np

__all__ = ['ST_MEM_1DCNN_MAE', 'stmem_1dcnn_base', 'stmem_1dcnn_finetune']


# --- 1D CNN Encoder Helpers ---

class ConvBlock1D(nn.Module):
    """표준 1D 합성곱 블록"""

    def __init__(self, in_channels, out_channels, kernel_size=15, stride=1, padding=7):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CNNEncoder(nn.Module):
    """MAE Pre-training을 위한 1D CNN 인코더"""

    # [수정된 부분]: seq_len=2250 적용
    def __init__(self, num_leads=12, embed_dim=256, seq_len=2250,num_classes=2,**kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_leads = num_leads
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        # 1. 초기 컨볼루션: Stride 4. (2250 -> 563)
        self.input_conv = ConvBlock1D(num_leads, 64, kernel_size=32, stride=4, padding=16)

        # 2. ResNet-like 블록
        self.layer1 = nn.Sequential(  # Stride 4. (563 -> 141)
            ConvBlock1D(64, 128, kernel_size=15, stride=4, padding=7),
            ConvBlock1D(128, 128),
        )
        self.layer2 = nn.Sequential(  # Stride 2. (141 -> 71)
            ConvBlock1D(128, embed_dim, kernel_size=7, stride=2, padding=3),
            ConvBlock1D(embed_dim, embed_dim),
        )
        # 최종 특징 맵 길이 (L_latent): 71

        self.final_proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward_features(self, x):
        # x shape: [B, num_leads, 2250]
        x = self.input_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.final_proj(x)
        x = self.gap(x)         # [B, embed_dim, 1]
        return x.squeeze(-1)    # [B, embed_dim] 👈 특징 벡터 (D_out=256)만 반환
    
    # forward도 수정:
    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)


class CNNDecoder(nn.Module):
    """1D CNN 디코더 (역합성곱/업샘플링)"""

    # [수정된 부분]: seq_len=2250 적용
    def __init__(self, embed_dim=256, decoder_embed_dim=128, num_leads=12, seq_len=2250):
        super().__init__()

        self.proj_up = nn.Conv1d(embed_dim, decoder_embed_dim, kernel_size=1)

        # 역합성곱을 사용하여 길이 복원 (Stride 2, 4, 4의 역순)
        self.deconv_layer2 = nn.Sequential(  # Stride 2 (71 -> 142)
            nn.ConvTranspose1d(decoder_embed_dim, 128, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.deconv_layer1 = nn.Sequential(  # Stride 4 (142 -> 568)
            nn.ConvTranspose1d(128, 64, kernel_size=15, stride=4, padding=7, output_padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # 마지막 길이를 정확히 seq_len=2250으로 맞추기
        self.upsample = nn.Upsample(size=seq_len, mode='linear', align_corners=False)

        # 최종 출력 채널을 리드 수로 맞추기
        self.final_conv = nn.Conv1d(64, num_leads, kernel_size=1)

    def forward(self, x):
        # x shape: [B, embed_dim, 71]
        x = self.proj_up(x)
        x = self.deconv_layer2(x)
        x = self.deconv_layer1(x)

        x = self.upsample(x)
        x = self.final_conv(x)
        # 최종 출력 shape: [B, num_leads, 2250]
        return x


# --- ST_MEM 1D-CNN MAE Wrapper (나머지 로직은 seq_len=2250에 맞춰 자동 조정됨) ---

class ST_MEM_1DCNN_MAE(nn.Module):
    # [수정된 부분]: seq_len=2250 적용
    def __init__(self,
                 patch_size,
                 num_leads=12,
                 seq_len=2250,
                 embed_dim=256,
                 decoder_embed_dim=128,
                 **kwargs):
        super().__init__()

        self.num_leads = num_leads
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size

        self.encoder = CNNEncoder(num_leads, embed_dim, seq_len)
        self.decoder = CNNDecoder(embed_dim, decoder_embed_dim, num_leads, seq_len)

        self.initialize_weights()
    # CNN은 initialize필요 없음.
    def initialize_weights(self): # <--- 이 메서드를 반드시 클래스 내부에 추가해야 합니다
        pass
        
    def forward(self, series, mask_ratio=0.75):
        
        # 1. 인코딩 (전체 신호 사용)
        latent = self.forward_encoder(series)
        
        # 2. 디코딩 (전체 특징 맵 사용)
        pred = self.forward_decoder(latent)
        
        # 3. 손실 계산을 위한 마스크 생성
        mask = self.mask_and_get_indices(series, mask_ratio)
        
        # 4. 손실 계산
        recon_loss = self.forward_loss(series, pred, mask)
        
        # main_pretrain.py가 요구하는 딕셔너리 형태로 반환
        return {"loss": recon_loss, "pred": pred, "mask": mask}
        
    def forward_encoder(self, series):
        """1D CNN은 마스킹되지 않은 입력만 받는 구조가 아니므로, 전체 신호를 인코딩합니다."""
        # CNNEncoder 인스턴스 (self.encoder)의 forward를 호출
        latent = self.encoder(series) # [B, embed_dim, L_latent]
        return latent
        
    def forward_decoder(self, latent):
        """전체 잠재 코드를 디코더에 전달하여 복원합니다."""
        # CNN 디코더 인스턴스 (self.decoder)의 forward를 호출
        pred = self.decoder(latent) # [B, num_leads, seq_len]
        return pred
    def mask_and_get_indices(self, series, mask_ratio):
        """
        1D 시계열 데이터를 블록 단위로 마스킹할 인덱스를 결정하고 마스크를 생성합니다.
        Loss 계산을 위해 [B, L_patch] 차원의 mask를 생성합니다.
        series shape: [B, C, L]
        """
        B, C, L = series.shape
        L_patch = self.num_patches
        
        # 1. 마스킹 비율에 따라 제거할 블록 수 결정
        num_mask = int(mask_ratio * L_patch)
        
        # 2. 마스킹 위치 결정 (L_patch 길이의 시퀀스에서)
        noise = torch.rand(B, L_patch, device=series.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1) # 원본 위치로 복원 인덱스
        
        # 3. 마스크 생성 (1: 손실 계산, 0: 손실 계산 제외)
        mask = torch.zeros([B, L_patch], device=series.device)
        # 마스킹할 부분(num_mask 개)을 1로 설정하여 손실 계산에 포함
        mask[:, :num_mask] = 1.
        
        # 4. 마스크를 원래 시간적 위치로 재배열
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask # [B, L_patch] 형태
        
    def forward_loss(self, series, pred, mask):
        """Loss는 마스킹된 부분에 대해서만 MSE를 계산합니다."""
        
        # 1. 신호를 마스킹 블록으로 분할 (Patchify)
        # target, pred_blocks shape: [B, L_patch, C * patch_size]
        
        # Target: [B, C, L] -> [B, C, L_patch, patch_size] -> [B, L_patch, C*patch_size]
        target = series.unfold(2, self.patch_size, self.patch_size).permute(0, 2, 1, 3).contiguous()
        pred_blocks = pred.unfold(2, self.patch_size, self.patch_size).permute(0, 2, 1, 3).contiguous()
        
        target = target.view(target.shape[0], target.shape[1], -1) 
        pred_blocks = pred_blocks.view(pred_blocks.shape[0], pred_blocks.shape[1], -1)

        # 2. 마스크 확장: [B, L_patch] -> [B, L_patch, C*patch_size]
        # Loss 계산을 위해 마스크를 시그널 차원까지 확장합니다.
        mask = mask.unsqueeze(-1).expand_as(target)
        
        # 3. 마스킹된 블록에 대해서만 MSE 계산
        loss = (pred_blocks - target) ** 2
        loss = loss * mask # 마스킹된 부분(1)의 손실만 남김
        
        # 4. 평균 손실 계산 (마스킹된 부분의 개수로 나눔)
        recon_loss = loss.sum() / mask.sum() 
        return recon_loss
    
def stmem_1dcnn_base(**kwargs):
    """YAML 설정에서 호출되는 기본 1D CNN MAE 모델"""
    model = ST_MEM_1DCNN_MAE(**kwargs)
    return model


def stmem_1dcnn_finetune(**kwargs):
    """Fine-tuning 시 Encoder만 추출하기 위한 더미 함수 추가"""
    model = CNNEncoder(**kwargs)
    return model

# 이 파일이 models/__init__.py 등에서 임포트되어야 합니다.