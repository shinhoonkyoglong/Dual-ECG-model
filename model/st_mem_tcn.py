import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np

__all__ = ['ST_MEM_TCN_MAE', 'stmem_tcn_base', 'stmem_tcn_finetune']


# --------------------------------------------------------
# 1. TCN Building Blocks
# --------------------------------------------------------

class TemporalBlock(nn.Module):
    """
    TCN의 기본 블록: Dilated Conv -> ReLU -> Dropout -> Residual
    입력과 출력의 시퀀스 길이(seq_len)를 동일하게 유지합니다.
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # 첫 번째 Dilated Convolution
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        # 두 번째 Dilated Convolution
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)

        # Residual 연결 시 채널 수가 다르면 1x1 Conv로 맞춰줌
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU(inplace=True)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        # Padding으로 인해 길이가 약간 달라질 경우 잘라냄 (Same Padding 보정)
        if out.size(2) != x.size(2):
            out = out[:, :, :x.size(2)]

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    """
    TCN Encoder: Dilated Convolution을 쌓아 넓은 수용 영역(Receptive Field) 확보
    """

    def __init__(self, num_leads=12, embed_dim=256, seq_len=2250, kernel_size=15, dropout=0.2, num_classes = 2, **kwargs):
        super().__init__()
        self.seq_len = seq_len

        # 채널 확장 전략: 입력 -> 64 -> 128 -> 256 ...
        num_channels = [64, 64, 128, 128, embed_dim, embed_dim]
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i  # 1, 2, 4, 8, 16, 32...
            in_channels = num_leads if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            # Same Padding 계산: (K-1) * D // 2
            padding = (kernel_size - 1) * dilation_size // 2

            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]

        self.tcn_network = nn.Sequential(*layers)

        # Fine-tuning/Linear Probing 시 사용될 Head (Pretrain때는 사용 안함)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim, num_classes)  # num_classes는 기본값 2, 필요시 수정

    def forward(self, x):
        # Fine-tuning 시 호출되는 forward
        x = self.tcn_network(x)  # [B, embed_dim, L]
        x = self.gap(x).squeeze(-1)  # [B, embed_dim]
        return self.head(x)

    def forward_features(self, x):
        # Pre-training 시 호출: 시간축 정보 유지
        return self.tcn_network(x)  # [B, embed_dim, L]


class TCNDecoder(nn.Module):
    """
    TCN Decoder: Feature Map을 원래 채널 수로 복원
    Encoder가 길이를 유지하므로 구조가 매우 단순함.
    """

    def __init__(self, embed_dim=256, num_leads=12, **kwargs):
        super().__init__()
        self.final_conv = nn.Conv1d(embed_dim, num_leads, kernel_size=1)

    def forward(self, x):
        return self.final_conv(x)  # [B, num_leads, L]


# --------------------------------------------------------
# 2. ST-MEM TCN MAE Wrapper
# --------------------------------------------------------

class ST_MEM_TCN_MAE(nn.Module):
    def __init__(self,
                 patch_size=75,
                 num_leads=12,
                 seq_len=2250,
                 embed_dim=256,
                 **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.seq_len = seq_len
        self.num_leads = num_leads
        self.num_patches = seq_len // patch_size

        # 인코더 & 디코더
        self.encoder = TCNEncoder(num_leads, embed_dim, seq_len, **kwargs)
        self.decoder = TCNDecoder(embed_dim, num_leads, **kwargs)

    def mask_input(self, x, mask_ratio):
        """
        입력 x에 마스크를 적용합니다.
        x: [B, C, L]
        return: x_masked, mask
        """
        B, C, L = x.shape
        L_patch = self.num_patches

        # 1. 마스킹할 패치 개수 결정
        num_mask = int(mask_ratio * L_patch)

        # 2. 랜덤 노이즈로 마스킹 위치 결정
        noise = torch.rand(B, L_patch, device=x.device)

        # 낮은 noise 값을 가진 순서대로 정렬 (작은 값이 마스킹 대상이 되거나 보존 대상이 됨)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 3. 마스크 생성 (0: Keep, 1: Mask) -> MAE 관례
        # noise가 작은 앞쪽 num_mask개를 마스킹(1) 처리
        mask = torch.ones([B, L_patch], device=x.device)
        mask[:, :L_patch - num_mask] = 0  # 앞쪽을 0(Keep)으로 설정하면 -> 뒤쪽이 Mask됨
        # 혹은 셔플링 방식에 따라 유연하게 설정. 여기선 일반적 방식 사용:

        mask_binary = torch.zeros([B, L_patch], device=x.device)
        mask_binary[:, :num_mask] = 1  # 앞쪽 num_mask개를 1(Mask)로 설정

        # 원래 순서로 복구
        mask = torch.gather(mask_binary, dim=1, index=ids_restore)  # [B, L_patch]

        # 4. 입력에 마스크 적용을 위해 차원 확장 (Upsample Mask)
        # Mask: [B, L_patch] -> [B, 1, L_patch] -> [B, 1, L_patch, patch_size] -> [B, 1, L]
        mask_expanded = mask.unsqueeze(1).repeat_interleave(self.patch_size, dim=2)  # [B, 1, L]

        # x_masked: 마스킹 된 부분은 0으로 처리
        # mask가 1인 부분이 지워져야 하므로 (1 - mask)를 곱함
        x_masked = x * (1 - mask_expanded)

        return x_masked, mask

    def forward(self, series, mask_ratio=0.5):
        # series: [B, 12, 2250]

        # 1. 마스킹 적용 (Masking Input)
        # 학습의 핵심: 인코더는 구멍 뚫린(0으로 채워진) 입력을 봅니다.
        x_masked, mask = self.mask_input(series, mask_ratio)

        # 2. 인코딩 (Encoding)
        # TCN은 구멍 뚫린 데이터에서 주변 문맥(Receptive Field)을 통해 정보를 채워넣는 법을 배웁니다.
        latent = self.encoder.forward_features(x_masked)  # [B, embed_dim, L]

        # 3. 디코딩 (Decoding)
        pred = self.decoder(latent)  # [B, 12, 2250]

        # 4. 손실 계산 (Loss Calculation)
        recon_loss = self.forward_loss(series, pred, mask)

        # main_pretrain.py가 요구하는 리턴 포맷 준수
        return {"loss": recon_loss, "pred": pred, "mask": mask}

    def forward_loss(self, series, pred, mask):
        """
        series: 원본 [B, C, L] -> [B, 12, 2250]
        pred: 예측본 [B, C, L] -> [B, 12, 2250]
        mask: 마스크 [B, L_patch] -> [B, 30] 
        """
        # 1. MSE Loss 계산 (채널 방향 평균)
        # 결과: [B, 12, 2250] -> [B, 2250]
        loss = (pred - series) ** 2
        loss = loss.mean(dim=1) 

        # 2. 마스크 확장 (Patch Domain -> Time Domain)
        # [B, 30] -> [B, 30 * 75] = [B, 2250]
        # 각 패치 마스크 값(0 또는 1)을 patch_size만큼 반복합니다.
        mask_expanded = torch.repeat_interleave(mask, self.patch_size, dim=1)

        # 3. Loss에 마스크 적용
        # 이제 loss와 mask_expanded 둘 다 [B, 2250]이므로 곱셈이 가능합니다.
        loss = (loss * mask_expanded).sum() / mask_expanded.sum()
        
        return loss


# --------------------------------------------------------
# 3. Factory Functions (Called by main_pretrain.py)
# --------------------------------------------------------

def stmem_tcn_base(**kwargs):
    """Pre-training을 위한 모델 생성"""
    model = ST_MEM_TCN_MAE(**kwargs)
    return model


def stmem_tcn_finetune(**kwargs):
    """Fine-tuning/Evaluation을 위한 인코더 생성"""
    # kwargs에서 MAE 전용 인자(patch_size 등)가 있다면 제거하거나 무시하도록 처리 필요할 수 있음
    # 여기선 안전하게 필요한 인자만 건넴
    model = TCNEncoder(**kwargs)
    return model