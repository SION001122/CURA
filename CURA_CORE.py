import torch
import torch.nn as nn
import torch.nn.functional as F

class CURA_CORE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # 게이트 회로: R (저항)처럼 입력 신호의 흐름을 조절
        # Gate circuit: acts like a resistor (R), regulating the flow of input signals
        self.gate_fc = nn.Linear(input_dim, hidden_dim)
        
        # 잔차 경로: C (커패시터)처럼 이전 정보를 유지/누적하는 경로
        # Residual path: works like a capacitor (C), preserving and accumulating past information
        self.residual_fc = nn.Linear(input_dim, hidden_dim)
        
        # 비선형 증폭기: 활성화와 선형변환으로 신호 증폭 (Op-Amp 느낌)
        # Nonlinear amplifier: amplifies signals using activation and linear transformation (Op-Amp analogy)
        self.relu_linear = nn.Linear(hidden_dim, hidden_dim)
        
        # CNN 필터: 특정 지역 정보를 강조하거나 감쇠하는 필터 (Band-pass filter 느낌)
        # CNN filter: emphasizes or attenuates local information (analogous to a band-pass filter)
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        
        # 출력 변환기: 원하는 출력 차원으로 변환 (최종 출력 버퍼 역할)
        # Output transformer: converts into desired output dimension (final output buffer)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 게이팅(R): 선택적 정보 통과
        # Gating (R): selective information flow
        gate = torch.sigmoid(self.gate_fc(x))
        
        # 잔차(C): 이전 정보 유지
        # Residual (C): keeps past information
        residual = self.residual_fc(x)
        
        # R-C 조합 효과: 감쇠된 정보 흐름
        # R-C combined effect: attenuated but preserved signal flow
        x = gate * residual + residual

        # 증폭기: 비선형적으로 신호 강화
        # Amplifier: nonlinearly strengthens the signal
        x = F.relu(self.relu_linear(x))

        # CNN 필터: 국소적 변화 감지/정제
        # CNN filter: detects and refines local variations
        x_cnn = self.conv(x.unsqueeze(1)).squeeze(1)

        # 출력 버퍼: 최종 결과 변환
        # Output buffer: final transformation to output
        out = self.output(x_cnn)
        return out.squeeze()
