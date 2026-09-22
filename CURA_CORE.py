import torch
import torch.nn as nn
import torch.nn.functional as F

class CURA_CORE_V1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # 게이트 회로: R (저항)처럼 입력 신호의 흐름을 조절
        # Gate circuit: acts like a resistor (R), regulating the flow of input signals
        # ゲート回路: R (抵抗器)のように入力信号の流量を調整
        self.gate_fc = nn.Linear(input_dim, hidden_dim)
        
        # 잔차 경로: C (커패시터)처럼 이전 정보를 유지/누적하는 경로
        # Residual path: works like a capacitor (C), preserving and accumulating past information
        # 残差パス: C (コンデンサ/キャパシタ)のように過去の情報を保持・蓄積するパス
        self.residual_fc = nn.Linear(input_dim, hidden_dim)
        
        # 비선형 증폭기: 활성화와 선형변환으로 신호 증폭 (Op-Amp 느낌)
        # Nonlinear amplifier: amplifies signals using activation and linear transformation (Op-Amp analogy)
        # 非線形増幅器: 活性化関数と線形変換で信号を増幅 (オペアンプのような役割)
        self.relu_linear = nn.Linear(hidden_dim, hidden_dim)
        
        # CNN 필터: 특정 지역 정보를 강조하거나 감쇠하는 필터 (Band-pass filter 느낌)
        # CNN filter: emphasizes or attenuates local information (analogous to a band-pass filter)
        # CNNフィルタ: 局所的な特徴を強調または減衰させるフィルタ (バンドパスフィルタのような役割)
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        
        # 출력 변환기: 원하는 출력 차원으로 변환 (최종 출력 버퍼 역할)
        # Output transformer: converts into desired output dimension (final output buffer)
        # 出力変換器: 希望する出力次元へ変換 (最終出力バッファの役割)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 게이팅(R): 선택적 정보 통과
        # Gating (R): selective information flow
        # ゲーティング (R): 選択的な情報の通過
        gate = torch.sigmoid(self.gate_fc(x))
        
        # 잔차(C): 이전 정보 유지
        # Residual (C): keeps past information
        # 残差 (C): 過去の情報を保持
        residual = self.residual_fc(x)
        
        # R-C 조합 효과: 감쇠된 정보 흐름
        # R-C combined effect: attenuated but preserved signal flow
        # R-C結合効果: 減衰しつつも保持された信号の流れ
        x = gate * residual + residual

        # 증폭기: 비선형적으로 신호 강화
        # Amplifier: nonlinearly strengthens the signal
        # 増幅器: 非線形に信号を強化
        x = F.relu(self.relu_linear(x))

        # CNN 필터: 국소적 변화 감지/정제
        # CNN filter: detects and refines local variations
        # CNNフィルタ: 局所的な変化の検出および精製
        x_cnn = self.conv(x.unsqueeze(1)).squeeze(1)

        # 출력 버퍼: 최종 결과 변환
        # Output buffer: final transformation to output
        # 出力バッファ: 最終結果への変換
        out = self.output(x_cnn)
        return out.squeeze()
