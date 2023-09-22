import torch
import torch.nn as nn
import torch.nn.functional as F


class Spec_match(nn.Module):
    def __init__(self, n_fft=256, hop_length=64):
        super(Spec_match, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, mix, aux):
        """
        mix (torch.Tensor): batched single-channel mixture tensor with M (M=1) audio channels and N samples. [B, M, N]
        aux (torch.Tensor): batched single-channel auxiliary speech tensor with M (M=1) audio channels and N samples. [B, M, N]
        """
        mix = mix.squeeze(1)
        mix = torch.stft(mix, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True).unsqueeze(1)  # [B, M, F, T]
        mix = mix.permute(0, 1, 3, 2)  # [B, M, T, F]

        aux = aux.squeeze(1)
        aux = torch.stft(aux, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True).unsqueeze(1)  # [B, M, F, T]
        aux = aux.permute(0, 1, 3, 2)  # [B, M, T, F]

        mix_batch = torch.split(mix, 1, dim=0)
        aux_batch = torch.split(aux, 1, dim=0)
        aux_match = [None] * mix.shape[0]
        for i in range(len(mix_batch)):
            mix_ = mix_batch[i].squeeze()  # [T, F]
            aux_ = aux_batch[i].squeeze()
            T_mix, T_aux = mix_.shape[0], aux_.shape[0]  # Here, we assume the T_aux is always larger than T_mix.
            hs = 126  # hop size
            Length = (T_aux - T_mix) // hs
            aux_r = F.pad(aux_.real, (0, 0, 0, hs - (T_aux - T_mix) % hs), mode='constant')
            aux_i = F.pad(aux_.imag, (0, 0, 0, hs - (T_aux - T_mix) % hs), mode='constant')
            aux_ = torch.complex(aux_r, aux_i).unsqueeze(0)  # [M, T, F]
            aux_ = [aux_[:, j * hs: (j * hs + T_mix), :] for j in range(Length + 2)]
            similarity = [(mix_.real * aux_.real + mix_.imag * aux_.imag) / (torch.abs(mix_) * torch.abs(aux_)) for aux_ in aux_]
            aux_match[i] = aux_[max(range(len(similarity)), key=lambda p: torch.median(similarity[p]).item())]

        aux_match = torch.stack(aux_match, dim=0)
        # mix_batch = torch.cat((mix.real, mix.imag), dim=1)  # [B, 2*M, T, F]
        aux_match = torch.cat((aux_match.real, aux_match.imag), dim=1)  # [B, 2*M, T, F]

        return aux_match

if __name__ == '__main__':
    mix = torch.randn(4, 1, 32000)
    aux = torch.randn(4, 1, 58400)
    model = Spec_match()
    out = model(mix, aux)
    print(out.shape)