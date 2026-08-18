import torch
import torch.nn as nn


class RestorationAutoencoder(nn.Module):
    def __init__(self):
        super(RestorationAutoencoder, self).__init__()

        # ---------- ENCODER (kept as separate layers so we can tap into them) ----------
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # (3,256,256) -> (32,128,128)
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (64,64,64)
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> (128,32,32)
            nn.ReLU(inplace=True),
        )

        # ---------- DECODER ----------
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (64,64,64)
            nn.ReLU(inplace=True),
        )
        # dec2 takes dec3's output CONCATENATED with enc2's output (skip connection)
        # 64 (from dec3) + 64 (from enc2) = 128 input channels
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),  # -> (32,128,128)
            nn.ReLU(inplace=True),
        )
        # dec1 takes dec2's output CONCATENATED with enc1's output (skip connection)
        # 32 (from dec2) + 32 (from enc1) = 64 input channels
        # Output is FEATURES here (not the final image) so we can add a raw-input skip below.
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> (32,256,256)
            nn.ReLU(inplace=True),
        )

        # ---------- FINAL FUSION LAYER ----------
        # Combines decoder features (32 channels) with the RAW degraded input (3 channels)
        # at full resolution. This gives the model direct access to fine texture that's
        # still faintly present in the noisy input, instead of forcing it to regenerate
        # every detail purely from compressed features -> helps preserve sharpness.
        self.final = nn.Sequential(
            nn.Conv2d(32 + 3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encoder path, saving intermediate outputs for skip connections
        e1 = self.enc1(x)   # (32,128,128)
        e2 = self.enc2(e1)  # (64,64,64)
        e3 = self.enc3(e2)  # (128,32,32)

        # Decoder path, concatenating matching-resolution encoder features
        d3 = self.dec3(e3)                          # (64,64,64)
        d3 = torch.cat([d3, e2], dim=1)              # (128,64,64) - skip connection

        d2 = self.dec2(d3)                           # (32,128,128)
        d2 = torch.cat([d2, e1], dim=1)               # (64,128,128) - skip connection

        d1 = self.dec1(d2)                            # (32,256,256) - features, not final image yet

        # Full-resolution skip: fuse decoder features with the original raw input
        fused = torch.cat([d1, x], dim=1)              # (35,256,256)
        output = self.final(fused)                     # (3,256,256)
        return output


if __name__ == "__main__":
    # Quick sanity test: does a fake image pass through without shape errors?
    model = RestorationAutoencoder()

    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)

    print("Input shape: ", dummy_input.shape)
    print("Output shape:", output.shape)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {num_params:,}")

    assert output.shape == dummy_input.shape, "Output shape does not match input shape!"
    print("PASSED: output shape matches input shape.")