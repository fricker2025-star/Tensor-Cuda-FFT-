from fft_lm import train_fixed_full as t

c = t.TrainConfig()
print('stage1_epochs', c.stage1_epochs)  # moved to tools/
print('stage2_epochs', c.stage2_epochs)
print('has_sawtooth_lr', hasattr(t, 'sawtooth_lr'))
