from .audio import load_mono_audio
import torch
import torchaudio
import random

#
# Helpers
#

def maybe(effect, p):
    return MaybeEffect(effect, p)

def one_of(*args):
    return OneOfEffect(args)

def rir(rirs):
    return RirEffect(rirs)

def background(files, min_snr = -5, max_snr = 20):
    return BackgroundEffect(files, min_snr, max_snr)

def effect(effect):
    return SimpleEffect(effect)

def sox(effect):
    return SoxEffect(effect)

def raw(effect, name):
    return RawEffect(name, effect)

def series(effects):
    return SeriesEffect(effects)

def permutate(effects):
    return PermutateEffect(effects)

def low_pass(*, low = 2000, high = 8000): # Max is 1/2 * sample rate
    return effect(lambda:sox(['sinc', '-n', str(random.randint(50, 200)), '-' + str(random.uniform(low, high))]))

def band_pass(*, low = 100, high = 1000, min_width = 2000, max_width = 4000):
    def impl():
        n = random.randint(50, 200)
        low_v = random.uniform(low, high)
        high_v = low_v + random.uniform(min_width, max_width)
        return sox(['sinc', '-n', str(n), str(low_v) + '-' + str(high_v)])
    return effect(impl)

def equalizer(*, low=100, high=4000, q_low=1, q_high=5, db_low: int = -30, db_high: int = 30):
    def impl():
        return sox(['equalizer', str(random.uniform(low, high)), str(random.randint(q_low, q_high)) + 'q',  str(random.randint(db_low, db_high))])
    return effect(impl)

def overdrive(*, gain_low=5, gain_high=40, colour_low=20, colour_high=80):
    def impl():
        return sox(['overdrive', str(random.uniform(gain_low, gain_high)), str(random.randint(colour_low, colour_high))])
    return effect(impl)

def reverbate():
    return effect(lambda:sox(['reverb', str(random.randint(0, 100)), str(random.randint(0, 100)), str(random.randint(0, 100))]))

def flanger():
    return effect(lambda:sox(['flanger']))

def phaser():
    return effect(lambda:sox(['phaser']))

def noise(*, min_level = 0, max_level = 0.2):
    return NoiseEffect(min_level, max_level)

def lost_packets(*, min_lost_segments = 0, max_lost_segments = 50, min_segment = 100, max_segment = 1000):
    return LostPacketsEffect(max_lost_segments, min_lost_segments, min_segment, max_segment)

def default_noisy_pipeline(*, rirs = [], bg = []):
    return series([

        # Room impulse response - add to clean audio to simulate room effects
        maybe(rir(rirs), 0.8),

        # Room impulse response - add to clean audio to simulate room effects
        maybe(background(bg), 0.8),

        # Background gausian noise - add to simulate microphone noise
        maybe(noise(), 0.5),

        # Main effects
        permutate([
            
            # Audio capture quality
            maybe(one_of(low_pass(), band_pass()), 0.3),

            # Audio effects
            reverbate(),
            # overdrive(),
            equalizer(),
        ]),

        # Corrupt base audio
        # maybe(lost_packets(), 0.5),
    ])

def light_noisy_pipeline(*, bg = []):
    return series([
        maybe(background(bg), 0.8),
        # maybe(noise(), 0.5)
    ])

def light_noisy_voiced_pipeline(*, bg = [], voices = []):
    return series([
        maybe(background(bg), 0.8),
        maybe(background(voices, min_snr = 1, max_snr = 20), 0.8),
        maybe(noise(), 0.5)
    ])

def light_noisy_common_pipeline():
    return series([
        permutate([
            maybe(one_of(low_pass(), band_pass()), 0.3),
            reverbate(),
            equalizer(),
        ])
    ])

#
# Base classes
#

class Effect():
    def resolve(self):
        raise NotImplementedError
    def _resolve(self, effect):
        if isinstance(effect, Effect):
            return effect.resolve()
        if isinstance(effect, list):
            return effect
        if effect is None:
            return []
        return [effect]

    def apply(self, audio, sample_rate, resolved = None):
        source_len = audio.shape[0]

        # Resolve effects
        if resolved is None:
            effects = self.resolve()
        else:
            effects = resolved

        # print("Apply effects")
        # for effect in effects:
        #     print(str(effect))

        # Process
        if effects is not None:
            sox_effects = []
            for effect in effects:
                # print("effect: " + str(effect))
                if isinstance(effect, SoxEffect):
                    sox_effects.append(effect.effect) # Store sox effect
                else:

                    # Apply sox before custom
                    if len(sox_effects) > 0:
                        audio = torchaudio.sox_effects.apply_effects_tensor(audio.unsqueeze(0), sample_rate, sox_effects, channels_first = True)[0][0]
                        sox_effects = []

                    # Apply custom effect
                    audio = effect.effect(audio, sample_rate)

            # Apply remaining sox effects
            if len(sox_effects) > 0:
                audio = torchaudio.sox_effects.apply_effects_tensor(audio.unsqueeze(0), sample_rate, sox_effects, channels_first = True)[0][0]

        # Pad or trim audio
        if audio.shape[0] < source_len:
            padding = source_len - audio.shape[0]
            audio = torch.nn.functional.pad(audio, (0, padding), value=0)
        else:
            audio = audio[0:source_len]
        
        return audio

class MaybeEffect(Effect):
    def __init__(self, effect, p):
        self.effect = effect
        self.p = p

    def resolve(self):
        if random.random() < self.p:
            return self._resolve(self.effect.resolve())
        return None

class OneOfEffect(Effect):
    def __init__(self, effects):
        self.effects = effects

    def resolve(self):
        return self._resolve(random.choice(list(self.effects)))

class RirEffect(Effect):
    def __init__(self, rirs):
        self.rirs = rirs

    def resolve(self):
        if len(self.rirs) == 0:
            return self._resolve(None)
        rir = random.choice(self.rirs)
        def apply_rir(audio, sr):
            return do_reverbrate(audio, load_mono_audio(rir, sr))
        return self._resolve(raw(apply_rir, f'rir({rir})'))

class BackgroundEffect(Effect):
    def __init__(self, files, min_snr, max_snr):
        self.files = files
        self.min_snr = min_snr
        self.max_snr = max_snr

    def resolve(self):
        if len(self.files) == 0:
            return self._resolve(None)
        f = random.choice(self.files)
        snr = random.uniform(self.min_snr, self.max_snr)
        def apply_bg(audio, sr):
            return do_background(audio, load_mono_audio(f, sr), snr)
        return self._resolve(raw(apply_bg, f'bg({f}, {str(snr)})'))

class NoiseEffect(Effect):
    def __init__(self, max_level, min_level):
        self.max_level = max_level
        self.min_level = min_level

    def resolve(self):
        level = random.uniform(self.min_level, self.max_level)
        cached_noise = [None]
        def apply_noise(audio, sr):
            if cached_noise[0] is None:
                cached_noise[0] = torch.randn_like(audio)
            return do_noise(audio, cached_noise[0], level)
        return self._resolve(raw(apply_noise, f'noise({str(level)})'))

class LostPacketsEffect(Effect):
    def __init__(self, max_lost_segments, min_lost_segments, min_segment, max_segment):
        self.max_lost_segments = max_lost_segments
        self.min_lost_segments = min_lost_segments
        self.min_segment = min_segment
        self.max_segment = max_segment

    def resolve(self):
        lost = random.randint(self.min_lost_segments, self.max_lost_segments)
        def apply_lost(audio, sr):
            return do_lost_packets(audio, lost, self.min_segment, self.max_segment)
        return self._resolve(raw(apply_lost, f'lost_packets({str(lost)})'))

class SimpleEffect(Effect):
    def __init__(self, effect):
        self.effect = effect

    def resolve(self):
        if isinstance(effect, SoxEffect):
            return self._resolve(self.effect)
        else:
            return self._resolve(self.effect())

class SeriesEffect(Effect):
    def __init__(self, effects):
        self.effects = effects

    def resolve(self):
        out = []
        for effect in self.effects:
            r = self._resolve(effect)
            if r is not None:
                out.extend(r)
        return self._resolve(out)

class PermutateEffect(Effect):
    def __init__(self, effects):
        self.effects = effects

    def resolve(self):
        out = []
        for effect in self.effects:
            r = self._resolve(effect)
            if r is not None:
                out.extend(r)
        random.shuffle(out)
        return self._resolve(out)

#
# Wrappers
#

class SoxEffect():
    def __init__(self, effect):
        self.effect = effect
    def __str__(self):
        return f"sox({self.effect})"

class RawEffect():
    def __init__(self, name, effect):
        self.name = name
        self.effect = effect
    def __str__(self):
        return self.name

#
# Audio
#

def do_reverbrate(waveforms, rir):
    assert len(waveforms.shape) == 1 # Only single dimension is allowed
    assert len(rir.shape) == 1 # Only single dimension is allowed

    # Source length
    source_len = waveforms.shape[0]

    # NOTE: THIS ALL NOT NEEDED for fftconvolve
    # Flip for convolution (we are adding previous values (aka "echo") for each point
    # rir = torch.flip(rir,[0])

    # Pad with zeros to match output time
    # waveforms = torch.cat((torch.zeros(rir.shape[0]-1,dtype=waveforms.dtype), waveforms), 0)

    # Calculate convolution
    waveforms = waveforms.unsqueeze(0).unsqueeze(0)
    rir = rir.unsqueeze(0).unsqueeze(0)
    # waveforms = torch.nn.functional.conv1d(waveforms, rir)
    waveforms = torchaudio.functional.fftconvolve(waveforms, rir)
    waveforms = waveforms.squeeze(dim=0).squeeze(dim=0)
    waveforms = waveforms[0:source_len]

    # Sometimes it overflows
    max_v = waveforms.max().abs()
    if max_v > 0.99:
        waveforms = (waveforms / max_v) * 0.98

    return waveforms

def do_noise(waveforms, noise, noise_level=0.1):
    
    # Create noise
    # noise = torch.randn_like(waveforms)

    # Compute energy
    energy = torch.sqrt(torch.mean(waveforms**2))
    noise_energy = torch.sqrt(torch.mean(noise**2))

    # Normalize noise
    noise = noise * (energy / noise_energy)

    # Apply noise
    return waveforms * (1 - noise_level) + noise * noise_level

def do_background(signal, bg, snr):

    # Calculate offsets
    signal_offset = 0
    bg_offset = 0
    l = bg.shape[0]
    if bg.shape[0] < signal.shape[0]:
        signal_offset = random.randint(0, signal.shape[0] - bg.shape[0])
        bg_offset = 0
        l = bg.shape[0]
    elif bg.shape[0] > signal.shape[0]:
        signal_offset = 0
        bg_offset = random.randint(0, bg.shape[0] - signal.shape[0])
        l = signal.shape[0]

    # Calculate noise tensor
    noise = torch.zeros_like(signal)
    noise[signal_offset:signal_offset + l] = bg[bg_offset:bg_offset + l]

    # Apply noise
    updated = torchaudio.functional.add_noise(signal.unsqueeze(0), noise.unsqueeze(0), torch.tensor([snr]))[0]

    # Sometimes it returns NaN - ignore noise then (when noise is zero)
    if torch.isnan(updated).any():
        return signal
    else:
        return updated

def do_lost_packets(waveforms, lost_segments, min_segment, max_segment):
    for _ in range(lost_segments):
        start = random.randint(0, waveforms.shape[0] - min_segment)
        end = start + random.randint(min_segment, max_segment)
        waveforms[start:end] = 0
    return waveforms