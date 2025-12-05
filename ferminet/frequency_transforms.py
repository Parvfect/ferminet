
import jax.numpy as jnp
import wandb


def get_dominant_frequencies(x, dt, k=5, use_radians=False):
    x = jnp.array(x)
    x = x - jnp.mean(x) # centering
    N = len(x)
    # FFT
    X = jnp.fft.fft(x)
    freqs = jnp.fft.fftfreq(N, dt)      # Hz

    # Only positive frequencies (ignore DC separately)
    pos = freqs > 0
    freqs = freqs[pos]
    amps = jnp.abs(X[pos])

    # Remove DC-like offset if small f=0 is present (optional)
    # But since pos>0, we already skipped f=0

    # Sort by amplitude
    idx = jnp.argsort(amps)[::-1]       # descending
    top_idx = idx[:k]

    # cast explicitly to float dtype
    freqs = jnp.real(freqs).astype(float)
    amps = jnp.real(amps).astype(float)

    # Return frequencies + amplitudes
    if use_radians:
        return 2*jnp.pi*freqs[top_idx], amps[top_idx]
    else:
        return freqs[top_idx], amps[top_idx]
    
