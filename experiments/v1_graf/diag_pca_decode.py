"""Diagnostic: is orientation present in the projection-encoder's feature at all?

If a batch PCA-3 of the per-trial log-rate decodes orientation well but sVJF's
filtered latent gives chance, then sVJF's filtering (not the readout/feature) is
losing the orientation information. Local, no VJF, ~30 s.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from experiments.v1_graf.graf_loader import load_array, bin_spikes, well_tuned_mask, STIM_MS

d = load_array(5)
counts = bin_spikes(d["spk_times"], bin_ms=10.0)        # (3600, 256, N)
dirs = d["ori"]
mask, _ = well_tuned_mask(counts, dirs)
counts = counts[:, :, mask]
nstim = int(STIM_MS / 10.0)

# per-trial stimulus-window mean of the link-matched log-rate feature
feat = np.log(counts[:, :nstim, :].mean(1) + 1e-2)      # (3600, Nkept)
y = np.round(dirs).astype(int)
clf = LogisticRegression(max_iter=3000)

acc_full = cross_val_score(clf, feat, y, cv=5).mean()
p3 = PCA(3).fit_transform(feat - feat.mean(0))
acc_pca3 = cross_val_score(clf, p3, y, cv=5).mean()
# also raw spike-count mean (no log), as a sanity reference
raw = counts[:, :nstim, :].mean(1)
acc_raw = cross_val_score(clf, raw, y, cv=5).mean()

print(f"Nkept={int(mask.sum())}  chance=1/72={1/72:.4f}")
print(f"decode acc, full log-rate feature ({feat.shape[1]}-d) = {acc_full:.3f}")
print(f"decode acc, PCA-3 of log-rate feature                = {acc_pca3:.3f}")
print(f"decode acc, raw mean counts ({raw.shape[1]}-d)        = {acc_raw:.3f}")
print(f"sVJF L=3 filtered-latent decode (reported)           = 0.014")
