import re

with open("lightweight_train/env_3d.py", "r") as f:
    content = f.read()

content = content.replace("W_ALT_IMBALANCE = -0.25\nALT_IMBALANCE_THRESH = 0.10", "W_ABANDON_PENALTY = -1.0\nW_SWITCH_BAD_PENALTY = -50.0")
content = re.sub(
    r"reward_altitude -= 25\.0(.*?)",
    r"reward_altitude += W_SWITCH_BAD_PENALTY\1",
    content,
    flags=re.DOTALL | re.MULTILINE,
    count=1
)

imbalance_block = """        all_alt_covs = [self._alt_cov(alt) for alt in range(N_ALTITUDES)]
        imbalance_gap = all_alt_covs[self.alt_idx] - min(all_alt_covs)
        if imbalance_gap > ALT_IMBALANCE_THRESH:
            reward_altitude += W_ALT_IMBALANCE * imbalance_gap"""

new_block = """        all_alt_covs = [self._alt_cov(alt) for alt in range(N_ALTITUDES)]
        incomplete_alts = [alt for alt, cov in enumerate(all_alt_covs) if 0.05 < cov < 0.90]
        if len(incomplete_alts) > 0 and self.alt_idx not in incomplete_alts:
            reward_altitude += W_ABANDON_PENALTY"""

content = content.replace(imbalance_block, new_block)

with open("lightweight_train/env_3d.py", "w") as f:
    f.write(content)
