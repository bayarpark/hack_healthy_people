from scipy import stats
import numpy as np


class SNP:

    def __make_rv(self):
        # odds = norm(loc=1, scale=0.05).rvs()
        odds = np.exp(stats.norm(loc=(np.log(1.1) + np.log(0.9)) / 2,
                           scale=(np.log(1.1) - np.log(0.9)) / 2).rvs())
        noise = stats.norm(loc=0, scale=0.02).rvs()
        freq = 1.5 - abs(1 - odds) - 1 + noise
        if freq < 0.05:
            freq = 0.05
        if freq > 0.5:
            freq = 0.5

        self.rv = stats.rv_discrete(name='allel',
                                    values=([0, 1, 2], [(1 - freq) ** 2, 2 * freq * (1 - freq), freq ** 2]))
        self.freq = freq
        self.odds = odds

    def __init__(self, snp_name):
        self.name = snp_name
        self.__make_rv()

    def rvs(self):
        self.val = self.rv.rvs()
        if self.val == 0:
            f0, f1, f2 = 1, 0, 0
        elif self.val == 1:
            f0, f1, f2 = 0, 1, 0
        else:
            f0, f1, f2 = 0, 0, 1

        return {f'{self.name}_f0': f0, f'{self.name}_f1': f1, f'{self.name}_f2': f2}

    def coeff(self):
        if self.val == 1:
            return self.odds
        elif self.val == 2:
            return self.odds**2
        else:
            return 1
