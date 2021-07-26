import pandas as pd
from scipy import stats
import itertools
from snp_generator import *
from scipy.stats import bernoulli


def age2interval(pd_age_series):
    return pd_age_series.apply(
        lambda x: pd.Interval(left=int(x[:x.find('-')]), right=int(x[x.find('-') + 1:]), closed='both'))


nw1000m = pd.read_csv('data/nw1000m.csv')
nw1000m['age'] = age2interval(nw1000m['age'])
nw1000m['female_proba'] = nw1000m['value'] / (nw1000m['value'] + 1000)

age_proba = pd.read_csv('data/age_proba.csv')
age_variable = stats.rv_discrete(name='russianage', values=(age_proba['age'], age_proba['proba']))

age_sex_risk = pd.read_csv('data/age_sex_risk.csv')
age_sex_risk['age'] = age2interval(age_sex_risk['age'])


def cad_age_sex(age, sex):
    risk_age = age_sex_risk[age_sex_risk['age'].apply(lambda x: age in x)]
    return float(risk_age[risk_age['sex'] == sex]['risk'])


def cad_smoking(smoking):
    return 1.3 if smoking == 1 else 1


smoking = pd.read_csv('data/smoking.csv')
smoking['age'] = age2interval(smoking['age'])


def smoking_proba_age_sex(age, sex):
    """
    age: возраст
    sex: 1 - муж., 0 - жен

    return: 1 - курит, 0 - не курит (с учетом пола и возраста)
    """
    smoking_age = smoking[smoking['age'].apply(lambda x: age in x)]
    smoking_proba = float(smoking_age[smoking_age['sex'] == sex]['smoking'])

    rv = stats.bernoulli(smoking_proba)
    return rv.rvs(), smoking_proba


all_SNPs = [SNP(f'snp_{i}') for i in range(1000)]

p = []
class People:
    risks = dict()

    def gen_age(self):
        age = lambda: q if (q := age_variable.rvs()) >= 20 else age()
        self.age = age()

        # self.risks['age'] = cad_age(self.age)
        # self.risk = self.risks['age']

    def gen_sex(self):
        fem_prob = float(nw1000m[nw1000m['age'].apply(lambda x: self.age in x)]['female_proba'])
        rv = bernoulli(1 - fem_prob)
        self.sex = rv.rvs()

        self.risks['age_sex'] = cad_age_sex(self.age, self.sex)
        self.risk = cad_age_sex(self.age, self.sex)

    def gen_smoking(self):
        self.smoking, smoking_proba = smoking_proba_age_sex(self.age, self.sex)
        self.risks['smoking'] = cad_smoking(self.smoking)

        self.risk *= 1.3 if self.smoking == 1 else 1  # (1.3 * smoking_proba + (1 - smoking_proba)) if self.smoking == 1 else 1

    def gen_overweight(self):
        overweight_proba = 0.247 if self.sex == 0 else 0.179
        rv = bernoulli(overweight_proba)
        self.overweight = rv.rvs()
        self.risk *= 2 if self.overweight == 1 else 1  # (2 * overweight_proba + (1 - overweight_proba)) if self.overweight == 1 else 1

    def gen_snps(self):
        snps_rvs = [snp.rvs() for snp in all_SNPs]
        for snp in all_SNPs:
            p.append(snp.coeff())
            self.risk *= snp.coeff()
        self.snps = dict(itertools.chain.from_iterable(dct.items() for dct in snps_rvs))

    def calc_target(self):
        rv = bernoulli(self.risk)
        self.target = rv.rvs()


    def to_series(self):
        return pd.Series(
            {
                  'age': self.age
                , 'sex': self.sex
                , 'smoking': self.smoking
                , 'overweight': self.overweight
                , **self.snps
                , 'overall_risk': self.risk
                , 'disease (target)': self.target
             })

    def gen_all(self):
        self.gen_age()
        self.gen_sex()
        self.gen_smoking()
        self.gen_overweight()
        # self.gen_diabetes()
        self.gen_snps()
        self.calc_target()
