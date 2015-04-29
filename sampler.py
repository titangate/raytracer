import random
import numpy
import math
class RegularSampler(object):
    def sample(self):
        return ((0.5,0.5),)

class MultiJitteredSampler(object):
    def __init__(self, sample_dim=2, pattern_size=83):
        self.sample_dim = sample_dim
        self.pattern_size = pattern_size
        self.patterns = []
        for i in xrange(pattern_size):
            samples = []
            idx_to_shuffle_row = range(sample_dim)
            idx_to_shuffle_col = range(sample_dim)
            random.shuffle(idx_to_shuffle_col)
            random.shuffle(idx_to_shuffle_row)
            dim = float(sample_dim)
            for i in xrange(sample_dim):
                for j in xrange(sample_dim):
                    samples.append(((i + random.uniform(0,1)) / dim ** 2 + idx_to_shuffle_row[i] / dim, (j + random.uniform(0,1)) / dim ** 2 + idx_to_shuffle_col[j] / dim))
            self.patterns.append(samples)

    def sample(self):
        return random.choice(self.patterns)

class ConcentricMapSampler(MultiJitteredSampler):
    def __init__(self, *args, **kwargs):
        super(ConcentricMapSampler, self).__init__(*args, **kwargs)
        concentric_patterns = []
        for samples in self.patterns:
            concentric_samples = []
            for sample in samples:
                x = sample[0] * 2 - 1
                y = sample[1] * 2 - 1
                if x > -y:
                    if x > y:
                        r = x
                        phi = y / x
                    else:
                        r = y
                        phi = 2 - x / y
                else:
                    if x < y:
                        r = -x
                        phi = 4 + y / x
                    else:
                        r = -y
                        if y != 0:
                            # at origin
                            phi = 6 - x / y
                        else:
                            phi = 0
                phi *= numpy.pi / 4.0
                concentric_samples.append((r * numpy.cos(phi), r * numpy.sin(phi)))
            concentric_patterns.append(concentric_samples)
        self.patterns = concentric_patterns

class HemisphereSampler(MultiJitteredSampler):
    def __init__(self, sample_dim=2, pattern_size=83, e=1):
        super(HemisphereSampler, self).__init__(sample_dim, pattern_size)

        hemisphere_patterns = []
        for samples in self.patterns:
            hemisphere_samples = []
            for sample in samples:
                x, y = sample
                cos_phi = numpy.cos(2 * numpy.pi * x)
                sin_phi = numpy.sin(2 * numpy.pi * x)
                cos_theta = (1.0 - y) ** (1.0 / (e + 1.0))
                sin_theta = (1.0 - cos_theta * cos_theta) ** 0.5
                pu = sin_theta * cos_phi
                pv = sin_theta * sin_phi
                pw = cos_theta

                hemisphere_samples.append((pu, pv, pw))
            hemisphere_patterns.append(hemisphere_samples)
        self.patterns = hemisphere_samples