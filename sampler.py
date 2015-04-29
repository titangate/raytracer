import random
import numpy
import math
class RegularSampler(object):
    def sample(self):
        return ((0.5,0.5),)

    def sample_unit_disk(self):
        return ((0.,0.),)

    def sample_unit_hemisphere_surface(self):
        return ((1.,0.,0.),)

class MultiJitteredSampler(object):
    def __init__(self, sample_dim=2, pattern_size=83, e=1):
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
                concentric_samples.append(numpy.array(r * numpy.cos(phi), r * numpy.sin(phi)))
            concentric_patterns.append(concentric_samples)
        self.concentric_patterns = concentric_patterns


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
        self.hemisphere_patterns = hemisphere_samples

    def sample(self):
        return random.choice(self.patterns)

    def sample_unit_disk(self):
        return random.choice(self.concentric_patterns)

    def sample_unit_hemisphere_surface(self):
        return random.choice(self.hemisphere_patterns)