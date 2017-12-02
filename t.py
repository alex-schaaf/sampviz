import sampviz
import scipy.stats

dist1 = scipy.stats.norm(0, 1)
dist2 = scipy.stats.norm(0, 1)

sv = sampviz.SampViz(dist1, dist2)

samples = sv.sample_monte_carlo(200)

print(samples.shape)

# sv.jointplot(samples)
sv.run(samples)  # , save="figures/normal_normal_mc_sampling.mp4")