# Convex hull selection

## Why use a convex hull?

A spectral image can contain many thousands of pixel spectra. To find useful
starting spectra, Lumic first looks for pixels at the extremes of their
spectral variation. A convex hull is the smallest convex shape enclosing a
set of points: in two dimensions, imagine stretching a rubber band around the
outermost points. Its corners, called vertices, provide a smaller candidate
pool when relatively few pixels lie on the boundary. This reduction is not
guaranteed: many pixels can be hull vertices, especially with noise or more
dimensions.

Lumic does not build the hull directly from the original channel values. It
first projects the spectra into `npc` SVD components, divides the remaining
coordinates by the first coordinate, and builds the hull in the resulting
`npc - 1` dimensional space. Every projected and normalized point inside
that hull can be written as a **convex combination** of its vertices: weights
are nonnegative and sum to one. This relationship holds in the projected,
normalized space. Because projection discards information, it does not
guarantee that the full original spectrum can be reconstructed exactly from
the original spectra at the hull vertices.

The hull returns the original pixel spectra at those vertices. The next
step, `unique_profiles`, chooses a requested number of distinct starting
spectra from the candidates. These starts can be used for unmixing with fixed
spectra or refined with MCR-ALS. A hull vertex is an extreme **measured**
spectrum; it is not necessarily a pure fluorophore. Background, mixtures,
noise and artifacts can all affect which pixels are selected.

## How many hull dimensions?

`npc` is the number of SVD components used for the projection; it is not
itself a measurement of the number of fluorophores. The hull has `npc - 1`
dimensions. It needs enough points that occupy those dimensions: if the
projected points lie in a lower-dimensional shape, the hull calculation can
fail with a numerical precision error. Increasing the dimension can also
make the calculation much slower or require much more memory.

In an idealized [synthetic signature evaluation](https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/evaluations/reference_results/hull_signatures/summary.md),
we sampled 3, 4, 6, 8 and 10 distinct Gaussian spectra through an approximate
32-channel LSM 980 detector. The noiseless data included a pure pixel for
each signature. In that run, the hull completed when `npc` matched the number
of signatures; asking for one extra component caused a Qhull precision error.
This illustrates the geometry for those favorable inputs, rather than a
guaranteed limit for every data set.

Do not use hull success or failure alone to infer how many fluorophores are
present. Noise can add variation in directions that do not correspond to
additional fluorophores. In a separate [FPbase fluorophore evaluation](https://github.com/CCI-GU-Sweden/lumic/blob/main/Python/evaluations/reference_results/hull_fluorophores/summary.md),
three simulated fluorophores produced a hull precision error above `npc=3`
without noise. With small added noise, the hull completed at `npc=8`, but
selected 125 of 150 pixels as vertices; higher values exceeded the run's
eight-second time limit. A timeout depends on the computer and does not mean
that a hull is impossible. Spectral overlap, background and missing pure pixels
also affect the interpretation of the candidates.

These evaluations examine candidate selection, not the accuracy of unmixing
or the maximum number of fluorophores recoverable from an image. When suitable
reference spectra are available, fixed-reference unmixing can be used without
the convex-hull selection step. Assess a proposed component count using
independent knowledge and fit diagnostics; a small residual alone does not
prove that the recovered spectra or abundances are correct.

